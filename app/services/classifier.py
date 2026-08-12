"""Message classification.

Flow:
    raw message
        -> sensitive detections (provided by caller)
        -> deterministic rule classification
        -> confidence evaluation
        -> accept if confident, otherwise LLM fallback
        -> structured, Pydantic-validated result

The LLM (when configured) only ever receives the masked/sanitized message, so
no raw sensitive value can reach an external provider.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Final

from app.models.classification import (
    Category,
    ClassificationResult,
    ClassifierMethod,
)
from app.models.message import RawMessage
from app.models.sensitive import RiskLevel, SensitiveDetection
from app.services.masker import Masker
from app.services.sensitive_detector import SensitiveDetector

_RULE_CONFIDENCE_FLOOR: Final = 0.65
_GENERAL_CONFIDENCE: Final = 0.5
_GENERAL_REASON: Final = "Informational statement without action, event or offer."

_REASONS: dict[Category, str] = {
    Category.ACTION_REQUIRED: "Request for action: task verb, direct ask or deadline.",
    Category.MEETING_OR_EVENT: "Meeting or event with schedule, date, time or location signals.",
    Category.PERSONAL_INFORMATION: "Personal fact, preference or profile detail.",
    Category.PROMOTIONAL: "Promotional content: offer, sale or promo code.",
    Category.GENERAL_INFORMATION: _GENERAL_REASON,
}

_ACTION_PHRASES: tuple[str, ...] = (
    "please submit",
    "please review",
    "please update",
    "please reply",
    "please confirm",
    "please send",
    "please share",
    "please call",
    "please arrange",
    "please provide",
    "please approve",
    "please help",
    "please let me know",
    "please make sure",
    "please join",
    "need you to",
    "submit by",
    "complete by",
    "reply by",
    "respond by",
    "due on",
    "is due",
    "deadline",
    "remember to",
    "don't forget",
    "dont forget",
    "renew the",
    "pay the",
    "review the",
    "update the",
    "fix the",
    "check the",
    "verify the",
    "sign the",
    "send the",
    "submit the",
    "complete the",
    "approve the",
    "share the",
    "fill the",
    "make sure to",
    "email the",
    "send it",
)

_WEAK_ACTION_PHRASES: tuple[str, ...] = (
    "can you",
    "could you",
    "are you able to",
    "are you available",
)

_MEETING_STRONG_NOUNS: tuple[str, ...] = (
    "calendar",
    "dinner",
    "lunch",
    "seminar",
    "workshop",
    "interview",
    "briefing",
    "orientation",
    "appointment",
    "stand-up",
    "standup",
    "catch-up",
    "catchup",
    "retreat",
    "conference",
    "celebration",
    "scheduled",
    "schedule",
    "demo",
    "event",
    "session",
    "meeting",
    "sync",
    "huddle",
)

_MEETING_WEAK_NOUNS: tuple[str, ...] = (
    "meet",
    "review",
    "reunion",
    "gathering",
)

_MEETING_CONTEXT_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\bat\s+\d{1,2}:\d{2}\b"
    r"|\bat\s+\d{1,2}\s*(?:am|pm)\b"
    r"|\btomorrow\b"
    r"|\bnext\s+week\b"
    r"|\bhappens\s+on\b"
    r"|\bmeeting\s+room\b"
    r"|\bconference\s+room\b"
    r"|\bauditorium\b"
    r"|\btraining\s+hall\b"
    r"|\bmain\s+office\b"
    r"|\bcity\s+clinic\b"
    r"|\blocation:?\b"
    r"|\bthe\s+library\b"
    r"|\bclassroom\b",
    re.IGNORECASE,
)

_DEADLINE_RE = re.compile(
    r"\b(?:by|before)\s+\d{4}-\d{2}-\d{2}\b"
    r"|\bdeadline\b"
    r"|\bdue\s+on\b"
    r"|\bis\s+due\b"
    r"|\basap\b",
    re.IGNORECASE,
)

_PROMO_PHRASES: tuple[str, ...] = (
    "discount",
    "sale",
    "offer",
    "deal",
    "coupon",
    "cashback",
    "promotion",
    "exclusive",
    "limited time",
    "buy one",
    "free delivery",
    "free shipping",
    "reward points",
    "flash sale",
    "festival",
    "weekend sale",
    "premium plan",
    "student plan",
    "upgrade your subscription",
    "subscribe",
    "buy now",
    "shop now",
    "save 30%",
    "save 25%",
    "% off",
    "get 25%",
    "get 50%",
    "use code",
    "you may like",
    "earn reward points",
)

_PROMO_CODE_RE = re.compile(
    r"\bcode\s+(?:SAVE|GET|FLAT|WELCOME|OFF|FEST|NEW|DEAL)[A-Z0-9]{0,6}\b"
)

_PERSONAL_PHRASES: tuple[str, ...] = (
    "my favourite",
    "my favorite",
    "my preference",
    "i prefer",
    "i might prefer",
    "i like",
    "i love",
    "i usually",
    "i always",
    "i drink",
    "i use dark mode",
    "my t-shirt size",
    "my shirt size",
    "my shoe size",
    "personal note",
    "for my profile",
    "my birthday",
    "my diet",
    "my hobby",
    "my favourite language",
    "emergency contact",
    "just so you know, i",
    "remember that i",
    "my age",
    "my hometown",
    "i was born",
    "i am vegetarian",
    "i am a vegetarian",
    "i prefer morning meetings",
    "i prefer evening meetings",
    "i play",
    "i read",
    "i want to",
    "i hope to",
)

_SENSITIVE_REASON = "Detected sensitive content; value masked before any external processing."


def _count_phrases(text: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for phrase in phrases if phrase in text)


class RuleClassifier:
    """Deterministic, context-aware rule based classifier.

    Operates on the sanitized message text plus the sensitive detections. It
    never sees an unmasked value.
    """

    def classify(
        self,
        message_id: str,
        safe_message: str,
        detections: tuple[SensitiveDetection, ...] = (),
    ) -> ClassificationResult:
        """Classify one sanitized message using deterministic rules."""
        if detections:
            return self._classify_sensitive(message_id, detections)
        return self._classify_text(message_id, safe_message)

    def _classify_sensitive(
        self, message_id: str, detections: tuple[SensitiveDetection, ...]
    ) -> ClassificationResult:
        types = ", ".join(sorted({det.sensitivity_type.value for det in detections}))
        high_risk = any(det.risk is RiskLevel.HIGH for det in detections)
        return ClassificationResult(
            message_id=message_id,
            category=Category.SENSITIVE_INFORMATION,
            confidence=0.97 if high_risk else 0.92,
            reason=f"{_SENSITIVE_REASON} Types: {types}.",
            method=ClassifierMethod.RULE_BASED,
        )

    def _classify_text(self, message_id: str, safe_message: str) -> ClassificationResult:
        text = safe_message.strip()
        lower = text.lower()

        scores: dict[Category, float] = {
            Category.MEETING_OR_EVENT: self._meeting_score(lower),
            Category.ACTION_REQUIRED: self._action_score(lower),
            Category.PROMOTIONAL: self._promo_score(lower),
            Category.PERSONAL_INFORMATION: self._personal_score(lower),
        }
        category = max(scores, key=lambda c: scores[c])
        confidence = scores[category]

        if confidence < _RULE_CONFIDENCE_FLOOR:
            category = Category.GENERAL_INFORMATION
            confidence = _GENERAL_CONFIDENCE

        return ClassificationResult(
            message_id=message_id,
            category=category,
            confidence=round(confidence, 2),
            reason=_REASONS[category],
            method=ClassifierMethod.RULE_BASED,
        )

    @staticmethod
    def _meeting_score(lower: str) -> float:
        strong = _count_phrases(lower, _MEETING_STRONG_NOUNS)
        weak = _count_phrases(lower, _MEETING_WEAK_NOUNS)
        context_hits = len(_MEETING_CONTEXT_RE.findall(lower))
        score = 0.5 + 0.13 * strong + 0.07 * weak + 0.08 * context_hits
        return min(score, 0.97)

    @staticmethod
    def _action_score(lower: str) -> float:
        strong = _count_phrases(lower, _ACTION_PHRASES)
        weak = _count_phrases(lower, _WEAK_ACTION_PHRASES)
        deadline = 1 if _DEADLINE_RE.search(lower) else 0
        score = 0.55 + 0.14 * strong + 0.05 * weak + 0.1 * deadline
        return min(score, 0.95)

    @staticmethod
    def _promo_score(lower: str) -> float:
        phrases = _count_phrases(lower, _PROMO_PHRASES)
        code = 1 if _PROMO_CODE_RE.search(lower) else 0
        score = 0.6 + 0.12 * phrases + 0.1 * code
        return min(score, 0.95)

    @staticmethod
    def _personal_score(lower: str) -> float:
        phrases = _count_phrases(lower, _PERSONAL_PHRASES)
        score = 0.6 + 0.1 * phrases
        return min(score, 0.9)


# --------------------------------------------------------------- LLM interface


class LLMResponseError(Exception):
    """Raised when an LLM response cannot be turned into a valid result."""


class MessageClassifierLLM(ABC):
    """Abstract interface for an LLM-based classifier provider.

    Implementations receive the sanitized message only. :meth:`classify`
    returns ``None`` (instead of raising) when the provider fails or returns
    something unusable, so the caller can fall back deterministically.
    """

    @abstractmethod
    def classify(
        self, *, message_id: str, safe_message: str
    ) -> ClassificationResult | None:
        """Return a classification or ``None`` on any failure."""


class BaseLLMClassifier(MessageClassifierLLM, ABC):
    """Shared template for LLM providers: prompt building + robust parsing."""

    def classify(
        self, *, message_id: str, safe_message: str
    ) -> ClassificationResult | None:
        prompt = self.build_prompt(message_id=message_id, safe_message=safe_message)
        try:
            raw_response = self._invoke(prompt=prompt)
        except Exception:
            return None
        if not raw_response or not raw_response.strip():
            return None
        try:
            return parse_llm_response(message_id=message_id, raw=raw_response)
        except LLMResponseError:
            return None

    def build_prompt(self, *, message_id: str, safe_message: str) -> str:
        """Prompt instructing the model to return exactly one category."""
        return _LLM_PROMPT_TEMPLATE.format(
            message_id=message_id, safe_message=safe_message
        )

    @abstractmethod
    def _invoke(self, *, prompt: str) -> str:
        """Send the prompt to the provider and return the raw text response."""


_LLM_PROMPT_TEMPLATE: Final = (
    "You are a strict message classifier.\n"
    "Choose exactly one category from: action_required, meeting_or_event, "
    "personal_information, general_information, promotional, sensitive_information.\n"
    "Rules:\n"
    "- Do not invent information; classify based only on the message content.\n"
    "- Preserve the message_id exactly.\n"
    "- confidence must be a number between 0 and 1.\n"
    "- reason must be a short single sentence.\n"
    "- Sensitive values in the message have already been masked; do not attempt "
    "to reconstruct them.\n"
    "- category and confidence are never null; use null only where allowed.\n"
    "Respond with only JSON:\n"
    '{{"message_id": "<id>", "category": "<one of the six>", '
    '"confidence": 0.0, "reason": "<short reason>"}}\n'
    "message_id: {message_id}\n"
    "message: {safe_message}"
)


def parse_llm_response(message_id: str, raw: str) -> ClassificationResult:
    """Parse a raw LLM response into a validated :class:`ClassificationResult`.

    Raises:
        LLMResponseError: if the response is empty, not JSON, misses the
            category, uses an unknown category, or reports a different
            message_id.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text:
        raise LLMResponseError("empty response")

    data = _extract_json_object(text)
    if not isinstance(data, dict):
        raise LLMResponseError("response is not a JSON object")

    category = _coerce_category(data.get("category"))
    if category is None:
        raise LLMResponseError(f"invalid category: {data.get('category')!r}")

    confidence = _coerce_confidence(data.get("confidence"))

    reason = data.get("reason")
    reason = reason.strip()[:300] if isinstance(reason, str) else ""

    reported_id = data.get("message_id")
    if reported_id is not None and str(reported_id).strip() != message_id:
        raise LLMResponseError(
            f"message_id mismatch: expected {message_id!r}, got {reported_id!r}"
        )

    return ClassificationResult(
        message_id=message_id,
        category=category,
        confidence=confidence,
        reason=reason,
        method=ClassifierMethod.LLM_FALLBACK,
    )


def _extract_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise LLMResponseError("no JSON object found in response") from None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMResponseError("malformed JSON object in response") from exc


def _coerce_category(value: Any) -> Category | None:
    if isinstance(value, Category):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        normalized = re.sub(r"[^a-z0-9_]", "", normalized)
        try:
            return Category(normalized)
        except ValueError:
            return None
    return None


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return min(max(confidence, 0.0), 1.0)


# -------------------------------------------------------------- orchestrator


class MessageClassifier:
    """Pipeline entry point: detect -> mask -> rule classify -> LLM fallback.

    The LLM fallback is only used when the rule-based confidence is below
    ``llm_confidence_threshold``. If no LLM is configured or it fails, the
    best deterministic rule result is returned (never a crash).
    """

    def __init__(
        self,
        *,
        detector: SensitiveDetector | None = None,
        masker: Masker | None = None,
        rule_classifier: RuleClassifier | None = None,
        llm: MessageClassifierLLM | None = None,
        llm_confidence_threshold: float = 0.75,
    ) -> None:
        if not 0.0 <= llm_confidence_threshold <= 1.0:
            raise ValueError("llm_confidence_threshold must be between 0 and 1")
        self._detector = detector or SensitiveDetector()
        self._masker = masker or Masker()
        self._rule_classifier = rule_classifier or RuleClassifier()
        self._llm = llm
        self._llm_confidence_threshold = llm_confidence_threshold

    def classify(self, message: RawMessage) -> ClassificationResult:
        """Classify a single raw message without leaking sensitive values."""
        detections = tuple(self._detector.detect(message.message))
        safe_message = self._masker.mask(message.message, detections)
        rule_result = self._rule_classifier.classify(
            message.message_id, safe_message, detections
        )
        if rule_result.confidence >= self._llm_confidence_threshold:
            return rule_result

        if self._llm is None:
            return rule_result

        try:
            llm_result = self._llm.classify(
                message_id=message.message_id, safe_message=safe_message
            )
        except Exception:
            return rule_result
        if llm_result is None:
            return rule_result
        return llm_result

    @property
    def llm_confidence_threshold(self) -> float:
        """Confidence below which the LLM fallback is consulted."""
        return self._llm_confidence_threshold
