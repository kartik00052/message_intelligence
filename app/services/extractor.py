"""Task / event / reminder / meeting extraction.

Flow:
    sanitized message
        -> deterministic rule extraction (calendar, reminder, schedule, ...)
        -> if nothing matched, optional LLM fallback
        -> structured, Pydantic-validated items

Design rule: a field is only populated when it is explicitly stated in the
message. Nothing is guessed or fabricated. The extractor only ever sees the
masked/sanitized message, so no raw sensitive value can reach an item or an
external provider.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Final

from app.models.message import RawMessage
from app.models.task_event import (
    ExtractedItem,
    ExtractionResult,
    ExtractorMethod,
    ItemType,
    Priority,
)
from app.services.masker import Masker
from app.services.sensitive_detector import SensitiveDetector

# A time as it appears in a message: "9", "9:00", "9 AM", "8pm", "14:30".
_TIME_RAW = r"\d{1,2}(?::\d{2})?\s*(?:am|pm)?"
_ISO_DATE_RAW = r"\d{4}-\d{2}-\d{2}"

_RELATIVE_DAY_RAW = r"(?:today|tomorrow|yesterday)"
_RELATIVE_WEEKDAY_RAW = (
    r"(?:(?:next|this)\s+)?"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
)
_RELATIVE_N_DAYS_RAW = (
    r"in\s+(?:a\s+day|\d+\s+days?"
    r"|(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+days?)"
)
_RELATIVE_DATE_RAW = rf"(?:{_RELATIVE_DAY_RAW}|{_RELATIVE_WEEKDAY_RAW}|{_RELATIVE_N_DAYS_RAW})"

# A date as it appears in a message: either an explicit ISO date or a relative
# expression that can be resolved against the message timestamp.
_DATE_RAW = rf"(?:{_ISO_DATE_RAW}|{_RELATIVE_DATE_RAW})"

_CALENDAR_RE = re.compile(
    r"calendar\s+update:\s*(?P<title>[^,]+?)\s*,\s*"
    r"(?P<date>" + _DATE_RAW + r")\s+at\s+(?P<time>" + _TIME_RAW + r")"
    r"(?:\s*,\s*(?P<loc>[^.!?\n]+))?",
    re.IGNORECASE,
)

_REMINDER_RE = re.compile(
    r"reminder:\s*(?P<title>[^,]+?)\s+happens\s+on\s+"
    r"(?P<date>" + _DATE_RAW + r")\s+at\s+(?P<time>" + _TIME_RAW + r")"
    r"\s+in\s+(?P<loc>[^.!?\n]+)",
    re.IGNORECASE,
)

_SCHEDULED_RE = re.compile(
    r"the\s+(?P<title>[^,]+?)\s+is\s+scheduled\s+for\s+"
    r"(?P<date>" + _DATE_RAW + r")\s+at\s+(?P<time>" + _TIME_RAW + r")"
    r"\s+in\s+(?P<loc>[^.!?\n]+)",
    re.IGNORECASE,
)

_JOIN_RE = re.compile(
    r"please\s+join\s+the\s+(?P<title>[^,]+?)\s*(?:on\s+)?"
    r"(?P<date>" + _DATE_RAW + r")\s*(?:,|at)?\s*(?P<time>" + _TIME_RAW + r")"
    r"\s+at\s+(?P<loc>[^.!?\n]+)",
    re.IGNORECASE,
)

_AVAILABLE_RE = re.compile(
    r"are\s+you\s+available\s+for\s+the\s+(?P<title>[^,]+?)\s+at\s+"
    r"(?P<time>" + _TIME_RAW + r")\s*(?:on\s+)?(?P<date>" + _DATE_RAW + r")"
    r"\s*\??\s*(?:location:\s*(?P<loc>[^.!?\n]+))?",
    re.IGNORECASE,
)

# Scheduled items where no explicit time is stated. The time stays null - it is
# never guessed.
_CALENDAR_NO_TIME_RE = re.compile(
    r"calendar\s+update:\s*(?P<title>[^,]+?)\s*,\s*(?P<date>" + _DATE_RAW + r")"
    r"(?:\s*,\s*(?P<loc>[^.!?\n]+))?",
    re.IGNORECASE,
)

_REMINDER_NO_TIME_RE = re.compile(
    r"reminder:\s*(?P<title>[^,]+?)\s+happens\s+on\s+"
    r"(?P<date>" + _DATE_RAW + r")(?:\s+in\s+(?P<loc>[^.!?\n]+))?",
    re.IGNORECASE,
)

_SCHEDULED_NO_TIME_RE = re.compile(
    r"the\s+(?P<title>[^,]+?)\s+is\s+scheduled\s+for\s+"
    r"(?P<date>" + _DATE_RAW + r")(?:\s+in\s+(?P<loc>[^.!?\n]+))?",
    re.IGNORECASE,
)

_DEADLINE_RE = re.compile(
    r"\b(?:deadline\s+is|is\s+due\s+on|due\s+on|by|before)\s+"
    r"(?P<date>" + _DATE_RAW + r")\b",
    re.IGNORECASE,
)

_CALL_NAME_RE = re.compile(
    r"\b(?:please\s+)?(?:call|contact|reach\s+out\s+to)\s+(?P<person>[A-Z][a-z]+)\b"
)

_CALL_TARGET_RE = re.compile(r"\bcall\s+the\s+(?P<target>[a-z][a-z ]{2,30})\b")

_HIGH_PRIORITY_RE = re.compile(
    r"\b(?:urgent|asap|critical|immediately|as\s+soon\s+as\s+possible)\b", re.IGNORECASE
)
_LOW_PRIORITY_RE = re.compile(
    r"\b(?:no\s+rush|no\s+hurry|no\s+urgency|when\s+you\s+are\s+free|whenever|"
    r"at\s+your\s+convenience|when\s+you\s+have\s+time)\b",
    re.IGNORECASE,
)

_MESSAGE_PREFIX_RE = re.compile(
    r"^(?:for\s+today\s*:?\s*|quick\s+update\s*:?\s*|fyi\s*:?\s*"
    r"|important\s*:?\s*|can\s+you\s+help\s*\??\s*|please\s+note\s*:?\s*"
    r"|one\s+more\s+thing\s*:?\s*|just\s+checking[\s\u2014-]*|hi\s*,?\s*)",
    re.IGNORECASE,
)

_SOFTENER_RE = re.compile(
    r"^(?:please\s+|could\s+you\s+please\s+|can\s+you\s+please\s+|can\s+you\s+"
    r"|could\s+you\s+|are\s+you\s+able\s+to\s+|i\s+need\s+you\s+to\s+"
    r"|i\s+would\s+like\s+you\s+to\s+|i'?d\s+like\s+you\s+to\s+"
    r"|don'?t\s+forget\s+to\s+|do\s+not\s+forget\s+to\s+|make\s+sure\s+to\s+"
    r"|remember\s+to\s+|kindly\s+|please\s+make\s+sure\s+to\s+)",
    re.IGNORECASE,
)

_NO_ITEMS_REASON = "No actionable or schedulable signal detected."


@dataclass(frozen=True)
class RuleMatch:
    """Result of a successful deterministic extraction rule.

    Attributes:
        items: The extracted items (normally exactly one).
        reason: Short justification of the matched rule.
    """

    items: tuple[ExtractedItem, ...]
    reason: str


_WEEKDAY_INDEX: Final = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_WORD_NUMBER: Final = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def make_item_id(message_id: str, item_type: ItemType, occurrence: int = 1) -> str:
    """Deterministic item identifier derived from the message and item type.

    Uses the ``<TYPE>_<message_id>`` form, e.g. ``TASK_MSG_0002``. When a single
    message yields several items of the same type they are indexed with a
    ``-N`` suffix, e.g. ``TASK_MSG_0002-2``.
    """
    base = f"{item_type.value.upper()}_{message_id}"
    if occurrence <= 1:
        return base
    return f"{base}-{occurrence}"


def resolve_relative_date(expr: str, reference: date) -> date | None:
    """Resolve a relative date expression against a reference date.

    Recognised expressions: ``today``, ``tomorrow``, ``yesterday``, bare or
    qualified weekday names (``friday``, ``this friday``, ``next friday``) and
    ``in <n> days`` (word or digit based). Weekday references resolve to the
    next occurrence at least one day ahead; ``this <weekday>`` may be today.

    The reference must be the message timestamp (never the current system date).

    Returns ``None`` for expressions that cannot be resolved.
    """
    normalized = " ".join(expr.strip().lower().split())
    if normalized == "today":
        return reference
    if normalized == "tomorrow":
        return reference + timedelta(days=1)
    if normalized == "yesterday":
        return reference - timedelta(days=1)

    if normalized.startswith("in "):
        amount = normalized[3:].strip()
        if amount in ("a day", "one day"):
            return reference + timedelta(days=1)
        if amount.isdigit():
            return reference + timedelta(days=int(amount))
        for word, number in _WORD_NUMBER.items():
            if amount == f"{word} days" or amount == f"{word} day":
                return reference + timedelta(days=number)
        return None

    match = re.fullmatch(r"((?:next|this)\s+)?([a-z]+)", normalized)
    if match is None:
        return None
    target = _WEEKDAY_INDEX.get(match.group(2))
    if target is None:
        return None
    days_ahead = (target - reference.weekday()) % 7
    qualifier = match.group(1)
    if days_ahead == 0 and qualifier != "this ":
        days_ahead = 7
    return reference + timedelta(days=days_ahead)


def normalize_time(raw: str) -> str:
    """Normalize a raw time expression to 24h ``HH:MM``.

    Accepts ``9``, ``9:00``, ``09:00``, ``9 AM``, ``8 pm``, ``14:30``.

    Raises:
        ValueError: if the raw value cannot be parsed as a valid time.
    """
    match = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*", raw, re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot parse time {raw!r}")
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)
    if meridiem is not None:
        meridiem = meridiem.lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
    elif hour == 24:
        hour = 0
    if hour > 23 or minute > 59:
        raise ValueError(f"time out of range: {raw!r}")
    return f"{hour:02d}:{minute:02d}"


def clean_clause(text: str) -> str:
    """Normalize a raw action clause into a short lowercase title.

    Strips message prefixes, polite softeners, trailing separators and lower
    cases the first character. Returns an empty string when nothing remains.
    """
    clause = text.split(";", 1)[0].strip(" \t.,;:!?\"'()[]")
    for _ in range(3):
        prefix = _MESSAGE_PREFIX_RE.match(clause)
        if prefix:
            clause = clause[prefix.end() :].strip(" \t.,;:!?\"'()[]")
    for _ in range(3):
        softener = _SOFTENER_RE.match(clause)
        if softener:
            clause = clause[softener.end() :].strip(" \t.,;:!?\"'()[]")
    for _ in range(3):
        separator = re.match(r"^(?:and\s+|also\s+|then\s+|plus\s+)", clause, re.IGNORECASE)
        if separator:
            clause = clause[separator.end() :].strip(" \t.,;:!?\"'()[]")
    if not clause:
        return ""
    return clause[0].lower() + clause[1:]


class ExtractionRules:
    """Deterministic extraction rules over sanitized message text.

    The rules are evaluated in a fixed precedence order. A message describing a
    single actionable or scheduled item produces exactly one item; a message
    with several distinct explicit deadlines produces one task per deadline.
    Fields are only set when explicitly present. Relative dates are resolved
    against the message timestamp (``reference_date``), never the system date.
    """

    def extract(
        self,
        message_id: str,
        safe_message: str,
        reference_date: date | None = None,
    ) -> RuleMatch | None:
        """Run the rules against ``safe_message``; return the first match."""
        text = safe_message.strip()
        if not text:
            return None

        rules: tuple[tuple[re.Pattern[str], ItemType, str], ...] = (
            (_CALENDAR_RE, ItemType.EVENT, "calendar-update"),
            (_REMINDER_RE, ItemType.REMINDER, "scheduled reminder"),
            (_SCHEDULED_RE, ItemType.MEETING, "scheduled meeting"),
            (_JOIN_RE, ItemType.EVENT, "join request"),
            (_AVAILABLE_RE, ItemType.MEETING, "availability request"),
            (_CALENDAR_NO_TIME_RE, ItemType.EVENT, "calendar-update"),
            (_REMINDER_NO_TIME_RE, ItemType.REMINDER, "scheduled reminder"),
            (_SCHEDULED_NO_TIME_RE, ItemType.MEETING, "scheduled meeting"),
        )
        for pattern, item_type, label in rules:
            if match := pattern.search(text):
                return self._event_match(
                    message_id, item_type, match, label, reference_date
                )

        deadline_matches = list(_DEADLINE_RE.finditer(text))
        if len(deadline_matches) > 1:
            return self._multiple_deadline_match(
                message_id, text, deadline_matches, reference_date
            )
        if len(deadline_matches) == 1:
            return self._deadline_match(message_id, text, deadline_matches[0], reference_date)
        if match := _CALL_NAME_RE.search(text):
            return self._call_match(message_id, text, match)
        return None

    # ------------------------------------------------------------------ rules

    def _event_match(
        self,
        message_id: str,
        item_type: ItemType,
        match: re.Match[str],
        label: str,
        reference_date: date | None,
    ) -> RuleMatch:
        location = match.group("loc")
        description = _clean_value(location)
        time = match.groupdict().get("time")
        item = ExtractedItem(
            item_id=make_item_id(message_id, item_type),
            type=item_type,
            title=_clean_value(match.group("title")),
            description=description or None,
            date=_parse_date_expr(match.group("date"), reference_date),
            deadline=None,
            time=normalize_time(time) if time else None,
            person=None,
            priority=_detect_priority(match.string),
            source_message_id=message_id,
        )
        return RuleMatch(
            items=(item,),
            reason=f"Matched a {label} with an explicit date"
            f"{' and time' if time else ''}.",
        )

    def _deadline_match(
        self,
        message_id: str,
        text: str,
        match: re.Match[str],
        reference_date: date | None,
    ) -> RuleMatch:
        clause = clean_clause(text[: match.start()])
        title = clause or "task"
        item = ExtractedItem(
            item_id=make_item_id(message_id, ItemType.TASK),
            type=ItemType.TASK,
            title=title,
            description=None,
            date=None,
            deadline=_parse_date_expr(match.group("date"), reference_date),
            time=None,
            person=None,
            priority=_detect_priority(text),
            source_message_id=message_id,
        )
        return RuleMatch(items=(item,), reason="Matched a task with an explicit deadline.")

    def _multiple_deadline_match(
        self,
        message_id: str,
        text: str,
        matches: list[re.Match[str]],
        reference_date: date | None,
    ) -> RuleMatch:
        items: list[ExtractedItem] = []
        priority = _detect_priority(text)
        for index, match in enumerate(matches):
            clause_start = matches[index - 1].end() if index else 0
            clause = clean_clause(text[clause_start : match.start()])
            items.append(
                ExtractedItem(
                    item_id=make_item_id(message_id, ItemType.TASK, index + 1),
                    type=ItemType.TASK,
                    title=clause or "task",
                    description=None,
                    date=None,
                    deadline=_parse_date_expr(match.group("date"), reference_date),
                    time=None,
                    person=None,
                    priority=priority,
                    source_message_id=message_id,
                )
            )
        return RuleMatch(
            items=tuple(items),
            reason=f"Matched {len(items)} distinct tasks with explicit deadlines.",
        )

    def _call_match(self, message_id: str, text: str, match: re.Match[str]) -> RuleMatch:
        person = match.group("person")
        title = f"call {person.lower()}"
        item = ExtractedItem(
            item_id=make_item_id(message_id, ItemType.TASK),
            type=ItemType.TASK,
            title=title,
            description=None,
            date=None,
            deadline=None,
            time=None,
            person=person,
            priority=_detect_priority(text),
            source_message_id=message_id,
        )
        return RuleMatch(
            items=(item,),
            reason=f"Matched a call request naming {person}.",
        )


# --------------------------------------------------------------- LLM interface


class LLMResponseError(Exception):
    """Raised when an LLM response cannot be turned into valid items."""


class MessageExtractorLLM(ABC):
    """Abstract interface for an LLM-based extraction provider.

    Implementations receive the sanitized message only. :meth:`extract`
    returns ``None`` (instead of raising) when the provider fails or returns
    something unusable, so the caller can fall back deterministically.
    """

    @abstractmethod
    def extract(
        self, *, message_id: str, safe_message: str
    ) -> tuple[ExtractedItem, ...] | None:
        """Return extracted items or ``None`` on any failure."""


class BaseLLMExtractor(MessageExtractorLLM, ABC):
    """Shared template for LLM providers: prompt building + robust parsing."""

    def extract(
        self, *, message_id: str, safe_message: str
    ) -> tuple[ExtractedItem, ...] | None:
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
        """Prompt instructing the model to return a JSON list of items."""
        return _LLM_EXTRACTION_PROMPT_TEMPLATE.format(
            message_id=message_id, safe_message=safe_message
        )

    @abstractmethod
    def _invoke(self, *, prompt: str) -> str:
        """Send the prompt to the provider and return the raw text response."""


_LLM_EXTRACTION_PROMPT_TEMPLATE: Final = (
    "You are a strict task and event extractor.\n"
    "Extract actionable or scheduled items from the message. An item is one of: "
    "task, meeting, event, reminder.\n"
    "Rules:\n"
    "- Only extract items that are clearly requested or explicitly scheduled.\n"
    "- type must be one of: task, meeting, event, reminder.\n"
    "- Do not invent information; leave date, deadline, time, person and "
    "priority as null when not explicitly stated.\n"
    "- date and deadline use YYYY-MM-DD; time uses 24h HH:MM.\n"
    "- title must be short and derived only from the message.\n"
    "- Sensitive values in the message have already been masked; do not "
    "reconstruct them.\n"
    "- Return an empty items list when nothing should be extracted.\n"
    "Respond with only JSON:\n"
    '{{"message_id": "<id>", "items": [{{"type": "<type>", "title": "<title>", '
    '"description": "<text or null>", "date": "<YYYY-MM-DD or null>", '
    '"deadline": "<YYYY-MM-DD or null>", "time": "<HH:MM or null>", '
    '"person": "<name or null>", "priority": "<low|medium|high|unknown>"}}]}}\n'
    "message_id: {message_id}\n"
    "message: {safe_message}"
)


def parse_llm_response(message_id: str, raw: str) -> tuple[ExtractedItem, ...]:
    """Parse a raw LLM response into validated items.

    Accepts either ``{"items": [...]}`` or a bare JSON list of item objects.

    Raises:
        LLMResponseError: if the response is empty, not JSON, misses the items
            field, contains an invalid item, or reports a different message_id.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text:
        raise LLMResponseError("empty response")

    data = _extract_json_object(text)
    if isinstance(data, dict):
        reported_id = data.get("message_id")
        if reported_id is not None and str(reported_id).strip() != message_id:
            raise LLMResponseError(
                f"message_id mismatch: expected {message_id!r}, got {reported_id!r}"
            )
        items = data.get("items")
    elif isinstance(data, list):
        items = data
    else:
        raise LLMResponseError("response is not a JSON object or list")

    if items is None:
        raise LLMResponseError("missing 'items' field in response")
    if not isinstance(items, list):
        raise LLMResponseError("'items' field is not a list")

    occurrences: dict[ItemType, int] = {}
    extracted: list[ExtractedItem] = []
    for position, raw_item in enumerate(items, start=1):
        if not isinstance(raw_item, dict):
            raise LLMResponseError(f"item #{position} is not an object")
        item = _coerce_item(message_id, raw_item, occurrences)
        extracted.append(item)
    return tuple(extracted)


def _coerce_item(
    message_id: str,
    raw: dict[str, Any],
    occurrences: dict[ItemType, int],
) -> ExtractedItem:
    """Build and validate one :class:`ExtractedItem` from a parsed LLM object."""
    item_type = _coerce_item_type(raw.get("type"))
    if item_type is None:
        raise LLMResponseError(f"invalid item type: {raw.get('type')!r}")

    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        raise LLMResponseError("item has an empty or missing title")

    count = occurrences.get(item_type, 0) + 1
    occurrences[item_type] = count

    return ExtractedItem(
        item_id=make_item_id(message_id, item_type, count),
        type=item_type,
        title=title.strip()[:120],
        description=_clean_value(raw.get("description")) or None,
        date=_coerce_date(raw.get("date"), "date"),
        deadline=_coerce_date(raw.get("deadline"), "deadline"),
        time=_coerce_time(raw.get("time")),
        person=_clean_value(raw.get("person")) or None,
        priority=_coerce_priority(raw.get("priority")),
        source_message_id=message_id,
    )


def _coerce_item_type(value: Any) -> ItemType | None:
    if isinstance(value, ItemType):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        try:
            return ItemType(normalized)
        except ValueError:
            return None
    return None


def _coerce_priority(value: Any) -> Priority:
    if isinstance(value, Priority):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return Priority(normalized)
        except ValueError:
            return Priority.UNKNOWN
    return Priority.UNKNOWN


def _coerce_date(value: Any, field: str) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise LLMResponseError(f"invalid {field}: {value!r}") from exc
    raise LLMResponseError(f"invalid {field}: {value!r}")


def _coerce_time(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            return normalize_time(value)
        except ValueError as exc:
            raise LLMResponseError(f"invalid time: {value!r}") from exc
    raise LLMResponseError(f"invalid time: {value!r}")


def _extract_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
        if not match:
            raise LLMResponseError("no JSON object found in response") from None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMResponseError("malformed JSON object in response") from exc


# ------------------------------------------------------------------ orchestrator


class MessageExtractor:
    """Pipeline entry point: mask -> rule extraction -> LLM fallback.

    The LLM fallback is only consulted when no deterministic rule matched. If
    no LLM is configured or it fails, an empty extraction result is returned
    (never a crash). Relative dates are resolved against the message timestamp.
    """

    def __init__(
        self,
        *,
        detector: SensitiveDetector | None = None,
        masker: Masker | None = None,
        rules: ExtractionRules | None = None,
        llm: MessageExtractorLLM | None = None,
    ) -> None:
        self._detector = detector or SensitiveDetector()
        self._masker = masker or Masker()
        self._rules = rules or ExtractionRules()
        self._llm = llm
        self._llm_failures = 0

    @property
    def llm_failures(self) -> int:
        """Number of failed or unusable LLM fallback calls so far."""
        return self._llm_failures

    def extract(self, message: RawMessage) -> ExtractionResult:
        """Extract items from a single raw message without leaking values."""
        detections = tuple(self._detector.detect(message.message))
        safe_message = self._masker.mask(message.message, detections)
        return self.extract_from_safe(
            message.message_id,
            safe_message,
            reference_date=message.timestamp.date(),
        )

    def extract_from_safe(
        self,
        message_id: str,
        safe_message: str,
        reference_date: date | None = None,
    ) -> ExtractionResult:
        """Extract items from an already-sanitized message.

        ``reference_date`` is the message timestamp; relative expressions are
        resolved against it. The current system date is never used.
        """
        match = self._rules.extract(message_id, safe_message, reference_date)
        if match is not None:
            return ExtractionResult(
                message_id=message_id,
                items=match.items,
                method=ExtractorMethod.RULE_BASED,
                reason=match.reason,
            )

        if self._llm is not None:
            try:
                items = self._llm.extract(message_id=message_id, safe_message=safe_message)
            except Exception:
                self._llm_failures += 1
                items = None
            if items:
                return ExtractionResult(
                    message_id=message_id,
                    items=items,
                    method=ExtractorMethod.LLM_FALLBACK,
                    reason=f"LLM fallback extracted {len(items)} item(s).",
                )
            self._llm_failures += 1

        return ExtractionResult(
            message_id=message_id,
            items=(),
            method=ExtractorMethod.NONE,
            reason=_NO_ITEMS_REASON,
        )


# ---------------------------------------------------------------------- helpers


def _parse_date_expr(value: str, reference_date: date | None) -> date | None:
    """Parse an explicit ISO date or resolve a relative one.

    Relative expressions require a reference date; when it is missing they
    resolve to ``None`` (nothing is guessed).
    """
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return date.fromisoformat(value)
    if reference_date is None:
        return None
    return resolve_relative_date(value, reference_date)


def _clean_value(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip(" \t.,;:!?\u201d\u201c\"'()[]")


def _detect_priority(text: str) -> Priority:
    if _HIGH_PRIORITY_RE.search(text):
        return Priority.HIGH
    if _LOW_PRIORITY_RE.search(text):
        return Priority.LOW
    return Priority.UNKNOWN
