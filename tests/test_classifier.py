"""Unit tests for deterministic rule classification and the LLM fallback path."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from app.models.classification import Category, ClassificationResult, ClassifierMethod
from app.models.message import RawMessage
from app.services.classifier import (
    BaseLLMClassifier,
    LLMResponseError,
    MessageClassifier,
    MessageClassifierLLM,
    parse_llm_response,
)


def make_message(text: str, message_id: str = "T001") -> RawMessage:
    """Build a valid ``RawMessage`` for tests."""
    return RawMessage(
        message_id=message_id,
        timestamp=datetime(2026, 9, 1, 8, 0, 0),
        sender="Meera",
        message=text,
    )


def json_payload(**overrides: object) -> str:
    """A well-formed LLM payload with sane defaults."""
    payload = {
        "message_id": "T001",
        "category": "general_information",
        "confidence": 0.7,
        "reason": "test reason",
    }
    payload.update(overrides)
    return json.dumps(payload)


class StubLLM(MessageClassifierLLM):
    """Deterministic LLM stub that records calls and honours a failure mode."""

    def __init__(self, *, payload: str | object, fail: bool = False) -> None:
        self._payload = payload
        self._fail = fail
        self.calls: list[tuple[str, str]] = []

    def classify(self, *, message_id: str, safe_message: str) -> ClassificationResult | None:
        self.calls.append((message_id, safe_message))
        if self._fail:
            return None
        payload = self._payload(message_id) if callable(self._payload) else self._payload
        return parse_llm_response(message_id, str(payload))


class RecordingLLM(BaseLLMClassifier):
    """Concrete provider used to inspect the built prompt."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.last_invoke: str | None = None

    def _invoke(self, *, prompt: str) -> str:
        self.last_prompt = prompt
        self.last_invoke = prompt
        return json_payload()


class TestRuleClassification:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (
                "Please submit the report by 2026-09-04.",
                Category.ACTION_REQUIRED,
            ),
            (
                "Can you review the privacy checklist before 2026-09-09?",
                Category.ACTION_REQUIRED,
            ),
            (
                "Don't forget to renew the subscription.",
                Category.ACTION_REQUIRED,
            ),
            (
                "Our team meeting is scheduled for 2026-09-16 at 11:00 in the city clinic.",
                Category.MEETING_OR_EVENT,
            ),
            (
                "Family dinner tomorrow at 10:00 at the library.",
                Category.MEETING_OR_EVENT,
            ),
            (
                "Exclusive offer: get 50% off this weekend.",
                Category.PROMOTIONAL,
            ),
            (
                "Use code SAVE17 for a limited time cashback deal.",
                Category.PROMOTIONAL,
            ),
            (
                "My favourite book is Dune and I prefer morning meetings.",
                Category.PERSONAL_INFORMATION,
            ),
            (
                "I am vegetarian and my t-shirt size is M.",
                Category.PERSONAL_INFORMATION,
            ),
        ],
    )
    def test_obvious_messages(self, message: str, expected: Category) -> None:
        result = MessageClassifier().classify(make_message(message))
        assert result.category is expected
        assert result.method is ClassifierMethod.RULE_BASED
        assert 0.0 <= result.confidence <= 1.0
        assert result.reason

    def test_general_informational_message(self) -> None:
        result = MessageClassifier().classify(
            make_message("The training material is on the portal.")
        )
        assert result.category is Category.GENERAL_INFORMATION
        assert result.method is ClassifierMethod.RULE_BASED

    def test_sensitive_message(self) -> None:
        result = MessageClassifier().classify(make_message("Your OTP is 482913."))
        assert result.category is Category.SENSITIVE_INFORMATION
        assert result.method is ClassifierMethod.RULE_BASED
        assert result.confidence >= 0.9
        assert "masked" in result.reason

    def test_message_id_preserved(self) -> None:
        result = MessageClassifier().classify(make_message("Please submit the report.", "MSG_9999"))
        assert result.message_id == "MSG_9999"


class TestLLMFallback:
    def test_ambiguous_message_consults_llm(self) -> None:
        llm = StubLLM(payload=json_payload(category="action_required", confidence=0.9))
        classifier = MessageClassifier(llm=llm, llm_confidence_threshold=0.75)
        result = classifier.classify(
            make_message("The weather forecast says rain tomorrow.")
        )
        assert llm.calls
        assert result.method is ClassifierMethod.LLM_FALLBACK
        assert result.category is Category.ACTION_REQUIRED

    def test_llm_unavailable_falls_back_to_rule_result(self) -> None:
        llm = StubLLM(payload=json_payload(), fail=True)
        classifier = MessageClassifier(llm=llm, llm_confidence_threshold=0.75)
        result = classifier.classify(
            make_message("The weather forecast says rain tomorrow.")
        )
        assert result.method is ClassifierMethod.RULE_BASED
        assert result.category is Category.GENERAL_INFORMATION
        assert llm.calls

    def test_llm_invalid_response_falls_back_to_rule_result(self) -> None:
        llm = StubLLM(payload='{"category": "bogus"}')
        classifier = MessageClassifier(llm=llm, llm_confidence_threshold=0.75)
        result = classifier.classify(
            make_message("The weather forecast says rain tomorrow.")
        )
        assert result.method is ClassifierMethod.RULE_BASED
        assert result.category is Category.GENERAL_INFORMATION

    def test_confident_rule_bypasses_llm(self) -> None:
        calls: list[str] = []

        class BoomLLM(MessageClassifierLLM):
            def classify(
                self, *, message_id: str, safe_message: str
            ) -> ClassificationResult | None:
                calls.append(message_id)
                return None

        classifier = MessageClassifier(llm=BoomLLM(), llm_confidence_threshold=0.75)
        result = classifier.classify(make_message("Please submit the report by 2026-09-04."))
        assert result.method is ClassifierMethod.RULE_BASED
        assert result.category is Category.ACTION_REQUIRED
        assert calls == []

    def test_high_threshold_triggers_llm(self) -> None:
        llm = StubLLM(payload=json_payload(category="general_information"))
        classifier = MessageClassifier(llm=llm, llm_confidence_threshold=0.75)
        result = classifier.classify(
            make_message("The weather forecast says rain tomorrow.")
        )
        assert result.method is ClassifierMethod.LLM_FALLBACK

    def test_low_threshold_accepts_rule_result(self) -> None:
        llm = StubLLM(payload=json_payload(category="promotional"))
        classifier = MessageClassifier(llm=llm, llm_confidence_threshold=0.5)
        result = classifier.classify(
            make_message("The weather forecast says rain tomorrow.")
        )
        assert result.method is ClassifierMethod.RULE_BASED
        assert result.category is Category.GENERAL_INFORMATION
        assert llm.calls == []

    def test_threshold_must_be_valid(self) -> None:
        with pytest.raises(ValueError):
            MessageClassifier(llm_confidence_threshold=1.5)
        with pytest.raises(ValueError):
            MessageClassifier(llm_confidence_threshold=-0.1)


class TestLLMSecurity:
    def test_llm_never_receives_raw_sensitive_value(self) -> None:
        llm = StubLLM(
            payload=json_payload(category="sensitive_information", confidence=0.95)
        )
        classifier = MessageClassifier(llm=llm, llm_confidence_threshold=1.0)
        result = classifier.classify(make_message("Your OTP is 482913."))
        assert llm.calls
        _, safe_message = llm.calls[0]
        assert "482913" not in safe_message
        assert "******" in safe_message
        assert result.category is Category.SENSITIVE_INFORMATION

    def test_sensitive_message_never_reaches_llm_by_default(self) -> None:
        llm = StubLLM(payload=json_payload())
        classifier = MessageClassifier(llm=llm, llm_confidence_threshold=0.75)
        classifier.classify(make_message("My card number is 4111 1111 1111 1111"))
        assert llm.calls == []

    def test_prompt_instructs_about_masking_and_exact_category(self) -> None:
        llm = RecordingLLM()
        result = llm.build_prompt(message_id="T001", safe_message="Your OTP is ******.")
        assert "choose exactly one category" in result.lower()
        assert "sensitive values" in result.lower()
        assert "already been masked" in result.lower()
        assert "482913" not in result
        assert "T001" in result
        assert "Your OTP is ******." in result

    def test_pipeline_passes_masked_message_to_llm(self) -> None:
        llm = StubLLM(payload=json_payload(category="sensitive_information", confidence=0.95))
        classifier = MessageClassifier(llm=llm, llm_confidence_threshold=1.0)
        raw = "Please let me know your thoughts on the draft. OTP is 482913"
        classifier.classify(make_message(raw))
        assert llm.calls
        _, safe_message = llm.calls[0]
        assert "482913" not in safe_message
        assert safe_message == "Please let me know your thoughts on the draft. OTP is ******"


class TestLLMResponseParsing:
    def test_valid_fenced_json(self) -> None:
        raw = (
            '```json\n{"message_id": "T001", "category": "promotional", '
            '"confidence": 0.8, "reason": "sale"}\n```'
        )
        result = parse_llm_response("T001", raw)
        assert result.category is Category.PROMOTIONAL
        assert result.confidence == 0.8
        assert result.reason == "sale"
        assert result.method is ClassifierMethod.LLM_FALLBACK
        assert result.message_id == "T001"

    def test_category_aliases_accepted(self) -> None:
        for alias in ("action-required", "Action Required", "ACTION_REQUIRED"):
            result = parse_llm_response("T001", json_payload(category=alias))
            assert result.category is Category.ACTION_REQUIRED

    def test_empty_response_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", "")

    def test_non_json_response_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", "not json at all")

    def test_malformed_json_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", '{"category": "promotional",}')

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", json_payload(category="urgent"))

    def test_message_id_mismatch_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", json_payload(message_id="OTHER"))

    def test_out_of_range_confidence_clamped(self) -> None:
        assert parse_llm_response("T001", json_payload(confidence=2)).confidence == 1.0
        assert parse_llm_response("T001", json_payload(confidence=-3)).confidence == 0.0

    def test_unparseable_confidence_becomes_zero(self) -> None:
        result = parse_llm_response("T001", json_payload(confidence="high"))
        assert result.confidence == 0.0

    def test_missing_message_id_ok(self) -> None:
        payload = json_payload()
        result = parse_llm_response("T001", payload)
        assert result.message_id == "T001"


class TestRuleOnRealDataset:
    def test_all_messages_classify_without_crash(self) -> None:
        from app.config import Settings
        from app.services.loader import load_messages_csv

        dataset = load_messages_csv(Settings().messages_csv_path)
        classifier = MessageClassifier()
        for message in dataset.messages:
            result = classifier.classify(message)
            assert result.message_id == message.message_id
            assert result.category in Category
            assert 0.0 <= result.confidence <= 1.0
            assert result.method is ClassifierMethod.RULE_BASED

    def test_raw_values_never_reach_llm_on_dataset(self) -> None:
        from app.config import Settings
        from app.services.loader import load_messages_csv
        from app.services.sensitive_detector import SensitiveDetector

        dataset = load_messages_csv(Settings().messages_csv_path)
        detector = SensitiveDetector()
        llm = StubLLM(
            payload=lambda mid: json_payload(message_id=mid, category="general_information")
        )
        classifier = MessageClassifier(llm=llm, llm_confidence_threshold=1.0)
        for message in dataset.messages:
            classifier.classify(message)
        assert llm.calls
        for (message_id, safe_message), message in zip(llm.calls, dataset.messages):
            for detection in detector.detect(message.message):
                assert detection.matched_value_internal_only not in safe_message, (
                    f"{message_id} leaked into LLM input"
                )
