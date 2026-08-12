"""Tests for the end-to-end pipeline orchestrator."""

from __future__ import annotations

import json
from datetime import datetime

from app.config import Settings
from app.models.classification import (
    Category,
    ClassificationResult,
    ClassifierMethod,
)
from app.models.message import RawMessage
from app.models.task_event import ExtractedItem, ItemType, Priority
from app.services.classifier import MessageClassifier, MessageClassifierLLM
from app.services.extractor import MessageExtractor, MessageExtractorLLM
from app.services.loader import load_messages_csv
from app.services.pipeline import PipelineRunner


def make_message(text: str, message_id: str = "T001") -> RawMessage:
    return RawMessage(
        message_id=message_id,
        timestamp=datetime(2026, 9, 1, 8, 0, 0),
        sender="Meera",
        message=text,
    )


class FailingClassifierLLM(MessageClassifierLLM):
    def classify(self, *, message_id: str, safe_message: str) -> ClassificationResult | None:
        raise RuntimeError("provider down")


class FailingExtractorLLM(MessageExtractorLLM):
    def extract(self, *, message_id: str, safe_message: str) -> tuple[ExtractedItem, ...] | None:
        raise RuntimeError("provider down")


class StubClassifierLLM(MessageClassifierLLM):
    def classify(self, *, message_id: str, safe_message: str) -> ClassificationResult | None:
        return ClassificationResult(
            message_id=message_id,
            category=Category.ACTION_REQUIRED,
            confidence=0.8,
            reason="llm fallback",
            method=ClassifierMethod.LLM_FALLBACK,
        )


class TestPipelineRunner:
    def test_full_dataset_processed_without_skips(self) -> None:
        dataset = load_messages_csv(Settings().messages_csv_path)
        run = PipelineRunner().run(dataset.messages)
        assert run.summary.total_messages == 900
        assert run.summary.classified_messages == 900
        assert len(run.messages) == 900

    def test_every_message_has_exactly_one_classification(self) -> None:
        dataset = load_messages_csv(Settings().messages_csv_path)
        run = PipelineRunner().run(dataset.messages)
        for result in run.messages:
            classification = result.classification
            assert classification.message_id == result.message_id
            assert classification.category in Category
            assert 0.0 <= classification.confidence <= 1.0
            assert classification.reason

    def test_all_message_ids_preserved(self) -> None:
        dataset = load_messages_csv(Settings().messages_csv_path)
        run = PipelineRunner().run(dataset.messages)
        output_ids = [result.message_id for result in run.messages]
        expected_ids = [message.message_id for message in dataset.messages]
        assert output_ids == expected_ids
        assert len(set(output_ids)) == 900

    def test_offline_mode_uses_rules_only(self) -> None:
        dataset = load_messages_csv(Settings().messages_csv_path)
        run = PipelineRunner().run(dataset.messages)
        assert run.summary.rule_based_classifications == 900
        assert run.summary.llm_fallback_classifications == 0
        assert run.summary.failures == 0
        assert run.summary.processing_duration_seconds >= 0

    def test_sensitive_detection_precedes_external_processing(self) -> None:
        messages = [
            make_message("Your OTP is 482913.", "S001"),
            make_message("Please submit the report by 2026-09-04.", "S002"),
        ]
        run = PipelineRunner().run(messages)
        assert run.messages[0].sensitive.has_detection
        assert "482913" not in run.messages[0].safe_message

    def test_llm_failure_never_loses_messages(self) -> None:
        classifier = MessageClassifier(llm=FailingClassifierLLM(), llm_confidence_threshold=1.0)
        extractor = MessageExtractor(llm=FailingExtractorLLM())
        runner = PipelineRunner(classifier=classifier, extractor=extractor)
        messages = [
            make_message("The weather is nice today.", "M001"),
            make_message("Please submit the report by 2026-09-04.", "M002"),
        ]
        run = runner.run(messages)
        assert len(run.messages) == 2
        assert run.summary.failures >= 1
        for result in run.messages:
            assert result.classification.category in Category

    def test_llm_fallback_method_is_recorded(self) -> None:
        classifier = MessageClassifier(llm=StubClassifierLLM(), llm_confidence_threshold=1.0)
        runner = PipelineRunner(classifier=classifier, extractor=MessageExtractor())
        result = runner.analyze(make_message("The weather is nice today.", "M001"))
        assert result.classification.method is ClassifierMethod.LLM_FALLBACK

    def test_summary_counts(self) -> None:
        messages = [
            make_message("Please submit the report by 2026-09-04.", "T001"),
            make_message("The weather is nice today.", "T002"),
        ]
        run = PipelineRunner().run(messages)
        assert run.summary.total_messages == 2
        assert run.summary.messages_with_extracted_items == 1
        assert run.summary.total_extracted_items == 1
        assert run.summary.items_by_type == {"task": 1}
        assert run.summary.messages_with_sensitive == 0


class TestFinalResults:
    def test_final_result_structure(self) -> None:
        dataset = load_messages_csv(Settings().messages_csv_path)
        run = PipelineRunner().run(dataset.messages)
        finals = run.to_final_results()
        assert len(finals) == 900
        sample = finals[0]
        dumped = json.loads(sample.model_dump_json())
        assert set(dumped) == {
            "message_id",
            "timestamp",
            "sender",
            "classification",
            "security",
            "extracted_items",
        }
        assert "message" not in dumped
        assert "safe_message" not in dumped

    def test_final_results_preserve_dataset_order(self) -> None:
        dataset = load_messages_csv(Settings().messages_csv_path)
        run = PipelineRunner().run(dataset.messages)
        finals = run.to_final_results()
        assert [result.message_id for result in finals] == [
            message.message_id for message in dataset.messages
        ]

    def test_no_raw_sensitive_value_in_final_results(self) -> None:
        messages = [make_message("Password is BlueRiver#29.", "SEC_1")]
        run = PipelineRunner().run(messages)
        finals = run.to_final_results()
        serialized = finals[0].model_dump_json()
        assert "BlueRiver#29" not in serialized
        assert "SEC_1" in serialized


class TestSettingsOfflineMode:
    def test_default_settings_are_offline(self) -> None:
        settings = Settings()
        assert settings.llm_enabled is False
        assert not settings.llm_configured

    def test_from_env_defaults_to_offline_without_keys(self) -> None:
        settings = Settings.from_env()
        assert settings.llm_enabled is False

    def test_llm_configured_requires_key_and_model(self) -> None:
        assert Settings(llm_enabled=True).llm_configured is False
        assert Settings(llm_enabled=True, llm_model="gpt-4o").llm_configured is False
        configured = Settings(llm_enabled=True, llm_model="gpt-4o", llm_api_key="sk-test")
        assert configured.llm_configured is True


class TestItemIdsOnDataset:
    def test_item_ids_follow_type_prefix_format(self) -> None:
        dataset = load_messages_csv(Settings().messages_csv_path)
        run = PipelineRunner().run(dataset.messages)
        for result in run.messages:
            for item in result.extraction.items:
                assert item.item_id.startswith(f"{item.type.value.upper()}_")
                assert item.item_id.endswith(item.source_message_id) or (
                    "-" in item.item_id[len(item.source_message_id) + len(item.type.value) + 1 :]
                )

    def test_item_ids_unique_across_dataset(self) -> None:
        dataset = load_messages_csv(Settings().messages_csv_path)
        run = PipelineRunner().run(dataset.messages)
        item_ids = [
            item.item_id
            for result in run.messages
            for item in result.extraction.items
        ]
        assert len(item_ids) == len(set(item_ids))

    def test_extracted_items_types_and_priority_valid(self) -> None:
        dataset = load_messages_csv(Settings().messages_csv_path)
        run = PipelineRunner().run(dataset.messages)
        for result in run.messages:
            for item in result.extraction.items:
                assert item.type in ItemType
                assert item.priority in Priority
                assert item.source_message_id == result.message_id
