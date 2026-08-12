"""Tests for mandatory-demo coverage.

The 15 mandatory message IDs come from ``mandatory_demo_ids.csv`` through
configuration - nothing is hardcoded in Python.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

import pytest

from app.config import Settings
from app.models.pipeline import FinalMessageResult
from app.services.loader import load_messages_csv
from app.services.mandatory_demo import (
    MandatoryDemoError,
    MandatoryDemoService,
    load_mandatory_ids,
)


class TestLoadMandatoryIds:
    def test_real_file_has_exactly_15_unique_ids(self) -> None:
        ids = load_mandatory_ids(Settings().mandatory_demo_ids_path)
        assert len(ids) == 15
        assert len(set(ids)) == 15

    def test_ids_are_loaded_from_file_not_hardcoded(self) -> None:
        ids = load_mandatory_ids(Settings().mandatory_demo_ids_path)
        dataset = load_messages_csv(Settings().messages_csv_path)
        dataset_ids = {message.message_id for message in dataset.messages}
        assert set(ids) <= dataset_ids

    def test_wrong_count_raises(self, tmp_path) -> None:
        path = tmp_path / "ids.csv"
        path.write_text("message_id\nMSG_0001\nMSG_0002\n", encoding="utf-8")
        with pytest.raises(MandatoryDemoError, match="exactly 15"):
            load_mandatory_ids(path)

    def test_duplicate_ids_raise(self, tmp_path) -> None:
        lines = ["message_id", "MSG_0001", "MSG_0002", "MSG_0002", "MSG_0003"]
        lines.extend(f"MSG_{i:04d}" for i in range(4, 15))
        path = tmp_path / "ids.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(MandatoryDemoError, match="unique"):
            load_mandatory_ids(path)

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(MandatoryDemoError, match="does not exist"):
            load_mandatory_ids(tmp_path / "nope.csv")

    def test_missing_column_raises(self, tmp_path) -> None:
        path = tmp_path / "ids.csv"
        path.write_text("id\nMSG_0001\n", encoding="utf-8")
        with pytest.raises(MandatoryDemoError, match="message_id"):
            load_mandatory_ids(path)


class TestMandatoryDemoCheck:
    def test_all_checks_pass_on_real_data(self) -> None:
        ids = load_mandatory_ids(Settings().mandatory_demo_ids_path)
        dataset = load_messages_csv(Settings().messages_csv_path)
        dataset_ids = [message.message_id for message in dataset.messages]
        check = MandatoryDemoService(ids).check(
            dataset_ids=dataset_ids,
            processed_ids=dataset_ids,
            classified_ids=dataset_ids,
        )
        assert check.ok
        assert check.provided_count == 15
        assert check.unique_count == 15
        assert check.missing_from_dataset == ()
        assert check.not_processed == ()
        assert check.not_classified == ()

    def test_missing_from_dataset_detected(self) -> None:
        check = MandatoryDemoService(["MSG_0001", "MSG_9999"]).check(
            dataset_ids=["MSG_0001", "MSG_0002"],
            processed_ids=["MSG_0001", "MSG_0002"],
            classified_ids=["MSG_0001", "MSG_0002"],
        )
        assert not check.ok
        assert check.missing_from_dataset == ("MSG_9999",)

    def test_not_processed_detected(self) -> None:
        check = MandatoryDemoService(["MSG_0001", "MSG_0002"]).check(
            dataset_ids=["MSG_0001", "MSG_0002"],
            processed_ids=["MSG_0001"],
            classified_ids=["MSG_0001"],
        )
        assert not check.ok
        assert check.not_processed == ("MSG_0002",)

    def test_not_classified_detected(self) -> None:
        check = MandatoryDemoService(["MSG_0001"]).check(
            dataset_ids=["MSG_0001"],
            processed_ids=["MSG_0001"],
            classified_ids=["MSG_0002"],
        )
        assert not check.ok
        assert check.not_classified == ("MSG_0001",)


def _final_result(message_id: str, timestamp: datetime) -> FinalMessageResult:
    from app.models.classification import Category, ClassificationResult, ClassifierMethod
    from app.models.pipeline import MessageSensitiveResult

    return FinalMessageResult(
        message_id=message_id,
        timestamp=timestamp,
        sender="Meera",
        safe_message=f"Safe text for {message_id}.",
        classification=ClassificationResult(
            message_id=message_id,
            category=Category.ACTION_REQUIRED,
            confidence=0.9,
            reason="explicit deadline",
            method=ClassifierMethod.RULE_BASED,
        ),
        security=MessageSensitiveResult(message_id=message_id, has_detection=False),
        extracted_items=(),
    )


class TestMandatoryDemoBuild:
    def test_returns_results_in_dataset_chronological_order(self) -> None:
        results = [
            _final_result("MSG_0001", datetime(2026, 9, 1, 8, 0, 0)),
            _final_result("MSG_0002", datetime(2026, 9, 1, 8, 37, 0)),
            _final_result("MSG_0003", datetime(2026, 9, 1, 9, 14, 0)),
            _final_result("MSG_0007", datetime(2026, 9, 1, 11, 42, 0)),
        ]
        service = MandatoryDemoService(["MSG_0007", "MSG_0001", "MSG_0003"])
        demo = service.build(results)
        assert [result.message_id for result in demo.results] == [
            "MSG_0001",
            "MSG_0003",
            "MSG_0007",
        ]
        assert demo.found == 3
        assert demo.processed == 3
        assert demo.missing == ()

    def test_missing_result_raises_never_fabricates(self) -> None:
        results = [_final_result("MSG_0001", datetime(2026, 9, 1, 8, 0, 0))]
        service = MandatoryDemoService(["MSG_0001", "MSG_0042"])
        with pytest.raises(MandatoryDemoError, match="no fake results"):
            service.build(results)

    def test_real_dataset_service_builds_all_fifteen(self) -> None:
        ids = load_mandatory_ids(Settings().mandatory_demo_ids_path)
        dataset = load_messages_csv(Settings().messages_csv_path)
        raw_messages = {message.message_id: message for message in dataset.messages}

        final_results = []
        for message_id in ids:
            message = raw_messages[message_id]
            final_results.append(
                _final_result(message_id, message.timestamp)
            )
        demo = MandatoryDemoService(ids).build(final_results)
        assert demo.found == 15
        assert demo.processed == 15
        assert demo.missing == ()
        assert len(demo.results) == 15


class TestMandatoryIntegration:
    def test_pipeline_outputs_cover_all_mandatory_messages(self) -> None:
        from app.services.pipeline import PipelineRunner
        from scripts.run_pipeline import (
            build_classifications_payload,
            build_final_results_payload,
        )

        ids = load_mandatory_ids(Settings().mandatory_demo_ids_path)
        dataset = load_messages_csv(Settings().messages_csv_path)
        run_result = PipelineRunner().run(dataset.messages)
        final_results = run_result.to_final_results()
        demo = MandatoryDemoService(ids).build(final_results)

        assert len(demo.results) == 15
        for result in demo.results:
            assert result.message_id in ids

        classification_payload = build_classifications_payload(run_result.messages)
        classification_ids = {
            str(record["message_id"])
            for record in cast("list[dict[str, object]]", classification_payload["messages"])
        }
        assert set(ids) <= classification_ids
        final_payload = build_final_results_payload(run_result)
        final_ids = {
            str(record["message_id"])
            for record in cast("list[dict[str, object]]", final_payload["messages"])
        }
        assert set(ids) <= final_ids
