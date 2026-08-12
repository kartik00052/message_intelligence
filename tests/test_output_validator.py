"""Unit tests for output validation and the sensitive-value leak scanner."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from app.models.message import RawMessage
from app.services.leak_scanner import LeakScanner
from app.services.output_validator import (
    build_report,
    validate_classifications,
    validate_extractions,
    validate_sensitive_results,
)

EXPECTED = [f"MSG_{i:04d}" for i in range(1, 6)]


def classification(message_id: str = "MSG_0001") -> dict[str, Any]:
    return {
        "message_id": message_id,
        "category": "action_required",
        "confidence": 0.9,
        "reason": "explicit deadline",
        "method": "rule_based",
    }


def sensitive_result(message_id: str = "MSG_0001") -> dict[str, Any]:
    return {"message_id": message_id, "has_detection": False, "detections": []}


def extraction_result(message_id: str = "MSG_0001") -> dict[str, Any]:
    return {
        "message_id": message_id,
        "items": [],
        "method": "none",
        "reason": "No actionable or schedulable signal detected.",
    }


def extraction_with_item(message_id: str = "MSG_0001") -> dict[str, Any]:
    return {
        "message_id": message_id,
        "items": [
            {
                "item_id": f"TASK_{message_id}",
                "type": "task",
                "title": "submit the report",
                "description": None,
                "date": None,
                "deadline": "2026-09-04",
                "time": None,
                "person": None,
                "priority": "unknown",
                "source_message_id": message_id,
            }
        ],
        "method": "rule_based",
        "reason": "Matched a task with an explicit deadline.",
    }


def final_result(message_id: str = "MSG_0001") -> dict[str, Any]:
    return {
        "message_id": message_id,
        "timestamp": "2026-09-01T08:00:00",
        "sender": "Meera",
        "classification": classification(message_id),
        "security": sensitive_result(message_id),
        "extracted_items": [],
    }


class TestValidateClassifications:
    def test_valid_records_pass(self) -> None:
        records = [classification(mid) for mid in EXPECTED]
        assert validate_classifications(records, expected_ids=EXPECTED) == []

    def test_invalid_record_reported(self) -> None:
        records = [classification()]
        records[0]["category"] = "bogus"
        issues = validate_classifications(records, expected_ids=["MSG_0001"])
        assert any(issue.code == "invalid_classification" for issue in issues)

    def test_missing_and_duplicate_ids_reported(self) -> None:
        records = [
            classification("MSG_0001"),
            classification("MSG_0001"),
            classification("MSG_0003"),
        ]
        issues = validate_classifications(records, expected_ids=EXPECTED)
        codes = {issue.code for issue in issues}
        assert "missing_message_id" in codes
        assert "duplicate_message_id" in codes
        assert "unknown_message_id" not in codes

    def test_unknown_id_reported(self) -> None:
        records = [classification("MSG_9999")]
        issues = validate_classifications(records, expected_ids=EXPECTED)
        codes = {issue.code for issue in issues}
        assert "unknown_message_id" in codes
        assert "missing_message_id" in codes


class TestValidateSensitive:
    def test_valid_records_pass(self) -> None:
        records = [sensitive_result(mid) for mid in EXPECTED]
        assert validate_sensitive_results(records, expected_ids=EXPECTED) == []

    def test_invalid_record_reported(self) -> None:
        records = [{"message_id": "MSG_0001", "detections": "not-a-list"}]
        issues = validate_sensitive_results(records, expected_ids=["MSG_0001"])
        assert any(issue.code == "invalid_sensitive_result" for issue in issues)

    def test_well_formed_public_detection_passes(self) -> None:
        records = [
            {
                "message_id": "MSG_0001",
                "has_detection": True,
                "detections": [
                    {
                        "detected": True,
                        "sensitivity_type": "one_time_password",
                        "risk": "high",
                        "masked_text": "******",
                        "recommended_action": "do not share",
                    }
                ],
            }
        ]
        assert validate_sensitive_results(records, expected_ids=["MSG_0001"]) == []

    def test_unexpected_extra_field_reported(self) -> None:
        records = [{"message_id": "MSG_0001", "otp_raw": "482913"}]
        issues = validate_sensitive_results(records, expected_ids=["MSG_0001"])
        assert any(issue.code == "invalid_sensitive_result" for issue in issues)


class TestValidateExtractions:
    def test_valid_records_pass(self) -> None:
        records = [extraction_with_item(mid) for mid in EXPECTED]
        assert validate_extractions(records, expected_ids=EXPECTED) == []

    def test_empty_items_pass(self) -> None:
        records = [extraction_result(mid) for mid in EXPECTED]
        assert validate_extractions(records, expected_ids=EXPECTED) == []

    def test_duplicate_item_id_reported(self) -> None:
        records = [extraction_with_item("MSG_0001"), extraction_with_item("MSG_0001")]
        issues = validate_extractions(records, expected_ids=["MSG_0001", "MSG_0001"])
        assert any(issue.code == "duplicate_item_id" for issue in issues)

    def test_source_message_mismatch_reported(self) -> None:
        record = extraction_with_item("MSG_0001")
        record["items"][0]["source_message_id"] = "MSG_0002"
        issues = validate_extractions([record], expected_ids=["MSG_0001"])
        assert any(issue.code == "item_source_mismatch" for issue in issues)

    def test_invalid_item_reported(self) -> None:
        record = extraction_with_item()
        record["items"][0]["time"] = "25:99"
        issues = validate_extractions([record], expected_ids=["MSG_0001"])
        assert any(issue.code == "invalid_extraction_result" for issue in issues)

    def test_missing_ids_reported(self) -> None:
        issues = validate_extractions([extraction_result()], expected_ids=EXPECTED)
        assert any(issue.code == "missing_message_id" for issue in issues)


class TestBuildReport:
    def test_ok_report(self) -> None:
        report = build_report(
            expected_ids=EXPECTED,
            classifications=[classification(mid) for mid in EXPECTED],
            sensitive_results=[sensitive_result(mid) for mid in EXPECTED],
            extractions=[extraction_result(mid) for mid in EXPECTED],
        )
        assert report.ok
        assert report.message_count == 5
        assert report.issues == ()
        assert report.artifact_counts == {
            "classifications": 5,
            "sensitive_detections": 5,
            "extracted_items": 5,
        }

    def test_failed_report_includes_all_issue_groups(self) -> None:
        report = build_report(
            expected_ids=EXPECTED,
            classifications=[classification()],
            sensitive_results=[sensitive_result()],
            extractions=[extraction_result()],
            extra_issues=[],
        )
        assert not report.ok
        codes = {issue.code for issue in report.issues}
        assert "missing_message_id" in codes

    def test_extra_issues_are_included(self) -> None:
        from app.services.output_validator import OutputIssue

        report = build_report(
            expected_ids=["MSG_0001"],
            classifications=[classification()],
            sensitive_results=[sensitive_result()],
            extractions=[extraction_result()],
            extra_issues=[OutputIssue(code="leak", detail="raw value in artifact")],
        )
        assert not report.ok
        assert any(issue.code == "leak" for issue in report.issues)


def make_message(text: str, message_id: str = "MSG_0001") -> RawMessage:
    return RawMessage(
        message_id=message_id,
        timestamp=datetime(2026, 9, 1, 8, 0, 0),
        sender="Meera",
        message=text,
    )


class TestLeakScanner:
    def test_leak_is_detected(self) -> None:
        messages = [make_message("Your OTP is 482913.", "MSG_0001")]
        artifact = json.dumps({"message_id": "MSG_0001", "note": "OTP is 482913"})
        findings = LeakScanner().scan_artifact(
            artifact_name="extracted_items.json",
            artifact_text=artifact,
            messages={m.message_id: m for m in messages},
        )
        assert len(findings) == 1
        assert findings[0].message_id == "MSG_0001"
        assert findings[0].artifact == "extracted_items.json"
        assert findings[0].sensitivity_type == "one_time_password"
        assert "482913" not in findings[0].model_dump_json()

    def test_no_leak_when_value_absent(self) -> None:
        messages = [make_message("Your OTP is 482913.", "MSG_0001")]
        artifact = json.dumps({"message_id": "MSG_0001", "masked": "******"})
        findings = LeakScanner().scan_artifact(
            artifact_name="x.json",
            artifact_text=artifact,
            messages={m.message_id: m for m in messages},
        )
        assert findings == []

    def test_card_number_leak_detected(self) -> None:
        messages = [make_message("My card number is 4111 1111 1111 1111.", "MSG_0001")]
        artifact = '{"card": "4111 1111 1111 1111"}'
        findings = LeakScanner().scan_artifact(
            artifact_name="x.json",
            artifact_text=artifact,
            messages={m.message_id: m for m in messages},
        )
        assert len(findings) == 1
        assert findings[0].sensitivity_type == "payment_card_number"

    def test_scan_across_multiple_artifacts(self) -> None:
        messages = [make_message("Password is BlueRiver#29.", "MSG_0001")]
        result = LeakScanner().scan(
            artifacts={
                "classifications.json": '{"ok": true}',
                "extracted_items.json": '{"leak": "BlueRiver#29"}',
            },
            messages=messages,
        )
        assert not result.ok
        assert len(result.findings) == 1
        assert result.findings[0].artifact == "extracted_items.json"

    def test_masked_text_is_not_a_leak(self) -> None:
        messages = [make_message("Your OTP is 482913.", "MSG_0001")]
        artifact = '{"masked": "*********", "masked2": "[REDACTED]"}'
        findings = LeakScanner().scan_artifact(
            artifact_name="x.json",
            artifact_text=artifact,
            messages={m.message_id: m for m in messages},
        )
        assert findings == []

    def test_full_dataset_leak_scan(self) -> None:
        from app.config import Settings
        from app.services.loader import load_messages_csv
        from app.services.pipeline import PipelineRunner
        from scripts.run_pipeline import (
            build_classifications_payload,
            build_final_results_payload,
            build_sensitive_payload,
            build_tasks_events_payload,
        )

        dataset = load_messages_csv(Settings().messages_csv_path)
        run_result = PipelineRunner().run(dataset.messages)
        artifacts = {
            "classifications.json": json.dumps(build_classifications_payload(run_result.messages)),
            "sensitive_detections.json": json.dumps(build_sensitive_payload(run_result.messages)),
            "tasks_events.json": json.dumps(build_tasks_events_payload(run_result.messages)),
            "final_results.json": json.dumps(build_final_results_payload(run_result)),
        }
        result = LeakScanner().scan(artifacts=artifacts, messages=dataset.messages)
        assert result.ok
        assert result.findings == ()


class TestQualityReport:
    """Tests for the consolidated ``validation_report.json`` quality report."""

    def _report(
        self,
        *,
        classifications: list[dict[str, Any]] | None = None,
        sensitive_results: list[dict[str, Any]] | None = None,
        extractions: list[dict[str, Any]] | None = None,
        final_results: list[dict[str, Any]] | None = None,
        mandatory_ids: list[str] | None = None,
        leak_ok: bool = True,
    ) -> Any:
        from app.services.output_validator import build_quality_report

        return build_quality_report(
            generated_at="2026-01-01T00:00:00+00:00",
            expected_ids=EXPECTED,
            classifications=classifications or [classification(mid) for mid in EXPECTED],
            sensitive_results=sensitive_results or [sensitive_result(mid) for mid in EXPECTED],
            extractions=extractions or [extraction_result(mid) for mid in EXPECTED],
            final_results=final_results or [final_result(mid) for mid in EXPECTED],
            mandatory_ids=mandatory_ids or EXPECTED,
            leak_ok=leak_ok,
        )

    def test_successful_validation(self) -> None:
        report = self._report()
        assert report.validation_status == "PASS"
        assert report.sensitive_value_leak_check == "PASS"
        assert report.total_input_messages == 5
        assert report.classified_messages == 5
        assert report.missing_message_ids == 0
        assert report.duplicate_message_ids == 0
        assert report.invalid_categories == 0
        assert report.invalid_confidence_scores == 0
        assert report.mandatory_messages_found == 5
        assert report.mandatory_messages_processed == 5
        assert report.mandatory_messages_missing == ()
        assert report.issues == ()

    def test_missing_message_id_fails_validation(self) -> None:
        records = [classification(mid) for mid in EXPECTED[:-1]]
        report = self._report(classifications=records)
        assert report.validation_status == "FAIL"
        assert report.missing_message_ids == 1
        assert any(issue.code == "missing_message_id" for issue in report.issues)

    def test_duplicate_message_id_fails_validation(self) -> None:
        records = [classification("MSG_0001"), classification("MSG_0001")]
        records.extend(classification(mid) for mid in EXPECTED[1:])
        report = self._report(classifications=records)
        assert report.validation_status == "FAIL"
        assert report.duplicate_message_ids == 1

    def test_invalid_category_fails_validation(self) -> None:
        records = [classification(mid) for mid in EXPECTED]
        records[0]["category"] = "not_a_category"
        report = self._report(classifications=records)
        assert report.validation_status == "FAIL"
        assert report.invalid_categories == 1

    def test_invalid_confidence_fails_validation(self) -> None:
        records = [classification(mid) for mid in EXPECTED]
        records[0]["confidence"] = 1.7
        report = self._report(classifications=records)
        assert report.validation_status == "FAIL"
        assert report.invalid_confidence_scores == 1

    def test_invalid_task_source_fails_validation(self) -> None:
        record = extraction_with_item("MSG_0001")
        record["items"][0]["source_message_id"] = "MSG_9999"
        rest = [extraction_result(mid) for mid in EXPECTED[1:]]
        report = self._report(extractions=[record] + rest)
        assert report.validation_status == "FAIL"
        assert any(issue.code == "item_source_mismatch" for issue in report.issues)

    def test_sensitive_leak_fails_validation(self) -> None:
        report = self._report(leak_ok=False)
        assert report.validation_status == "FAIL"
        assert report.sensitive_value_leak_check == "FAIL"

    def test_missing_mandatory_message_fails_validation(self) -> None:
        mandatory = [*EXPECTED, "MSG_0099"]
        report = self._report(mandatory_ids=mandatory)
        assert report.validation_status == "FAIL"
        assert report.mandatory_messages_found == 5
        assert report.mandatory_messages_processed == 5
        assert report.mandatory_messages_missing == ("MSG_0099",)

    def test_task_event_count_and_sensitive_count(self) -> None:
        records = [extraction_with_item(mid) for mid in EXPECTED]
        report = self._report(extractions=records)
        assert report.task_event_count == 5
        sensitive = [sensitive_result(mid) for mid in EXPECTED]
        sensitive[0]["has_detection"] = True
        sensitive[0]["detections"] = [
            {
                "detected": True,
                "sensitivity_type": "one_time_password",
                "risk": "high",
                "masked_text": "******",
                "recommended_action": "do not share",
            }
        ]
        report = self._report(sensitive_results=sensitive)
        assert report.sensitive_message_count == 1
