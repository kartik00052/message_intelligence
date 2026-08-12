"""Pipeline runner.

Loads the dataset, runs detection/masking/classification/extraction over every
message, writes the structured JSON artifacts (``classifications.json``,
``tasks_events.json``, ``sensitive_detections.json``, ``final_results.json``),
validates every artifact, scans for sensitive leaks, checks the mandatory-demo
coverage and writes ``validation_report.json``.

Exit code is 0 on success and 1 when validation or the leak scan fails.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings
from app.models.pipeline import MessagePipelineResult, PipelineRunResult
from app.services.leak_scanner import LeakScanner, LeakScanResult
from app.services.loader import load_messages_csv
from app.services.mandatory_demo import MandatoryDemoService, load_mandatory_ids
from app.services.output_validator import (
    QualityReport,
    build_quality_report,
)
from app.services.pipeline import PipelineRunner

CLASSIFICATIONS_FILENAME = "classifications.json"
TASKS_EVENTS_FILENAME = "tasks_events.json"
SENSITIVE_FILENAME = "sensitive_detections.json"
FINAL_RESULTS_FILENAME = "final_results.json"
VALIDATION_REPORT_FILENAME = "validation_report.json"


def build_classifications_payload(
    results: tuple[MessagePipelineResult, ...],
) -> dict[str, object]:
    """Build the ``classifications.json`` document."""
    return {
        "generated_at": _now(),
        "count": len(results),
        "messages": [result.classification.model_dump(mode="json") for result in results],
    }


def build_tasks_events_payload(
    results: tuple[MessagePipelineResult, ...],
) -> dict[str, object]:
    """Build the ``tasks_events.json`` document."""
    return {
        "generated_at": _now(),
        "count": len(results),
        "messages": [result.extraction.model_dump(mode="json") for result in results],
    }


def build_sensitive_payload(
    results: tuple[MessagePipelineResult, ...],
) -> dict[str, object]:
    """Build the ``sensitive_detections.json`` document."""
    return {
        "generated_at": _now(),
        "count": len(results),
        "messages": [result.sensitive.model_dump(mode="json") for result in results],
    }


def build_final_results_payload(run_result: PipelineRunResult) -> dict[str, object]:
    """Build the ``final_results.json`` document from the pipeline run."""
    return {
        "generated_at": _now(),
        "count": len(run_result.messages),
        "messages": [
            final.model_dump(mode="json") for final in run_result.to_final_results()
        ],
    }


def build_validation_payload(
    *,
    run: PipelineRunResult,
    report: QualityReport,
    leak_scan: LeakScanResult,
) -> dict[str, object]:
    """Build the ``validation_report.json`` document."""
    return {
        "generated_at": _now(),
        "summary": run.summary.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
        "leak_scan": {
            "ok": leak_scan.ok,
            "findings": [finding.model_dump(mode="json") for finding in leak_scan.findings],
        },
        "validation_status": report.validation_status,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON artifact atomically (tmp file + rename)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(
    *,
    settings: Settings,
    runner: PipelineRunner | None = None,
    leak_scanner: LeakScanner | None = None,
) -> tuple[PipelineRunResult, QualityReport, LeakScanResult]:
    """Execute the full pipeline and validation, writing the JSON artifacts."""
    dataset = load_messages_csv(
        settings.messages_csv_path, expected_count=settings.expected_message_count
    )
    run_result = (runner or PipelineRunner()).run(dataset.messages)

    outputs_dir = settings.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    classifications_path = outputs_dir / CLASSIFICATIONS_FILENAME
    tasks_events_path = outputs_dir / TASKS_EVENTS_FILENAME
    sensitive_path = outputs_dir / SENSITIVE_FILENAME
    final_results_path = outputs_dir / FINAL_RESULTS_FILENAME

    write_json(classifications_path, build_classifications_payload(run_result.messages))
    write_json(tasks_events_path, build_tasks_events_payload(run_result.messages))
    write_json(sensitive_path, build_sensitive_payload(run_result.messages))
    write_json(final_results_path, build_final_results_payload(run_result))

    expected_ids = [message.message_id for message in dataset.messages]
    mandatory_ids = load_mandatory_ids(
        settings.mandatory_demo_ids_path,
        expected_count=settings.expected_mandatory_count,
    )
    mandatory_service = MandatoryDemoService(mandatory_ids)
    mandatory_service.check(
        dataset_ids=expected_ids,
        processed_ids=expected_ids,
        classified_ids=[str(record["message_id"]) for record in _records(classifications_path)],
    )

    leak_scan = (leak_scanner or LeakScanner()).scan(
        artifacts=_artifact_texts(
            classifications_path,
            tasks_events_path,
            sensitive_path,
            final_results_path,
        ),
        messages=dataset.messages,
    )

    report = build_quality_report(
        generated_at=_now(),
        expected_ids=expected_ids,
        classifications=_records(classifications_path),
        sensitive_results=_records(sensitive_path),
        extractions=_records(tasks_events_path),
        final_results=_records(final_results_path),
        mandatory_ids=mandatory_ids,
        leak_ok=leak_scan.ok,
    )

    write_json(
        outputs_dir / VALIDATION_REPORT_FILENAME,
        build_validation_payload(run=run_result, report=report, leak_scan=leak_scan),
    )
    return run_result, report, leak_scan


def _records(path: Path) -> list[dict[str, object]]:
    """Load the ``messages`` list out of a written artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["messages"])


def _artifact_texts(*paths: Path) -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in paths}


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m scripts.run_pipeline``."""
    del argv  # no custom CLI flags yet; everything comes from settings/env
    settings = Settings.from_env()
    run_result, report, leak_scan = run(settings=settings)

    summary = run_result.summary
    print(f"Loaded dataset: {settings.messages_csv_path}")
    print(f"Messages processed: {summary.total_messages}")
    print(f"Messages classified: {summary.classified_messages}")
    print(
        f"Messages with extracted items: {summary.messages_with_extracted_items} "
        f"({summary.total_extracted_items} items, {summary.items_by_type})"
    )
    print(f"Sensitive messages: {summary.messages_with_sensitive}")
    print(f"Rule-based classifications: {summary.rule_based_classifications}")
    print(f"LLM fallback classifications: {summary.llm_fallback_classifications}")
    print(f"LLM failures: {summary.failures}")
    print(f"Processing duration: {summary.processing_duration_seconds}s")
    print(f"Artifacts written to: {settings.outputs_dir}")
    print(
        "Mandatory demo: "
        f"{report.mandatory_messages_found} found / "
        f"{report.mandatory_messages_processed} processed / "
        f"{len(report.mandatory_messages_missing)} missing"
    )
    print(f"Validation status: {report.validation_status}")
    for issue in report.issues:
        print(f"  {issue}")
    print(f"Leak findings: {len(leak_scan.findings)}")
    for finding in leak_scan.findings:
        print(f"  {finding.artifact} message {finding.message_id}: {finding.sensitivity_type}")
    print(f"Overall status: {'OK' if report.validation_status == 'PASS' else 'FAILED'}")
    return 0 if report.validation_status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
