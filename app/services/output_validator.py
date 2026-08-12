"""Output validation for the generated JSON artifacts.

Validates that every artifact is Pydantic-conformant and complete with respect
to the input dataset: all message IDs are preserved exactly once, no IDs are
duplicated, missing or invented, and every record parses into the expected
typed model. Validation is strict - a corrupt artifact fails loudly instead of
being silently written.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel

from app.models.classification import Category, ClassificationResult
from app.models.pipeline import FinalMessageResult, MessageSensitiveResult
from app.models.task_event import ExtractionResult

_VALID_CATEGORIES = {category.value for category in Category}


class OutputIssue(BaseModel):
    """A single output-validation problem.

    Attributes:
        code: Machine readable category of the problem.
        detail: Human readable description, including the affected message ID.
    """

    code: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.detail}"


class OutputValidationReport(BaseModel):
    """Aggregated validation report for all generated artifacts.

    Attributes:
        ok: True when every artifact is complete and Pydantic-conformant.
        message_count: Number of input messages the artifacts must cover.
        artifact_counts: Number of records in each validated artifact.
        issues: Every detected problem across all artifacts.
    """

    ok: bool
    message_count: int
    artifact_counts: dict[str, int]
    issues: tuple[OutputIssue, ...] = ()


def validate_classifications(
    records: Sequence[dict[str, Any]], *, expected_ids: Sequence[str]
) -> list[OutputIssue]:
    """Validate a ``classifications.json`` records list."""
    issues: list[OutputIssue] = []
    for record in records:
        try:
            ClassificationResult.model_validate(record)
        except Exception as exc:  # noqa: BLE001 - report every invalid record
            issues.append(
                OutputIssue(
                    code="invalid_classification",
                    detail=f"message_id={record.get('message_id')!r}: {exc}",
                )
            )
    issues.extend(_completeness_issues(records, expected_ids, "classifications"))
    return issues


def validate_sensitive_results(
    records: Sequence[dict[str, Any]], *, expected_ids: Sequence[str]
) -> list[OutputIssue]:
    """Validate a ``sensitive_detections.json`` records list."""
    issues: list[OutputIssue] = []
    for record in records:
        try:
            MessageSensitiveResult.model_validate(record)
        except Exception as exc:  # noqa: BLE001
            issues.append(
                OutputIssue(
                    code="invalid_sensitive_result",
                    detail=f"message_id={record.get('message_id')!r}: {exc}",
                )
            )
    issues.extend(_completeness_issues(records, expected_ids, "sensitive_detections"))
    return issues


def validate_extractions(
    records: Sequence[dict[str, Any]], *, expected_ids: Sequence[str]
) -> list[OutputIssue]:
    """Validate a ``tasks_events.json`` records list."""
    issues: list[OutputIssue] = []
    seen_item_ids: set[str] = set()
    for record in records:
        try:
            result = ExtractionResult.model_validate(record)
        except Exception as exc:  # noqa: BLE001
            issues.append(
                OutputIssue(
                    code="invalid_extraction_result",
                    detail=f"message_id={record.get('message_id')!r}: {exc}",
                )
            )
            continue
        for item in result.items:
            if item.item_id in seen_item_ids:
                issues.append(
                    OutputIssue(
                        code="duplicate_item_id",
                        detail=f"item_id {item.item_id!r} appears more than once.",
                    )
                )
            seen_item_ids.add(item.item_id)
            if item.source_message_id != result.message_id:
                issues.append(
                    OutputIssue(
                        code="item_source_mismatch",
                        detail=f"item {item.item_id!r} references source "
                        f"{item.source_message_id!r} but belongs to "
                        f"{result.message_id!r}.",
                    )
                )
    issues.extend(_completeness_issues(records, expected_ids, "tasks_events"))
    return issues


def validate_final_results(
    records: Sequence[dict[str, Any]], *, expected_ids: Sequence[str]
) -> list[OutputIssue]:
    """Validate a ``final_results.json`` records list."""
    issues: list[OutputIssue] = []
    for record in records:
        try:
            FinalMessageResult.model_validate(record)
        except Exception as exc:  # noqa: BLE001
            issues.append(
                OutputIssue(
                    code="invalid_final_result",
                    detail=f"message_id={record.get('message_id')!r}: {exc}",
                )
            )
    issues.extend(_completeness_issues(records, expected_ids, "final_results"))
    return issues


def build_report(
    *,
    expected_ids: Sequence[str],
    classifications: Sequence[dict[str, Any]],
    sensitive_results: Sequence[dict[str, Any]],
    extractions: Sequence[dict[str, Any]],
    extra_issues: Sequence[OutputIssue] = (),
) -> OutputValidationReport:
    """Combine per-artifact validation into a single report."""
    issues = [
        *validate_classifications(classifications, expected_ids=expected_ids),
        *validate_sensitive_results(sensitive_results, expected_ids=expected_ids),
        *validate_extractions(extractions, expected_ids=expected_ids),
        *extra_issues,
    ]
    return OutputValidationReport(
        ok=not issues,
        message_count=len(expected_ids),
        artifact_counts={
            "classifications": len(classifications),
            "sensitive_detections": len(sensitive_results),
            "extracted_items": len(extractions),
        },
        issues=tuple(issues),
    )


def _completeness_issues(
    records: Sequence[dict[str, Any]],
    expected_ids: Sequence[str],
    artifact: str,
) -> list[OutputIssue]:
    """Check that every expected message ID appears exactly once."""
    actual_ids = [str(record.get("message_id")) for record in records]
    expected = set(expected_ids)
    actual = set(actual_ids)
    issues: list[OutputIssue] = []

    missing = sorted(expected - actual)
    if missing:
        issues.append(
            OutputIssue(
                code="missing_message_id",
                detail=f"{artifact}: missing {len(missing)} message ID(s): {missing}.",
            )
        )

    extra = sorted(actual - expected)
    if extra:
        issues.append(
            OutputIssue(
                code="unknown_message_id",
                detail=f"{artifact}: {len(extra)} message ID(s) not in the dataset: {extra}.",
            )
        )

    seen: set[str] = set()
    for position, message_id in enumerate(actual_ids):
        if message_id in seen:
            issues.append(
                OutputIssue(
                    code="duplicate_message_id",
                    detail=f"{artifact}: row {position} repeats message_id {message_id!r}.",
                )
            )
        seen.add(message_id)

    return issues


class QualityReport(BaseModel):
    """High-level validation report over the completed pipeline outputs.

    Mirrors the report structure produced in ``validation_report.json``.
    """

    generated_at: str
    total_input_messages: int
    classified_messages: int
    missing_message_ids: int
    duplicate_message_ids: int
    invalid_categories: int
    invalid_confidence_scores: int
    task_event_count: int
    sensitive_message_count: int
    mandatory_messages_found: int
    mandatory_messages_processed: int
    mandatory_messages_missing: tuple[str, ...] = ()
    sensitive_value_leak_check: Literal["PASS", "FAIL"]
    validation_status: Literal["PASS", "FAIL"]
    issues: tuple[OutputIssue, ...] = ()


def build_quality_report(
    *,
    generated_at: str,
    expected_ids: Sequence[str],
    classifications: Sequence[dict[str, Any]],
    sensitive_results: Sequence[dict[str, Any]],
    extractions: Sequence[dict[str, Any]],
    final_results: Sequence[dict[str, Any]],
    mandatory_ids: Sequence[str],
    leak_ok: bool,
) -> QualityReport:
    """Validate every artifact and produce the consolidated quality report.

    Counts every requirement the pipeline must satisfy: dataset integrity
    (900 in / 900 out, no missing or duplicate IDs), exactly one valid
    classification per message, valid task/event items, sensitive detections,
    mandatory demo coverage and the sensitive-value leak check.
    """
    issues = [
        *validate_classifications(classifications, expected_ids=expected_ids),
        *validate_sensitive_results(sensitive_results, expected_ids=expected_ids),
        *validate_extractions(extractions, expected_ids=expected_ids),
        *validate_final_results(final_results, expected_ids=expected_ids),
    ]

    expected = set(expected_ids)
    classification_ids = [str(record.get("message_id")) for record in classifications]
    actual = set(classification_ids)
    missing_ids = expected - actual
    seen: set[str] = set()
    duplicate_ids = 0
    for message_id in classification_ids:
        if message_id in seen:
            duplicate_ids += 1
        seen.add(message_id)

    invalid_categories = sum(
        1
        for record in classifications
        if not isinstance(record.get("category"), str)
        or record.get("category") not in _VALID_CATEGORIES
    )
    invalid_confidence_scores = sum(
        1
        for record in classifications
        if not _is_valid_confidence(record.get("confidence"))
    )

    task_event_count = 0
    for record in extractions:
        try:
            result = ExtractionResult.model_validate(record)
        except Exception:  # noqa: BLE001 - already reported above
            continue
        task_event_count += len(result.items)

    sensitive_message_count = sum(
        1
        for record in sensitive_results
        if bool(record.get("has_detection"))
        and isinstance(record.get("message_id"), str)
    )

    mandatory_set = set(mandatory_ids)
    mandatory_found = len(mandatory_set & expected)
    mandatory_processed = len(mandatory_set & actual)
    mandatory_missing = tuple(
        sorted(message_id for message_id in mandatory_ids if message_id not in actual)
    )

    leak_check = "PASS" if leak_ok else "FAIL"
    ok = (
        not missing_ids
        and duplicate_ids == 0
        and invalid_categories == 0
        and invalid_confidence_scores == 0
        and not issues
        and leak_ok
        and mandatory_missing == ()
    )
    return QualityReport(
        generated_at=generated_at,
        total_input_messages=len(expected_ids),
        classified_messages=len(classifications),
        missing_message_ids=len(missing_ids),
        duplicate_message_ids=duplicate_ids,
        invalid_categories=invalid_categories,
        invalid_confidence_scores=invalid_confidence_scores,
        task_event_count=task_event_count,
        sensitive_message_count=sensitive_message_count,
        mandatory_messages_found=mandatory_found,
        mandatory_messages_processed=mandatory_processed,
        mandatory_messages_missing=mandatory_missing,
        sensitive_value_leak_check=leak_check,
        validation_status="PASS" if ok else "FAIL",
        issues=tuple(issues),
    )


def _is_valid_confidence(value: Any) -> bool:
    """True when ``value`` is a number in the inclusive ``[0, 1]`` range."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= confidence <= 1.0
