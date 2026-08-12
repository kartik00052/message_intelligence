"""Business logic for the FastAPI application.

Route handlers stay thin: all filtering, pagination, statistics and mandatory
demo composition happen here against the validated artifact repository.
"""

from __future__ import annotations

from datetime import date

from app.models.api import (
    LeakScanReport,
    MandatoryDemoResponse,
    MessageDetail,
    MessageListItem,
    MessageListResponse,
    SensitiveListResponse,
    StatsResponse,
    TaskItem,
    TaskListResponse,
    ValidationReportResponse,
)
from app.models.pipeline import FinalMessageResult
from app.models.task_event import ExtractedItem, ItemType, Priority
from app.services.mandatory_demo import MandatoryDemoService
from app.services.output_repository import OutputRepository


class MessageNotFoundError(Exception):
    """Raised when a requested message ID has no processed result."""


class ApiService:
    """High-level API operations over the pipeline artifacts."""

    def __init__(self, repository: OutputRepository) -> None:
        self._repository = repository

    # --------------------------------------------------------------- stats

    def stats(self) -> StatsResponse:
        """Aggregate statistics from the validation report."""
        document = self._repository.load_validation_document()
        summary = _require_mapping(document.get("summary"))
        report = _require_mapping(document.get("report"))
        return StatsResponse(
            total_messages=_require_int(summary.get("total_messages")),
            classified_messages=_require_int(summary.get("classified_messages")),
            sensitive_messages=_require_int(summary.get("messages_with_sensitive")),
            task_event_count=_require_int(summary.get("total_extracted_items")),
            rule_based_count=_require_int(summary.get("rule_based_classifications")),
            llm_fallback_count=_require_int(summary.get("llm_fallback_classifications")),
            validation_status=str(report.get("validation_status") or "UNKNOWN"),
        )

    # ------------------------------------------------------------- messages

    def list_messages(
        self,
        *,
        search: str | None,
        category: str | None,
        sensitive: bool | None,
        limit: int,
        offset: int,
    ) -> MessageListResponse:
        """List messages with optional search / category / sensitive filters."""
        query = (search or "").strip().lower()
        items: list[MessageListItem] = []
        for final in self._repository.load_final_results():
            if category and final.classification.category.value != category:
                continue
            if sensitive is not None and final.security.has_detection != sensitive:
                continue
            if query and not _matches_search(final, query):
                continue
            items.append(
                MessageListItem(
                    message_id=final.message_id,
                    timestamp=final.timestamp,
                    sender=final.sender,
                    category=final.classification.category,
                    confidence=final.classification.confidence,
                    method=final.classification.method,
                    has_sensitive=final.security.has_detection,
                )
            )
        total = len(items)
        return MessageListResponse(
            total=total,
            offset=offset,
            limit=limit,
            items=tuple(items[offset : offset + limit]),
        )

    def get_message(self, message_id: str) -> MessageDetail:
        """Return the full sanitized detail for one message."""
        for final in self._repository.load_final_results():
            if final.message_id == message_id:
                return MessageDetail.from_final(final)
        raise MessageNotFoundError(f"No processed result for message {message_id!r}.")

    # ---------------------------------------------------------------- tasks

    def list_tasks(
        self,
        *,
        item_type: ItemType | None,
        priority: Priority | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> TaskListResponse:
        """List extracted tasks/events with optional filters."""
        items: list[TaskItem] = []
        for final in self._repository.load_final_results():
            for item in final.extracted_items:
                if not _task_matches(item, item_type, priority, date_from, date_to):
                    continue
                items.append(TaskItem.from_item(item))
        total = len(items)
        return TaskListResponse(
            total=total,
            offset=offset,
            limit=limit,
            items=tuple(items[offset : offset + limit]),
        )

    # ------------------------------------------------------------- sensitive

    def list_sensitive(self, *, limit: int, offset: int) -> SensitiveListResponse:
        """List sanitized sensitive-detection results (masked values only)."""
        items = [
            result
            for result in self._repository.load_sensitive_results()
            if result.has_detection
        ]
        total = len(items)
        return SensitiveListResponse(
            total=total,
            offset=offset,
            limit=limit,
            items=tuple(items[offset : offset + limit]),
        )

    # ------------------------------------------------------ mandatory demo

    def mandatory_demo(self) -> MandatoryDemoResponse:
        """Compose the 15 mandatory demo messages in dataset order."""
        mandatory_ids = self._repository.load_mandatory_ids()
        finals = self._repository.load_final_results()
        demo = MandatoryDemoService(mandatory_ids).build(finals)
        return MandatoryDemoResponse(
            requested_ids=demo.requested_ids,
            results=tuple(MessageDetail.from_final(final) for final in demo.results),
            found=demo.found,
            processed=demo.processed,
            missing=demo.missing,
        )

    # ------------------------------------------------------------ validation

    def validation_report(self) -> ValidationReportResponse:
        """Return the validated validation-report document."""
        document = self._repository.load_validation_document()
        leak_scan = _require_mapping(document.get("leak_scan"))
        findings = leak_scan.get("findings")
        if not isinstance(findings, list):
            findings = []
        safe_findings = tuple(
            {
                str(key): str(value)
                for key, value in finding.items()
                if isinstance(finding, dict)
            }
            for finding in findings
            if isinstance(finding, dict)
        )
        return ValidationReportResponse(
            generated_at=str(document.get("generated_at") or ""),
            summary=_require_mapping(document.get("summary")),
            report=_require_mapping(document.get("report")),
            leak_scan=LeakScanReport(
                ok=bool(leak_scan.get("ok")),
                findings=safe_findings,
            ),
            validation_status=str(document.get("validation_status") or "UNKNOWN"),
        )

    # ------------------------------------------------------------ metadata

    @property
    def repository(self) -> OutputRepository:
        """The underlying artifact repository."""
        return self._repository


# ------------------------------------------------------------------- helpers


def _matches_search(final: FinalMessageResult, query: str) -> bool:
    haystack = " ".join(
        (
            final.message_id,
            final.sender,
            final.safe_message,
            final.classification.category.value,
        )
    ).lower()
    return query in haystack


def _task_matches(
    item: ExtractedItem,
    item_type: ItemType | None,
    priority: Priority | None,
    date_from: date | None,
    date_to: date | None,
) -> bool:
    if item_type is not None and item.type != item_type:
        return False
    if priority is not None and item.priority != priority:
        return False
    item_date = item.date if item.date is not None else item.deadline
    if item_date is None:
        return date_from is None and date_to is None
    if date_from is not None and item_date < date_from:
        return False
    if date_to is not None and item_date > date_to:
        return False
    return True


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("validation document is malformed")
    return value


def _require_int(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("validation document is malformed")
    return value
