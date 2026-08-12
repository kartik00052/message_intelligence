"""Pydantic response models for the FastAPI application.

These models define the exact JSON shape of every API response. They reuse the
validated pipeline models (:class:`ClassificationResult`,
:class:`ExtractedItem`, :class:`MessageSensitiveResult`,
:class:`FinalMessageResult`) so the API can never emit a value that the
pipeline did not already mark as safe to expose.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.classification import Category, ClassificationResult, ClassifierMethod
from app.models.pipeline import FinalMessageResult, MessageSensitiveResult
from app.models.task_event import ExtractedItem, ItemType, Priority


class HealthResponse(BaseModel):
    """Simple liveness response for ``GET /health``."""

    status: str


class StatsResponse(BaseModel):
    """Aggregate statistics for ``GET /api/stats`` (never message content)."""

    total_messages: int
    classified_messages: int
    sensitive_messages: int
    task_event_count: int
    rule_based_count: int
    llm_fallback_count: int
    validation_status: str


class MessageListItem(BaseModel):
    """One row of the message list/table (lightweight, no message text)."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    timestamp: datetime
    sender: str
    category: Category
    confidence: float
    method: ClassifierMethod
    has_sensitive: bool


class MessageListResponse(BaseModel):
    """Paginated ``GET /api/messages`` response."""

    model_config = ConfigDict(frozen=True)

    total: int
    offset: int
    limit: int
    items: tuple[MessageListItem, ...]


class MessageDetail(BaseModel):
    """Full detail for one message (masked content only)."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    timestamp: datetime
    sender: str
    safe_message: str
    classification: ClassificationResult
    security: MessageSensitiveResult
    extracted_items: tuple[ExtractedItem, ...] = ()

    @classmethod
    def from_final(cls, final: FinalMessageResult) -> MessageDetail:
        """Build the API detail from a validated final result."""
        return cls(
            message_id=final.message_id,
            timestamp=final.timestamp,
            sender=final.sender,
            safe_message=final.safe_message,
            classification=final.classification,
            security=final.security,
            extracted_items=final.extracted_items,
        )


class TaskItem(BaseModel):
    """One extracted task/event/reminder/meeting exposed via ``GET /api/tasks``."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    type: ItemType
    title: str
    description: str | None = None
    date: Date | None = None
    deadline: Date | None = None
    time: str | None = None
    person: str | None = None
    priority: Priority = Priority.UNKNOWN
    source_message_id: str

    @classmethod
    def from_item(cls, item: ExtractedItem) -> TaskItem:
        """Build the API task from a validated extracted item."""
        return cls(
            item_id=item.item_id,
            type=item.type,
            title=item.title,
            description=item.description,
            date=item.date,
            deadline=item.deadline,
            time=item.time,
            person=item.person,
            priority=item.priority,
            source_message_id=item.source_message_id,
        )


class TaskListResponse(BaseModel):
    """Paginated ``GET /api/tasks`` response."""

    model_config = ConfigDict(frozen=True)

    total: int
    offset: int
    limit: int
    items: tuple[TaskItem, ...]


class SensitiveListResponse(BaseModel):
    """Paginated ``GET /api/sensitive`` response."""

    model_config = ConfigDict(frozen=True)

    total: int
    offset: int
    limit: int
    items: tuple[MessageSensitiveResult, ...]


class MandatoryDemoResponse(BaseModel):
    """The 15 mandatory demo messages in dataset chronological order."""

    model_config = ConfigDict(frozen=True)

    requested_ids: tuple[str, ...]
    results: tuple[MessageDetail, ...]
    found: int
    processed: int
    missing: tuple[str, ...] = ()


class LeakScanReport(BaseModel):
    """Leak-scan outcome included in the validation response."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    findings: tuple[dict[str, str], ...] = ()


class ValidationReportResponse(BaseModel):
    """The full ``validation_report.json`` document for ``GET /api/validation``."""

    model_config = ConfigDict(frozen=True)

    generated_at: str
    summary: dict[str, object]
    report: dict[str, object]
    leak_scan: LeakScanReport
    validation_status: str
