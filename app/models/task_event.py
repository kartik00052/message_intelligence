"""Models for extracted tasks, events, reminders and meetings.

An :class:`ExtractedItem` represents one validated actionable or scheduled item
found in a message. Fields that are genuinely unavailable must be left as
``None`` / ``unknown`` - nothing is ever guessed or fabricated.
"""

from __future__ import annotations

import re
from datetime import date as Date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class ItemType(StrEnum):
    """Allowed kinds of extracted items."""

    TASK = "task"
    MEETING = "meeting"
    EVENT = "event"
    REMINDER = "reminder"


class Priority(StrEnum):
    """Allowed priority levels for an extracted item."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class ExtractedItem(BaseModel):
    """A single validated extracted item.

    Attributes:
        item_id: Deterministic identifier derived from type and source message.
        type: One of ``task`` / ``meeting`` / ``event`` / ``reminder``.
        title: Short human readable title.
        description: Optional supporting context (never contains a raw value).
        date: Calendar date of the item when present.
        deadline: Due date for tasks when explicitly stated.
        time: Normalised 24h ``HH:MM`` time when explicitly stated.
        person: A person clearly identified in the message, else ``None``.
        priority: Explicit urgency when present, else ``unknown``.
        source_message_id: The message the item was extracted from.
    """

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

    @field_validator("time")
    @classmethod
    def _validate_time_format(cls, value: str | None) -> str | None:
        if value is None or _TIME_RE.fullmatch(value):
            return value
        raise ValueError(f"invalid time {value!r}; expected HH:MM (24h)")


class ExtractorMethod(StrEnum):
    """How an extraction result was produced."""

    RULE_BASED = "rule_based"
    LLM_FALLBACK = "llm_fallback"
    NONE = "none"


class ExtractionResult(BaseModel):
    """Extraction outcome for a single message.

    Attributes:
        message_id: Identifier of the analysed message.
        items: Extracted items, empty when nothing was extracted.
        method: ``rule_based``, ``llm_fallback`` or ``none``.
        reason: Short, factual justification of the extraction outcome.
    """

    model_config = ConfigDict(frozen=True)

    message_id: str
    items: tuple[ExtractedItem, ...] = ()
    method: ExtractorMethod = ExtractorMethod.RULE_BASED
    reason: str = ""
