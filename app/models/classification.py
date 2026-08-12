"""Models for message classification."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Category(StrEnum):
    """The six mutually exclusive message categories."""

    ACTION_REQUIRED = "action_required"
    MEETING_OR_EVENT = "meeting_or_event"
    PERSONAL_INFORMATION = "personal_information"
    GENERAL_INFORMATION = "general_information"
    PROMOTIONAL = "promotional"
    SENSITIVE_INFORMATION = "sensitive_information"


class ClassifierMethod(StrEnum):
    """How a classification result was produced."""

    RULE_BASED = "rule_based"
    LLM_FALLBACK = "llm_fallback"


class ClassificationResult(BaseModel):
    """Structured classification of a single message.

    Attributes:
        message_id: Identifier of the classified message.
        category: Exactly one of the six categories.
        confidence: Confidence in the prediction, in ``[0, 1]``.
        reason: Short, factual justification derived only from the message.
        method: ``rule_based`` or ``llm_fallback``.
    """

    message_id: str
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    method: ClassifierMethod = ClassifierMethod.RULE_BASED

    @field_validator("reason")
    @classmethod
    def _clamp_reason(cls, value: str) -> str:
        return value.strip()[:300]
