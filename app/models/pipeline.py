"""Models describing end-to-end pipeline results.

A :class:`MessagePipelineResult` is the validated outcome for a single message
across all pipeline stages (sensitive detection, classification, extraction).
A :class:`FinalMessageResult` is the sanitized per-message model exposed in the
``final_results.json`` artifact - it never carries a raw sensitive value. A
:class:`PipelineRunResult` bundles every message plus a summary.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.classification import ClassificationResult
from app.models.sensitive import PublicSensitiveDetection
from app.models.task_event import ExtractedItem, ExtractionResult


class MessageSensitiveResult(BaseModel):
    """Sanitized sensitive-detection outcome for one message.

    This model only ever contains public (masked) representations, never raw
    sensitive values. Unknown fields are rejected so that a stray raw value can
    never sneak into the artifact.
    """

    model_config = ConfigDict(extra="forbid")

    message_id: str
    has_detection: bool = False
    detections: tuple[PublicSensitiveDetection, ...] = ()


class MessagePipelineResult(BaseModel):
    """Validated pipeline outcome for a single message.

    Attributes:
        message_id: Identifier of the analysed message.
        timestamp: Original message timestamp.
        sender: Original message sender.
        safe_message: The message with all sensitive values masked. This is the
            only form that may be sent to an external service.
        sensitive: Public sensitive-detection outcome.
        classification: Classification outcome.
        extraction: Extraction outcome.
    """

    model_config = ConfigDict(frozen=True)

    message_id: str
    timestamp: datetime
    sender: str
    safe_message: str
    sensitive: MessageSensitiveResult
    classification: ClassificationResult
    extraction: ExtractionResult


class FinalMessageResult(BaseModel):
    """Sanitized final per-message result exposed to consumers.

    Deliberately contains no raw message text and no raw sensitive value; only
    the masked/structured representations are exposed.
    """

    model_config = ConfigDict(frozen=True)

    message_id: str
    timestamp: datetime
    sender: str
    classification: ClassificationResult
    security: MessageSensitiveResult
    extracted_items: tuple[ExtractedItem, ...] = ()


class PipelineSummary(BaseModel):
    """Aggregate statistics over a full pipeline run."""

    total_messages: int
    classified_messages: int
    messages_with_sensitive: int
    messages_with_extracted_items: int
    total_extracted_items: int
    items_by_type: dict[str, int]
    rule_based_classifications: int
    llm_fallback_classifications: int
    failures: int
    processing_duration_seconds: float


class PipelineRunResult(BaseModel):
    """Validated result of processing every message in the dataset."""

    messages: tuple[MessagePipelineResult, ...]
    summary: PipelineSummary

    def to_final_results(self) -> tuple[FinalMessageResult, ...]:
        """Derive the sanitized per-message final results."""
        return tuple(
            FinalMessageResult(
                message_id=result.message_id,
                timestamp=result.timestamp,
                sender=result.sender,
                classification=result.classification,
                security=result.sensitive,
                extracted_items=result.extraction.items,
            )
            for result in self.messages
        )
