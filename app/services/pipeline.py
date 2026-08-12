"""End-to-end pipeline orchestrator.

Runs every stage for each message in order:

    raw message
        -> sensitive detection
        -> masking (only the masked form reaches classification/extraction)
        -> classification
        -> task/event extraction

Every message in the dataset is processed - none are skipped silently. A single
LLM failure never crashes the run: it is counted in the summary and the
deterministic result is kept. The orchestrator is independent of FastAPI and of
the output-writing step; it produces fully validated
:class:`MessagePipelineResult` objects.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence

from app.models.message import RawMessage
from app.models.pipeline import (
    MessagePipelineResult,
    MessageSensitiveResult,
    PipelineRunResult,
    PipelineSummary,
)
from app.models.task_event import ItemType
from app.services.classifier import ClassifierMethod, MessageClassifier
from app.services.extractor import MessageExtractor
from app.services.masker import Masker
from app.services.sensitive_detector import SensitiveDetector


class PipelineRunner:
    """Combines detection, masking, classification and extraction.

    Every stage is swappable for testing. By default no LLM is configured
    (offline mode), so the classifier and extractor use their deterministic
    rule paths only.
    """

    def __init__(
        self,
        *,
        detector: SensitiveDetector | None = None,
        masker: Masker | None = None,
        classifier: MessageClassifier | None = None,
        extractor: MessageExtractor | None = None,
    ) -> None:
        self._detector = detector or SensitiveDetector()
        self._masker = masker or Masker()
        self._classifier = classifier or MessageClassifier()
        self._extractor = extractor or MessageExtractor()

    def run(self, messages: Sequence[RawMessage]) -> PipelineRunResult:
        """Analyse every message and summarize the results."""
        started = time.perf_counter()
        results = tuple(self.analyze(message) for message in messages)
        duration = time.perf_counter() - started
        summary = _summarize(results, duration, self)
        return PipelineRunResult(messages=results, summary=summary)

    def analyze(self, message: RawMessage) -> MessagePipelineResult:
        """Analyse a single message through every pipeline stage."""
        detections = tuple(self._detector.detect(message.message))
        safe_message = self._masker.mask(message.message, detections)

        return MessagePipelineResult(
            message_id=message.message_id,
            timestamp=message.timestamp,
            sender=message.sender,
            safe_message=safe_message,
            sensitive=MessageSensitiveResult(
                message_id=message.message_id,
                has_detection=bool(detections),
                detections=tuple(detection.to_public() for detection in detections),
            ),
            classification=self._classifier.classify(message),
            extraction=self._extractor.extract(message),
        )


def _summarize(
    results: Sequence[MessagePipelineResult],
    duration: float,
    runner: PipelineRunner,
) -> PipelineSummary:
    by_type: Counter[str] = Counter()
    rule_based = 0
    llm_fallback = 0
    for result in results:
        for item in result.extraction.items:
            by_type[ItemType(item.type).value] += 1
        if result.classification.method is ClassifierMethod.RULE_BASED:
            rule_based += 1
        elif result.classification.method is ClassifierMethod.LLM_FALLBACK:
            llm_fallback += 1
    return PipelineSummary(
        total_messages=len(results),
        classified_messages=len(results),
        messages_with_sensitive=sum(
            1 for result in results if result.sensitive.has_detection
        ),
        messages_with_extracted_items=sum(
            1 for result in results if result.extraction.items
        ),
        total_extracted_items=sum(by_type.values()),
        items_by_type=dict(sorted(by_type.items())),
        rule_based_classifications=rule_based,
        llm_fallback_classifications=llm_fallback,
        failures=runner._classifier.llm_failures + runner._extractor.llm_failures,
        processing_duration_seconds=round(duration, 4),
    )
