"""Masking of sensitive values in message text.

Applies the redaction decided by the detector, preserving all surrounding
context. The output of :meth:`Masker.mask` is the only form of a message that
may be sent to an external service, logged, or stored.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.models.sensitive import SensitiveDetection


class Masker:
    """Replaces detected sensitive spans with their masked representations."""

    def mask(self, text: str, detections: Sequence[SensitiveDetection]) -> str:
        """Return ``text`` with every detection's span replaced by its mask.

        Spans are applied from right to left so that offsets to the left of an
        already-applied span remain valid. All non-sensitive context is kept.
        """
        if not detections:
            return text

        result = text
        for detection in sorted(detections, key=lambda d: d.start, reverse=True):
            result = (
                result[: detection.start]
                + detection.masked_text
                + result[detection.end :]
            )
        return result
