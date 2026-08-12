"""Leak scanning for generated JSON artifacts.

Defense-in-depth check: re-run sensitive detection on every original message
and verify that no raw matched value appears anywhere in the serialized JSON
of any generated artifact. Findings never include the raw value itself - the
report only names the message, the artifact and the sensitivity type.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from app.models.message import RawMessage
from app.models.sensitive import SensitiveType
from app.services.sensitive_detector import SensitiveDetector


class LeakFinding(BaseModel):
    """A detected leak of a sensitive value into an artifact.

    Deliberately contains no raw sensitive value.
    """

    message_id: str
    artifact: str
    sensitivity_type: str


class LeakScanResult(BaseModel):
    """Outcome of scanning every artifact for raw sensitive values."""

    ok: bool
    findings: tuple[LeakFinding, ...] = ()


class LeakScanner:
    """Scans serialized artifact text for raw sensitive values."""

    def __init__(self, *, detector: SensitiveDetector | None = None) -> None:
        self._detector = detector or SensitiveDetector()

    def scan_artifact(
        self,
        *,
        artifact_name: str,
        artifact_text: str,
        messages: Mapping[str, RawMessage],
    ) -> list[LeakFinding]:
        """Scan one artifact's JSON text against every original message.

        Args:
            artifact_name: Name of the artifact, used only in findings.
            artifact_text: The serialized artifact content to scan.
            messages: Mapping of message_id to the original raw message.

        Returns:
            Findings for every raw sensitive value found in the artifact text.
        """
        findings: list[LeakFinding] = []
        for message_id, message in messages.items():
            for detection in self._detector.detect(message.message):
                value = detection.matched_value_internal_only
                if value and value in artifact_text:
                    findings.append(
                        LeakFinding(
                            message_id=message_id,
                            artifact=artifact_name,
                            sensitivity_type=SensitiveType(
                                detection.sensitivity_type
                            ).value,
                        )
                    )
        return findings

    def scan(
        self,
        *,
        artifacts: Mapping[str, str],
        messages: Sequence[RawMessage],
    ) -> LeakScanResult:
        """Scan several artifacts against the dataset messages.

        Args:
            artifacts: Mapping of artifact name to its serialized JSON text.
            messages: The original messages in the dataset.
        """
        messages_by_id = {message.message_id: message for message in messages}
        findings: list[LeakFinding] = []
        for artifact_name, artifact_text in artifacts.items():
            findings.extend(
                self.scan_artifact(
                    artifact_name=artifact_name,
                    artifact_text=artifact_text,
                    messages=messages_by_id,
                )
            )
        return LeakScanResult(ok=not findings, findings=tuple(findings))
