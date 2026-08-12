"""Models for sensitive information detection.

There are two representations:

- :class:`SensitiveDetection` is the **internal** representation. It may hold
  the raw matched value in a private attribute for the masking step. Because it
  is a Pydantic ``PrivateAttr`` it is excluded from ``model_dump()`` and
  ``model_dump_json()`` and therefore can never leak into serialized output.
- :class:`PublicSensitiveDetection` is the sanitized representation exposed to
  API responses / UI / JSON artifacts. It never contains a raw sensitive value.

Security rule: raw sensitive values must never be written to logs, JSON
artifacts, API responses, UI, reports or screenshots.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, PrivateAttr


class SensitiveType(StrEnum):
    """Category of detected sensitive information."""

    ONE_TIME_PASSWORD = "one_time_password"
    PASSWORD = "password"
    PIN = "pin"
    AUTHENTICATION_TOKEN = "authentication_token"
    ACCOUNT_RECOVERY_CODE = "account_recovery_code"
    PAYMENT_CARD_NUMBER = "payment_card_number"
    BANK_ACCOUNT_NUMBER = "bank_account_number"
    UPI_PAYMENT_IDENTIFIER = "upi_payment_identifier"
    PRIVATE_PHONE_NUMBER = "private_phone_number"
    PRIVATE_EMAIL = "private_email"
    PRIVATE_ADDRESS = "private_address"
    IDENTIFICATION_NUMBER = "identification_number"
    HEALTH_INFORMATION = "health_information"
    OTHER_SENSITIVE_CREDENTIAL = "other_sensitive_credential"


class RiskLevel(StrEnum):
    """Severity of a sensitive information exposure."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PublicSensitiveDetection(BaseModel):
    """Sanitized, safe-to-serialize detection result.

    This model is intentionally free of any raw sensitive value.
    """

    detected: bool
    sensitivity_type: SensitiveType
    risk: RiskLevel
    masked_text: str
    recommended_action: str


class SensitiveDetection(BaseModel):
    """Internal detection result used while masking a message.

    The raw matched value is stored in a private attribute only. It is
    available via :attr:`matched_value_internal_only` for the masking step but
    is excluded from every serialization, so it cannot leak to JSON/API/UI.

    ``start`` / ``end`` are the character offsets of the matched value within
    the original message; they are also private.
    """

    model_config = ConfigDict(frozen=True)

    detected: bool = True
    sensitivity_type: SensitiveType
    risk: RiskLevel
    masked_text: str
    recommended_action: str

    _matched_value_internal_only: str = PrivateAttr(default="")
    _start: int = PrivateAttr(default=0)
    _end: int = PrivateAttr(default=0)

    @classmethod
    def create(
        cls,
        *,
        sensitivity_type: SensitiveType,
        risk: RiskLevel,
        masked_text: str,
        recommended_action: str,
        matched_value_internal_only: str,
        start: int,
        end: int,
    ) -> SensitiveDetection:
        """Build an internal detection carrying a private raw value and span."""
        instance = cls(
            detected=True,
            sensitivity_type=sensitivity_type,
            risk=risk,
            masked_text=masked_text,
            recommended_action=recommended_action,
        )
        object.__setattr__(instance, "_matched_value_internal_only", matched_value_internal_only)
        object.__setattr__(instance, "_start", start)
        object.__setattr__(instance, "_end", end)
        return instance

    @property
    def matched_value_internal_only(self) -> str:
        """The raw matched value. Internal only - do not serialize or log."""
        return self._matched_value_internal_only

    @property
    def start(self) -> int:
        """Start offset (inclusive) of the matched value in the source text."""
        return self._start

    @property
    def end(self) -> int:
        """End offset (exclusive) of the matched value in the source text."""
        return self._end

    def to_public(self) -> PublicSensitiveDetection:
        """Return a sanitized representation with no raw value."""
        return PublicSensitiveDetection(
            detected=self.detected,
            sensitivity_type=self.sensitivity_type,
            risk=self.risk,
            masked_text=self.masked_text,
            recommended_action=self.recommended_action,
        )


class SensitiveAnalysis(BaseModel):
    """Result of detection + masking for a single message.

    Attributes:
        message_id: Identifier of the analysed message.
        detections: Internal detections (never serialized to public output).
        safe_message: The message with all sensitive values masked. This is the
            only form that may be sent to an external service.
        has_detection: Whether at least one sensitive value was detected.
    """

    message_id: str
    detections: tuple[SensitiveDetection, ...] = ()
    safe_message: str
    has_detection: bool = False
