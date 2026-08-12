"""Sensitive information detection.

Combines carefully designed regular expressions, contextual keyword signals and
validation heuristics (length checks, Luhn, plain-word rejection) so that
credentials and private data are flagged while ordinary numbers and content are
left alone.

Security: this module never logs raw values. Detected values are returned only
inside :class:`SensitiveDetection` private attributes.
"""

from __future__ import annotations

import re

from app.models.sensitive import (
    RiskLevel,
    SensitiveDetection,
    SensitiveType,
)

_LUHN_MIN_LENGTH = 13

_PLAIN_WORD_RE = re.compile(r"^[a-z][a-z'-]*$")
_STARS_ONLY_RE = re.compile(r"^\*{4,}$")
_REDACTED_MARKER_RE = re.compile(r"^\[REDACTED")

_PHONE_DIGITS_MIN = 10
_PHONE_DIGITS_MAX = 13

# Single-label email domains that should not be treated as UPI handles.
_EMAIL_SINGLE_LABEL_DOMAINS = {
    "gmail",
    "yahoo",
    "hotmail",
    "outlook",
    "icloud",
    "aol",
    "protonmail",
    "rediffmail",
    "live",
    "msn",
    "zoho",
    "yandex",
}


def _is_plain_word(value: str) -> bool:
    """True when ``value`` is a plain lowercase English-like word."""
    return bool(_PLAIN_WORD_RE.match(value))


def _is_plausible_secret(value: str) -> bool:
    """A plausible credential value: non-trivial and not a plain word.

    Also rejects values that are already masked (stars or ``[REDACTED...]``).
    """
    if len(value) < 4:
        return False
    if _STARS_ONLY_RE.match(value):
        return False
    if _REDACTED_MARKER_RE.match(value):
        return False
    return not _is_plain_word(value)


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum validation for payment card numbers."""
    if len(digits) < _LUHN_MIN_LENGTH or not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _card_number_luhn_valid(value: str) -> bool:
    """Luhn validation over the digits of a card-like value."""
    return _luhn_valid(re.sub(r"\D", "", value))


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int, str]:
    """Trim leading/trailing whitespace from a span and return adjusted values."""
    value = text[start:end]
    leading = len(value) - len(value.lstrip())
    value = value.strip()
    return start + leading, start + leading + len(value), value


class SensitiveDetector:
    """Detects sensitive values inside a single message text.

    The detector is stateless (all patterns are compiled once) and safe to
    reuse across the whole pipeline.
    """

    def __init__(self) -> None:
        self._otp_re = re.compile(
            r"\b(?:otp|one[\s-]?time\s*(?:password|passcode|code)?|verification\s+code"
            r"|security\s+code|auth(?:entication)?\s*code)\b"
            r"[^0-9A-Za-z]{0,6}(?:is|:|=)?[^0-9A-Za-z]{0,6}"
            r"(?P<value>\d{4,8}(?:-\d{1,3})?)(?=\b)",
            re.IGNORECASE,
        )
        self._password_re = re.compile(
            r"\b(?:pass(?:word|code|wd)|pwd)\b\s*(?:(?:is|are)\s*|[:=]\s*)?"
            r"(?P<value>[^\s.,;:!?]+)",
            re.IGNORECASE,
        )
        self._pin_re = re.compile(
            r"\b(?:pin(?!\s*code\b)|atm\s*pin|upi\s*pin)\s*(?:number)?\s*"
            r"(?:is\s*|[:=]\s*)?(?P<value>\d{4,8}(?:-\d{1,3})?)(?=\b)",
            re.IGNORECASE,
        )
        self._token_context_re = re.compile(
            r"\b(?:access\s+token|auth(?:entication)?\s+token|api\s*key|secret\s*key|token)\b"
            r"\s*(?:(?:is|are)\s*|[:=]\s*)?(?P<value>[^\s.,;:!?]+)",
            re.IGNORECASE,
        )
        self._token_prefix_re = re.compile(
            r"\b(?P<value>(?:sk-|pk-|rk-|ghp_|gho_|ghu_|xox[bap]-|tok_)[A-Za-z0-9_-]{6,})\b"
        )
        self._jwt_re = re.compile(
            r"\b(?P<value>eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,})\b"
        )
        self._recovery_re = re.compile(
            r"\b(?:account\s+recovery\s+code|recovery\s+code|backup\s+code|reset\s+code"
            r"|recovery\s+key)\b\s*(?:(?:is|are)\s*|[:=]\s*)?(?P<value>[^\s.,;:!?]+)",
            re.IGNORECASE,
        )
        self._card_context_re = re.compile(
            r"\b(?:card\s*(?:number|no)|credit\s*card|debit\s*card)\b"
            r"[^0-9A-Za-z]{0,6}(?:is|:|=)?[^0-9A-Za-z]{0,6}"
            r"(?P<value>\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}(?:-\d{1,3})?"
            r"|\d{13,19}(?:-\d{1,3})?)(?=\b)",
            re.IGNORECASE,
        )
        self._card_spaced_re = re.compile(
            r"\b(?P<value>\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}(?:-\d{1,3})?)\b"
        )
        self._bank_re = re.compile(
            r"\b(?:bank\s+account(?:\s+number)?|account\s+number|account\s+no"
            r"|a\s*/\s*c\s+no|savings\s+account)\b\s*(?:is|:|=)?\s*"
            r"(?P<value>\d{9,18}(?:-\d{1,3})?)",
            re.IGNORECASE,
        )
        self._upi_context_re = re.compile(
            r"\b(?:upi(?:\s+(?:id|address|number))?|phonepe|google\s*pay|gpay|paytm|bhim)\b"
            r"\s*(?:is|:|=)?\s*(?P<value>[A-Za-z0-9._-]{2,}@[A-Za-z][A-Za-z0-9_.-]*)",
            re.IGNORECASE,
        )
        self._upi_fmt_re = re.compile(
            r"\b(?P<value>[A-Za-z0-9._-]{2,}@[A-Za-z]{2,})(?!\.\w)\b"
        )
        self._phone_context_re = re.compile(
            r"\b(?:call(?:\s+me)?(?:\s+(?:at|on))?|contact(?:\s+me)?(?:\s+(?:at|on))?"
            r"|reach(?:\s+me)?\s+at|phone(?:\s+number)?|mobile(?:\s+number)?"
            r"|whatsapp(?:\s+number)?|cell(?:\s+number)?)\b\s*(?:is\s*|[:=]\s*)?"
            r"(?P<value>[+()\d][\d\s()-]{7,19})",
            re.IGNORECASE,
        )
        self._email_re = re.compile(
            r"\b(?P<value>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"
        )
        self._email_context_re = re.compile(
            r"my\s*(?:personal\s*)?(?:e-?mail|mail)\b"
            r"|e-?mail\s+id|e-?mail\s+address|personal\s+e-?mail"
            r"|contact\s+me|reach\s+me|mail\s+me|e-?mail\s+me"
            r"|send\s+it\s+to|write\s+to\s+me|this\s+is\s+my|reachable\s+at",
            re.IGNORECASE,
        )
        self._address_re = re.compile(
            r"\b(?:(?:my|home|my\s+home|residential|my\s+residential|private)\s+address"
            r"|my\s+residence|residing\s+at|i\s+live\s+at|i\s+stay\s+at|my\s+place)\b"
            r"\s*(?:is|:|=)?\s*(?P<value>[^.;!?\n]+)",
            re.IGNORECASE,
        )
        self._pan_re = re.compile(r"\b(?P<value>[A-Z]{5}\d{4}[A-Z])\b")
        self._aadhaar_re = re.compile(
            r"\b(?:aadhaar|aadhar|uid\s*number|12[\s-]?digit\s*(?:id|number))\b"
            r"\s*(?:is|:|=)?\s*(?P<value>\d{4}\s*\d{4}\s*\d{4}|\d{12})(?=\b)",
            re.IGNORECASE,
        )
        self._passport_re = re.compile(
            r"\b(?:passport(?:\s+(?:number|no))?)\b\s*(?:is|:|=)?\s*"
            r"(?P<value>[A-Z]\d{7}|[A-Z]\d{8}|[A-Z]\d{7}[A-Z])",
            re.IGNORECASE,
        )
        self._generic_id_re = re.compile(
            r"\b(?:id\s+number|id\s+no|identification\s+number|id\s+card"
            r"|identity\s+(?:number|card)|ssn|social\s+security(?:\s+number)?"
            r"|voter\s*id|driving\s+licen[cs]e|driver'?s\s+licen[cs]e"
            r"|licen[cs]e\s+number)\b\s*(?:is|:|=)?\s*"
            r"(?P<value>[A-Z0-9][A-Z0-9 -]{3,})",
            re.IGNORECASE,
        )
        self._health_re = re.compile(
            r"\b(?:test\s+result|blood\s+(?:report|test|pressure|sugar)"
            r"|diagnos\w*|prescription|medical\s+(?:report|history|condition)"
            r"|vitamin\s+[A-Z]|allerg\w*|diabet\w*|thyroid|cholesterol"
            r"|haemoglobin|hemoglobin|sugar\s+level|mental\s+health|depression"
            r"|anxiety|cancer|pregnancy|hiv\b|std\b|fever|surgery|heart\s+(?:disease|condition))\b",
            re.IGNORECASE,
        )
        self._cvv_re = re.compile(
            r"\b(?:cvv|cvc)\b\s*(?:is\s*|[:=]\s*)?(?P<value>\d{3,4})(?=\b)",
            re.IGNORECASE,
        )
        self._other_re = re.compile(
            r"\b(?:security\s+question|net\s*banking\s*password|banking\s+password"
            r"|login\s+(?:details|password|credentials?)|credentials?|seed\s+phrase"
            r"|private\s+key|wallet\s+key|backup\s+phrase|master\s+password"
            r"|wifi\s+password|router\s+password)\b\s*(?:(?:is|are)\s*|[:=]\s*)?"
            r"(?P<value>:?[^\s.,;!?]+)",
            re.IGNORECASE,
        )

    # ------------------------------------------------------------------ public

    def detect(self, text: str) -> list[SensitiveDetection]:
        """Detect sensitive values in ``text``.

        Returns a list of internal detections sorted by position. The list is
        empty when no sensitive information is found.
        """
        if not text or not text.strip():
            return []
        detections: list[SensitiveDetection] = []
        detections.extend(self._detect_otp(text))
        detections.extend(self._detect_password(text))
        detections.extend(self._detect_pin(text))
        detections.extend(self._detect_token(text))
        detections.extend(self._detect_recovery(text))
        detections.extend(self._detect_card(text))
        detections.extend(self._detect_bank(text))
        detections.extend(self._detect_upi(text))
        detections.extend(self._detect_phone(text))
        detections.extend(self._detect_email(text))
        detections.extend(self._detect_address(text))
        detections.extend(self._detect_identification(text))
        detections.extend(self._detect_health(text))
        detections.extend(self._detect_other(text))

        return self._dedupe(detections)

    # ---------------------------------------------------------------- per type

    def _detect_otp(self, text: str) -> list[SensitiveDetection]:
        return [
            self._make(SensitiveType.ONE_TIME_PASSWORD, text=text, match=match)
            for match in self._otp_re.finditer(text)
        ]

    def _detect_password(self, text: str) -> list[SensitiveDetection]:
        return [
            self._make(SensitiveType.PASSWORD, text=text, match=match)
            for match in self._password_re.finditer(text)
            if _is_plausible_secret(match.group("value"))
        ]

    def _detect_pin(self, text: str) -> list[SensitiveDetection]:
        return [
            self._make(SensitiveType.PIN, text=text, match=match)
            for match in self._pin_re.finditer(text)
        ]

    def _detect_token(self, text: str) -> list[SensitiveDetection]:
        out = [
            self._make(SensitiveType.AUTHENTICATION_TOKEN, text=text, match=match)
            for match in self._token_context_re.finditer(text)
            if _is_plausible_secret(match.group("value"))
        ]
        out.extend(
            self._make(SensitiveType.AUTHENTICATION_TOKEN, text=text, match=match)
            for match in self._token_prefix_re.finditer(text)
        )
        out.extend(
            self._make(SensitiveType.AUTHENTICATION_TOKEN, text=text, match=match)
            for match in self._jwt_re.finditer(text)
        )
        return out

    def _detect_recovery(self, text: str) -> list[SensitiveDetection]:
        return [
            self._make(SensitiveType.ACCOUNT_RECOVERY_CODE, text=text, match=match)
            for match in self._recovery_re.finditer(text)
            if _is_plausible_secret(match.group("value"))
        ]

    def _detect_card(self, text: str) -> list[SensitiveDetection]:
        out = [
            self._make(SensitiveType.PAYMENT_CARD_NUMBER, text=text, match=match)
            for match in self._card_context_re.finditer(text)
        ]
        out.extend(
            self._make(SensitiveType.PAYMENT_CARD_NUMBER, text=text, match=match)
            for match in self._card_spaced_re.finditer(text)
            if _card_number_luhn_valid(match.group("value"))
        )
        return out

    def _detect_bank(self, text: str) -> list[SensitiveDetection]:
        return [
            self._make(SensitiveType.BANK_ACCOUNT_NUMBER, text=text, match=match)
            for match in self._bank_re.finditer(text)
        ]

    def _detect_upi(self, text: str) -> list[SensitiveDetection]:
        out: list[SensitiveDetection] = []
        seen_spans: set[tuple[int, int]] = set()
        for match in self._upi_context_re.finditer(text):
            value = match.group("value")
            if self._looks_like_upi(value):
                out.append(self._make(SensitiveType.UPI_PAYMENT_IDENTIFIER, text=text, match=match))
                seen_spans.add((match.start("value"), match.end("value")))
        for match in self._upi_fmt_re.finditer(text):
            span = (match.start("value"), match.end("value"))
            if span in seen_spans:
                continue
            if self._looks_like_upi(match.group("value")):
                out.append(self._make(SensitiveType.UPI_PAYMENT_IDENTIFIER, text=text, match=match))
        return out

    def _detect_phone(self, text: str) -> list[SensitiveDetection]:
        out = []
        for match in self._phone_context_re.finditer(text):
            start, end, value = _trimmed_span(text, match.start("value"), match.end("value"))
            digits = re.sub(r"\D", "", value)
            if _PHONE_DIGITS_MIN <= len(digits) <= _PHONE_DIGITS_MAX:
                out.append(
                    self._make(
                        SensitiveType.PRIVATE_PHONE_NUMBER,
                        start=start,
                        end=end,
                        value=value,
                    )
                )
        return out

    def _detect_email(self, text: str) -> list[SensitiveDetection]:
        out = []
        for match in self._email_re.finditer(text):
            if self._looks_like_upi(match.group("value")):
                continue
            window_start = max(0, match.start("value") - 80)
            if self._email_context_re.search(text[window_start : match.start("value")]):
                out.append(self._make(SensitiveType.PRIVATE_EMAIL, text=text, match=match))
        return out

    def _detect_address(self, text: str) -> list[SensitiveDetection]:
        out = []
        for match in self._address_re.finditer(text):
            start, end, value = _trimmed_span(text, match.start("value"), match.end("value"))
            if self._is_address_value(value):
                out.append(
                    self._make(
                        SensitiveType.PRIVATE_ADDRESS,
                        start=start,
                        end=end,
                        value=value,
                    )
                )
        return out

    def _detect_identification(self, text: str) -> list[SensitiveDetection]:
        out = []
        for match in self._pan_re.finditer(text):
            out.append(self._make(SensitiveType.IDENTIFICATION_NUMBER, text=text, match=match))
        for match in self._aadhaar_re.finditer(text):
            out.append(self._make(SensitiveType.IDENTIFICATION_NUMBER, text=text, match=match))
        for match in self._passport_re.finditer(text):
            out.append(self._make(SensitiveType.IDENTIFICATION_NUMBER, text=text, match=match))
        for match in self._generic_id_re.finditer(text):
            start, end, value = _trimmed_span(text, match.start("value"), match.end("value"))
            if re.search(r"\d", value) and not _is_plain_word(value):
                out.append(
                    self._make(
                        SensitiveType.IDENTIFICATION_NUMBER,
                        start=start,
                        end=end,
                        value=value,
                    )
                )
        return out

    def _detect_health(self, text: str) -> list[SensitiveDetection]:
        out = []
        for match in self._health_re.finditer(text):
            clause_end = self._clause_end(text, match.start())
            start, end, value = _trimmed_span(text, match.start(), clause_end)
            if len(value) >= 3:
                out.append(
                    self._make(
                        SensitiveType.HEALTH_INFORMATION,
                        start=start,
                        end=end,
                        value=value,
                    )
                )
        return out

    def _detect_other(self, text: str) -> list[SensitiveDetection]:
        out = [
            self._make(SensitiveType.OTHER_SENSITIVE_CREDENTIAL, text=text, match=match)
            for match in self._cvv_re.finditer(text)
        ]
        out.extend(
            self._make(SensitiveType.OTHER_SENSITIVE_CREDENTIAL, text=text, match=match)
            for match in self._other_re.finditer(text)
            if _is_plausible_secret(match.group("value"))
        )
        return out

    # ----------------------------------------------------------------- helpers

    def _make(
        self,
        sensitivity_type: SensitiveType,
        *,
        text: str | None = None,
        match: re.Match[str] | None = None,
        start: int | None = None,
        end: int | None = None,
        value: str | None = None,
    ) -> SensitiveDetection:
        if match is not None:
            start = match.start("value")
            end = match.end("value")
            value = match.group("value")
        assert start is not None and end is not None and value is not None
        return SensitiveDetection.create(
            sensitivity_type=sensitivity_type,
            risk=_RISK_BY_TYPE[sensitivity_type],
            masked_text=self._mask_for(sensitivity_type, value),
            recommended_action=_RECOMMENDED_ACTIONS[sensitivity_type],
            matched_value_internal_only=value,
            start=start,
            end=end,
        )

    @staticmethod
    def _mask_for(sensitivity_type: SensitiveType, value: str) -> str:
        if sensitivity_type in _STAR_MASKED_TYPES:
            return "*" * len(value)
        return _BRACKET_MASKS[sensitivity_type]

    @staticmethod
    def _clause_end(text: str, start: int) -> int:
        """Index of the end of the clause starting at ``start``."""
        match = re.search(r"[.;!?\n]", text[start:])
        if match:
            return start + match.start()
        return len(text)

    @staticmethod
    def _is_address_value(value: str) -> bool:
        if _REDACTED_MARKER_RE.match(value):
            return False
        if len(value) < 6:
            return False
        if re.search(r"\d", value):
            return True
        return "," in value or len(value) >= 12

    @staticmethod
    def _looks_like_upi(value: str) -> bool:
        if "@" not in value:
            return False
        local, _, domain = value.rpartition("@")
        if not local or not domain:
            return False
        if "." in domain:
            return False
        return domain.lower() not in _EMAIL_SINGLE_LABEL_DOMAINS

    @staticmethod
    def _dedupe(detections: list[SensitiveDetection]) -> list[SensitiveDetection]:
        ordered = sorted(detections, key=lambda d: (d.start, -(d.end - d.start)))
        accepted: list[SensitiveDetection] = []
        for detection in ordered:
            if any(
                detection.start >= other.start and detection.end <= other.end
                for other in accepted
            ):
                continue
            accepted.append(detection)
        return accepted


_STAR_MASKED_TYPES = {
    SensitiveType.ONE_TIME_PASSWORD,
    SensitiveType.PASSWORD,
    SensitiveType.PIN,
    SensitiveType.ACCOUNT_RECOVERY_CODE,
    SensitiveType.PAYMENT_CARD_NUMBER,
    SensitiveType.BANK_ACCOUNT_NUMBER,
    SensitiveType.UPI_PAYMENT_IDENTIFIER,
}

_BRACKET_MASKS: dict[SensitiveType, str] = {
    SensitiveType.AUTHENTICATION_TOKEN: "[REDACTED_TOKEN]",
    SensitiveType.PRIVATE_PHONE_NUMBER: "[REDACTED_PHONE]",
    SensitiveType.PRIVATE_EMAIL: "[REDACTED_EMAIL]",
    SensitiveType.PRIVATE_ADDRESS: "[REDACTED_ADDRESS]",
    SensitiveType.IDENTIFICATION_NUMBER: "[REDACTED_ID]",
    SensitiveType.HEALTH_INFORMATION: "[REDACTED_HEALTH]",
    SensitiveType.OTHER_SENSITIVE_CREDENTIAL: "[REDACTED]",
}

_RISK_BY_TYPE: dict[SensitiveType, RiskLevel] = {
    SensitiveType.ONE_TIME_PASSWORD: RiskLevel.HIGH,
    SensitiveType.PASSWORD: RiskLevel.HIGH,
    SensitiveType.PIN: RiskLevel.HIGH,
    SensitiveType.AUTHENTICATION_TOKEN: RiskLevel.HIGH,
    SensitiveType.ACCOUNT_RECOVERY_CODE: RiskLevel.HIGH,
    SensitiveType.PAYMENT_CARD_NUMBER: RiskLevel.HIGH,
    SensitiveType.BANK_ACCOUNT_NUMBER: RiskLevel.HIGH,
    SensitiveType.UPI_PAYMENT_IDENTIFIER: RiskLevel.MEDIUM,
    SensitiveType.PRIVATE_PHONE_NUMBER: RiskLevel.MEDIUM,
    SensitiveType.PRIVATE_EMAIL: RiskLevel.MEDIUM,
    SensitiveType.PRIVATE_ADDRESS: RiskLevel.MEDIUM,
    SensitiveType.IDENTIFICATION_NUMBER: RiskLevel.HIGH,
    SensitiveType.HEALTH_INFORMATION: RiskLevel.HIGH,
    SensitiveType.OTHER_SENSITIVE_CREDENTIAL: RiskLevel.HIGH,
}

_RECOMMENDED_ACTIONS: dict[SensitiveType, str] = {
    SensitiveType.ONE_TIME_PASSWORD: (
        "Do not share the one-time password; treat it as compromised if forwarded."
    ),
    SensitiveType.PASSWORD: "Rotate the password and enable two-factor authentication.",
    SensitiveType.PIN: "Do not share the PIN; change it immediately if exposed.",
    SensitiveType.AUTHENTICATION_TOKEN: "Revoke the token and rotate the related credentials.",
    SensitiveType.ACCOUNT_RECOVERY_CODE: (
        "Keep recovery codes offline; regenerate them if exposed."
    ),
    SensitiveType.PAYMENT_CARD_NUMBER: (
        "Do not store the full card number; contact the issuer if compromised."
    ),
    SensitiveType.BANK_ACCOUNT_NUMBER: (
        "Mask the account number and monitor the account for unusual activity."
    ),
    SensitiveType.UPI_PAYMENT_IDENTIFIER: (
        "Do not share UPI identifiers; enable app lock and transaction alerts."
    ),
    SensitiveType.PRIVATE_PHONE_NUMBER: (
        "Mask the personal number and share it only with trusted parties."
    ),
    SensitiveType.PRIVATE_EMAIL: (
        "Avoid sharing personal email addresses in public channels."
    ),
    SensitiveType.PRIVATE_ADDRESS: (
        "Mask the residential address; do not share it in public messages."
    ),
    SensitiveType.IDENTIFICATION_NUMBER: (
        "Protect the identification number and report suspected misuse."
    ),
    SensitiveType.HEALTH_INFORMATION: (
        "Keep health details confidential and share only with medical staff."
    ),
    SensitiveType.OTHER_SENSITIVE_CREDENTIAL: (
        "Protect this credential; rotate or revoke it if it may be exposed."
    ),
}
