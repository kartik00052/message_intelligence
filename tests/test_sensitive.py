"""Unit tests for sensitive information detection and masking."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.models.sensitive import RiskLevel, SensitiveType
from app.services.loader import load_messages_csv
from app.services.masker import Masker
from app.services.sensitive_detector import SensitiveDetector

DETECTOR = SensitiveDetector()
MASKER = Masker()


def detect(message: str) -> list:
    return DETECTOR.detect(message)


def mask(message: str) -> str:
    return MASKER.mask(message, detect(message))


def types(message: str) -> list[str]:
    return [detection.sensitivity_type.value for detection in detect(message)]


class TestPerType:
    @pytest.mark.parametrize(
        ("message", "expected_type"),
        [
            ("Your OTP is 482913.", "one_time_password"),
            ("Your OTP: 482913", "one_time_password"),
            ("The verification code is 731204", "one_time_password"),
            ("Use password BlueRiver#29 to sign in.", "password"),
            ("Password is BlueRiver#29", "password"),
            ("My ATM PIN is 4829", "pin"),
            ("Your PIN: 2456", "pin"),
            ("The temporary access token is tok_demo_A8K29Q-53.", "authentication_token"),
            ("The access token is tok_demo_A8K29Q", "authentication_token"),
            ("Here is my sk-abc1234567 secret.", "authentication_token"),
            ("My account recovery code is RC-88-KL-19-59.", "account_recovery_code"),
            ("The backup code is ABCD-EFGH-IJKL-MNOP", "account_recovery_code"),
            ("My card number is 4111 1111 1111 1111", "payment_card_number"),
            ("My card number is 4111 1111 1111 1111-92.", "payment_card_number"),
            ("My bank account number is 006418220145", "bank_account_number"),
            ("Please note my bank account number 006418220145-38.", "bank_account_number"),
            ("My UPI ID is rahul@ybl", "upi_payment_identifier"),
            ("Pay via PhonePe at 99xxxxx00@ybl", "upi_payment_identifier"),
            ("You can contact me on 98765 43210", "private_phone_number"),
            ("Call me at +91 9876543210", "private_phone_number"),
            ("Contact me at jane.doe@gmail.com", "private_email"),
            ("My email is jane.doe@gmail.com", "private_email"),
            ("My home address is 42 Lake View Road, Chennai-45.", "private_address"),
            ("I live at 42 Lake View Road, Chennai.", "private_address"),
            ("My identification number is ID-7842-XY-94.", "identification_number"),
            ("My PAN is ABCDE1234F", "identification_number"),
            ("My recent test result says vitamin D deficiency-97.", "health_information"),
            ("My blood report shows high sugar level.", "health_information"),
            ("The card CVV is 123", "other_sensitive_credential"),
            ("My login credentials are admin:Passw0rd", "other_sensitive_credential"),
        ],
    )
    def test_detects(self, message: str, expected_type: str) -> None:
        assert expected_type in types(message)

    def test_otp_masking(self) -> None:
        assert mask("Your OTP is 482913") == "Your OTP is ******"

    def test_password_masking(self) -> None:
        assert mask("Password is BlueRiver#29") == "Password is ************"

    def test_card_masking(self) -> None:
        assert mask("My card number is 4111 1111 1111 1111") == (
            "My card number is *******************"
        )

    def test_address_masking(self) -> None:
        assert mask("My home address is 42 Lake View Road, Chennai-45.") == (
            "My home address is [REDACTED_ADDRESS]."
        )

    def test_token_masking(self) -> None:
        assert mask("The access token is tok_demo_A8K29Q") == (
            "The access token is [REDACTED_TOKEN]"
        )

    def test_risk_levels(self) -> None:
        assert detect("Your OTP is 482913.")[0].risk is RiskLevel.HIGH
        assert detect("My home address is 42 Lake View Road, Chennai.")[0].risk is RiskLevel.MEDIUM

    def test_detection_metadata(self) -> None:
        detections = detect("Your OTP is 482913.")
        assert len(detections) == 1
        d = detections[0]
        assert d.detected is True
        assert d.sensitivity_type is SensitiveType.ONE_TIME_PASSWORD
        assert d.masked_text == "******"
        assert d.recommended_action
        assert d.matched_value_internal_only == "482913"
        assert d.start >= 0 and d.end > d.start


class TestMultipleAndContext:
    def test_multiple_sensitive_values_one_message(self) -> None:
        message = "My card number is 4111 1111 1111 1111 and my OTP is 482913."
        detections = detect(message)
        assert sorted(d.sensitivity_type.value for d in detections) == [
            "one_time_password",
            "payment_card_number",
        ]
        safe = MASKER.mask(message, detections)
        assert "4111" not in safe
        assert "482913" not in safe
        assert "and my" in safe

    def test_context_is_preserved(self) -> None:
        safe = mask("Your OTP is 482913. It expires in 10 minutes.")
        assert safe == "Your OTP is ******. It expires in 10 minutes."

    def test_health_masking_preserves_prefix(self) -> None:
        safe = mask("My recent test result says vitamin D deficiency-97.")
        assert safe == "My recent [REDACTED_HEALTH]."


class TestFalsePositives:
    @pytest.mark.parametrize(
        "message",
        [
            "I have 482913 messages to read today.",
            "Please submit the report by 2026-09-04.",
            "The year is 2026 and the event is on 09-19.",
            "Use code SAVE17 for a 10% discount.",
            "Special festival discount on clothing. Use code SAVE17.",
            "The training material is on the portal.",
            "The laptop battery is fully charged.",
            "The building entrance has moved temporarily.",
            "My employee id is 42.",
            "Room 101 is booked for the demo.",
            "I ordered 2 pizzas and 3 coffees.",
            "The serial number starts with 98765 43210 digits.",
            "The password is on the portal.",
            "Call me tomorrow to discuss the plan.",
            "My address is on file.",
            "Please reply to the client email by 2026-09-04.",
            "I will send the login details separately.",
            "The webinar recording is now available.",
            "It expires in 10 minutes.",
            "The project folder was reorganized.",
        ],
    )
    def test_no_false_positive(self, message: str) -> None:
        assert detect(message) == []

    def test_newsletter_email_not_personal(self) -> None:
        message = "For support, contact help@store.example.com."
        assert detect(message) == []


class TestAlreadyMasked:
    @pytest.mark.parametrize(
        "message",
        [
            "Your OTP is ******",
            "Password is ************",
            "My card number is *******************",
            "My home address is [REDACTED_ADDRESS]",
            "The access token is [REDACTED_TOKEN]",
            "You can contact me on [REDACTED_PHONE]",
        ],
    )
    def test_already_masked_is_not_redetected(self, message: str) -> None:
        assert detect(message) == []
        assert mask(message) == message


class TestCaseVariations:
    @pytest.mark.parametrize(
        "message",
        [
            "YOUR OTP IS 482913",
            "your otp is 482913",
            "Your Otp Is 482913",
            "PASSWORD IS BLUERIVER#29",
            "Use Password blueRiver#29 to sign in.",
            "MY HOME ADDRESS IS 42 LAKE VIEW ROAD, CHENNAI.",
            "My Home Address Is 42 Lake View Road, Chennai.",
            "THE ACCESS TOKEN IS tok_demo_A8K29Q",
            "Contact Me On 98765 43210",
        ],
    )
    def test_case_insensitive_detection(self, message: str) -> None:
        assert detect(message)


class TestNoSensitiveInformation:
    def test_empty_and_whitespace(self) -> None:
        assert detect("") == []
        assert detect("   ") == []


class TestSecurity:
    RAW_VALUES = [
        "482913",
        "BlueRiver#29",
        "4829",
        "tok_demo_A8K29Q",
        "RC-88-KL-19-59",
        "4111 1111 1111 1111",
        "006418220145",
        "rahul@ybl",
        "98765 43210",
        "jane.doe@gmail.com",
        "42 Lake View Road, Chennai-45",
        "ID-7842-XY-94",
        "ABCDE1234F",
    ]
    MESSAGES = [
        "Your OTP is 482913. It expires in 10 minutes.",
        "Use password BlueRiver#29 to sign in to the test account.",
        "Your PIN is 4829",
        "The temporary access token is tok_demo_A8K29Q-53.",
        "My account recovery code is RC-88-KL-19-59.",
        "My card number is 4111 1111 1111 1111-92.",
        "My bank account number is 006418220145-38.",
        "My UPI ID is rahul@ybl",
        "You can contact me on 98765 43210-86.",
        "Contact me at jane.doe@gmail.com",
        "My home address is 42 Lake View Road, Chennai-45.",
        "My identification number is ID-7842-XY-94.",
        "My PAN is ABCDE1234F",
    ]

    def test_raw_values_never_in_masked_output(self) -> None:
        for message, raw in zip(self.MESSAGES, self.RAW_VALUES):
            safe = mask(message)
            assert raw not in safe, f"leaked {raw!r} into {safe!r}"

    def test_raw_values_never_in_serialized_detection(self) -> None:
        for message, raw in zip(self.MESSAGES, self.RAW_VALUES):
            for detection in detect(message):
                assert raw not in detection.model_dump_json()

    def test_public_detection_has_no_raw_value(self) -> None:
        for message in self.MESSAGES:
            for detection in detect(message):
                public = detection.to_public()
                assert raw_values_not_in(public.model_dump_json(), self.RAW_VALUES)


def raw_values_not_in(text: str, values: list[str]) -> bool:
    return all(value not in text for value in values)


class TestRealDataset:
    def test_detects_expected_sensitive_messages(self) -> None:
        ds = load_messages_csv(Settings().messages_csv_path)
        sensitive_ids = {m.message_id for m in ds.messages if detect(m.message)}
        assert len(sensitive_ids) == 100

    def test_no_leaks_across_real_dataset(self) -> None:
        ds = load_messages_csv(Settings().messages_csv_path)
        leaked: list[str] = []
        for message in ds.messages:
            detections = detect(message.message)
            safe = MASKER.mask(message.message, detections)
            for detection in detections:
                if detection.matched_value_internal_only in safe:
                    leaked.append(message.message_id)
        assert leaked == []
