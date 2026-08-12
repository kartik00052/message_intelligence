"""Unit tests for deterministic task/event extraction and the LLM fallback path."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import pytest

from app.models.message import RawMessage
from app.models.task_event import ExtractedItem, ExtractorMethod, ItemType, Priority
from app.services.extractor import (
    BaseLLMExtractor,
    LLMResponseError,
    MessageExtractor,
    MessageExtractorLLM,
    clean_clause,
    make_item_id,
    normalize_time,
    parse_llm_response,
)


def make_message(text: str, message_id: str = "T001") -> RawMessage:
    """Build a valid ``RawMessage`` for tests."""
    return RawMessage(
        message_id=message_id,
        timestamp=datetime(2026, 9, 1, 8, 0, 0),
        sender="Meera",
        message=text,
    )


def items_payload(*items: dict[str, object], message_id: str = "T001") -> str:
    return json.dumps({"message_id": message_id, "items": list(items)})


def one_item(**overrides: object) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "task",
        "title": "submit the report",
        "description": None,
        "date": None,
        "deadline": "2026-09-04",
        "time": None,
        "person": None,
        "priority": "unknown",
    }
    item.update(overrides)
    return item


class StubLLM(MessageExtractorLLM):
    """Deterministic LLM stub that records calls and honours a failure mode."""

    def __init__(self, *, payload: str | object, fail: bool = False) -> None:
        self._payload = payload
        self._fail = fail
        self.calls: list[tuple[str, str]] = []

    def extract(
        self, *, message_id: str, safe_message: str
    ) -> tuple[ExtractedItem, ...] | None:
        self.calls.append((message_id, safe_message))
        if self._fail:
            return None
        payload = self._payload(message_id) if callable(self._payload) else self._payload
        return parse_llm_response(message_id, str(payload))


class RecordingLLM(BaseLLMExtractor):
    """Concrete provider used to inspect the built prompt."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def _invoke(self, *, prompt: str) -> str:
        self.last_prompt = prompt
        return items_payload()


class TestNormalizeTime:
    def test_24h_times(self) -> None:
        assert normalize_time("9:00") == "09:00"
        assert normalize_time("09:00") == "09:00"
        assert normalize_time("14:30") == "14:30"
        assert normalize_time("00:05") == "00:05"

    def test_meridiem_times(self) -> None:
        assert normalize_time("9 AM") == "09:00"
        assert normalize_time("8 pm") == "20:00"
        assert normalize_time("12 PM") == "12:00"
        assert normalize_time("12 am") == "00:00"
        assert normalize_time("12:30pm") == "12:30"

    def test_invalid_times_rejected(self) -> None:
        for bad in ("25:00", "9:99", "banana", "9:00:00"):
            with pytest.raises(ValueError):
                normalize_time(bad)


class TestMakeItemId:
    def test_single_occurrence(self) -> None:
        assert make_item_id("MSG_0001", ItemType.TASK) == "TASK_MSG_0001"
        assert make_item_id("MSG_0001", ItemType.EVENT) == "EVENT_MSG_0001"

    def test_repeat_occurrence(self) -> None:
        assert make_item_id("MSG_0001", ItemType.TASK, 2) == "TASK_MSG_0001-2"


class TestCleanClause:
    def test_strips_prefixes_and_softeners(self) -> None:
        assert clean_clause("For today: Please submit the weekly report") == (
            "submit the weekly report"
        )
        assert clean_clause("Can you help? Don't forget to pay the bill") == "pay the bill"
        assert clean_clause("I need you to review the model results") == "review the model results"

    def test_handles_semicolon_deadline_split(self) -> None:
        assert clean_clause("Don't forget to pay the electricity bill;") == (
            "pay the electricity bill"
        )

    def test_lowercases_first_character(self) -> None:
        assert clean_clause("Complete the onboarding form") == "complete the onboarding form"


class TestRuleExtraction:
    def test_calendar_update_event(self) -> None:
        result = MessageExtractor().extract(
            make_message(
                "For today: Calendar update: family dinner, 2026-09-19 at 10:00, the library."
            )
        )
        assert result.method is ExtractorMethod.RULE_BASED
        assert len(result.items) == 1
        item = result.items[0]
        assert item.type is ItemType.EVENT
        assert item.title == "family dinner"
        assert item.date == date(2026, 9, 19)
        assert item.time == "10:00"
        assert item.description == "the library"
        assert item.item_id == "EVENT_T001"
        assert item.source_message_id == "T001"

    def test_reminder_happens_on(self) -> None:
        result = MessageExtractor().extract(
            make_message(
                "FYI: Reminder: mentor catch-up happens on 2026-09-16 at 11:00 in the city clinic."
            )
        )
        assert len(result.items) == 1
        item = result.items[0]
        assert item.type is ItemType.REMINDER
        assert item.title == "mentor catch-up"
        assert item.date == date(2026, 9, 16)
        assert item.time == "11:00"
        assert item.description == "the city clinic"

    def test_scheduled_meeting(self) -> None:
        result = MessageExtractor().extract(
            make_message("The client discussion is scheduled for 2026-09-07 at 14:00 in Zoom.")
        )
        assert len(result.items) == 1
        item = result.items[0]
        assert item.type is ItemType.MEETING
        assert item.title == "client discussion"
        assert item.date == date(2026, 9, 7)
        assert item.time == "14:00"
        assert item.description == "Zoom"
        assert item.deadline is None

    def test_join_request(self) -> None:
        result = MessageExtractor().extract(
            make_message(
                "Please join the internship orientation on 2026-09-18, 9:00 at Conference Room 2."
            )
        )
        assert len(result.items) == 1
        item = result.items[0]
        assert item.type is ItemType.EVENT
        assert item.title == "internship orientation"
        assert item.date == date(2026, 9, 18)
        assert item.time == "09:00"
        assert item.description == "Conference Room 2"

    def test_availability_request(self) -> None:
        result = MessageExtractor().extract(
            make_message(
                "Are you available for the design review at 14:00 on 2026-09-15? "
                "Location: the main office."
            )
        )
        assert len(result.items) == 1
        item = result.items[0]
        assert item.type is ItemType.MEETING
        assert item.title == "design review"
        assert item.date == date(2026, 9, 15)
        assert item.time == "14:00"
        assert item.description == "the main office"

    @pytest.mark.parametrize(
        "message",
        [
            "Please reply to the client email by 2026-09-04.",
            "Can you update the project tracker before 2026-09-04?",
            "Don't forget to pay the electricity bill; deadline is 2026-09-09.",
            "Complete the onboarding form is due on 2026-09-10.",
            "For today: I need you to renew the library book by 2026-09-07.",
            "Please submit the weekly report by 2026-09-08.",
            "Can you help? Don't forget to email the signed document; deadline is 2026-09-04.",
            "For today: Can you review the privacy checklist before 2026-09-06?",
            "Please confirm the interview slot by 2026-09-09.",
            "Send the expense receipt is due on 2026-09-07.",
        ],
    )
    def test_deadline_tasks(self, message: str) -> None:
        result = MessageExtractor().extract(make_message(message))
        assert result.method is ExtractorMethod.RULE_BASED
        assert len(result.items) == 1
        item = result.items[0]
        assert item.type is ItemType.TASK
        assert item.deadline is not None
        assert item.title

    def test_deadline_task_fields(self) -> None:
        result = MessageExtractor().extract(
            make_message("For today: I need you to back up the project files by 2026-09-08.")
        )
        item = result.items[0]
        assert item.title == "back up the project files"
        assert item.deadline == date(2026, 9, 8)
        assert item.date is None
        assert item.person is None

    def test_call_request_with_person(self) -> None:
        result = MessageExtractor().extract(
            make_message("Please call Maya when you are free.")
        )
        assert len(result.items) == 1
        item = result.items[0]
        assert item.type is ItemType.TASK
        assert item.title == "call maya"
        assert item.person == "Maya"
        assert item.priority is Priority.LOW

    def test_call_with_deadline_uses_deadline_rule(self) -> None:
        result = MessageExtractor().extract(
            make_message("Don't forget to call the service centre; deadline is 2026-09-09.")
        )
        item = result.items[0]
        assert item.type is ItemType.TASK
        assert item.title == "call the service centre"
        assert item.deadline == date(2026, 9, 9)
        assert item.person is None

    def test_priority_detection(self) -> None:
        high = MessageExtractor().extract(make_message("Please submit ASAP by 2026-09-04."))
        assert high.items[0].priority is Priority.HIGH

        low = MessageExtractor().extract(make_message("No rush: submit the report by 2026-09-04."))
        assert low.items[0].priority is Priority.LOW

        unknown = MessageExtractor().extract(
            make_message("Please submit the report by 2026-09-04.")
        )
        assert unknown.items[0].priority is Priority.UNKNOWN

    def test_message_prefixes_do_not_break_rules(self) -> None:
        for prefix in (
            "For today: ",
            "Quick update: ",
            "FYI: ",
            "Important: ",
            "Can you help? ",
            "Please note: ",
            "One more thing: ",
            "Just checking\u2014",
            "Hi, ",
        ):
            result = MessageExtractor().extract(
                make_message(
                    prefix + "The product demo is scheduled for 2026-09-07 at 10:00 in Zoom."
                )
            )
            assert len(result.items) == 1, prefix
            assert result.items[0].title == "product demo"

    @pytest.mark.parametrize(
        "message",
        [
            "Tomorrow is a public holiday.",
            "The report may be needed tomorrow.",
            "The review could be Friday afternoon.",
            "The cafeteria closes at 8 PM.",
            "Let us meet sometime next week.",
            "The training material is on the portal.",
            "For my profile, i am vegetarian.",
            "Book movie tickets today and receive cashback. Use code SAVE26.",
            "Flash sale on laptops starts at 6 PM. Use code SAVE43.",
            "The library has extended weekend hours.",
        ],
    )
    def test_no_extraction_for_noise(self, message: str) -> None:
        result = MessageExtractor().extract(make_message(message))
        assert result.method is ExtractorMethod.NONE
        assert result.items == ()
        assert result.reason

    def test_message_id_preserved(self) -> None:
        result = MessageExtractor().extract(
            make_message("Please submit the report by 2026-09-04.", "MSG_7777")
        )
        assert result.message_id == "MSG_7777"
        assert result.items[0].item_id == "TASK_MSG_7777"

    def test_empty_message(self) -> None:
        result = MessageExtractor().extract(make_message("   "))
        assert result.method is ExtractorMethod.NONE
        assert result.items == ()


class TestRelativeDates:
    """Relative expressions are resolved against the message timestamp."""

    REFERENCE = datetime(2026, 9, 1, 8, 0, 0)  # Tuesday

    def test_tomorrow_deadline(self) -> None:
        result = MessageExtractor().extract(
            make_message("Please submit the report by tomorrow.")
        )
        assert len(result.items) == 1
        assert result.items[0].type is ItemType.TASK
        assert result.items[0].deadline == date(2026, 9, 2)
        assert result.items[0].item_id == "TASK_T001"

    def test_tomorrow_meeting(self) -> None:
        result = MessageExtractor().extract(
            make_message("The client discussion is scheduled for tomorrow at 14:00 in Zoom.")
        )
        assert len(result.items) == 1
        item = result.items[0]
        assert item.type is ItemType.MEETING
        assert item.date == date(2026, 9, 2)
        assert item.time == "14:00"
        assert item.deadline is None

    def test_weekday_reminder(self) -> None:
        result = MessageExtractor().extract(
            make_message(
                "Reminder: doctor appointment happens on Friday at 09:00 in the city clinic."
            )
        )
        assert len(result.items) == 1
        item = result.items[0]
        assert item.type is ItemType.REMINDER
        assert item.date == date(2026, 9, 4)  # Tuesday -> next Friday

    def test_in_two_days_event(self) -> None:
        result = MessageExtractor().extract(
            make_message("Please join the orientation in two days at 10:00 at the auditorium.")
        )
        assert len(result.items) == 1
        assert result.items[0].date == date(2026, 9, 3)
        assert result.items[0].time == "10:00"

    def test_in_three_days_deadline(self) -> None:
        result = MessageExtractor().extract(
            make_message("Please send the report; deadline is in three days.")
        )
        assert result.items[0].deadline == date(2026, 9, 4)

    def test_next_friday_meeting(self) -> None:
        result = MessageExtractor().extract(
            make_message("The sync is scheduled for next friday at 09:00 in the meeting room.")
        )
        assert result.items[0].date == date(2026, 9, 4)

    def test_available_on_weekday(self) -> None:
        result = MessageExtractor().extract(
            make_message(
                "Are you available for the design review at 14:00 on Friday? "
                "Location: the main office."
            )
        )
        assert len(result.items) == 1
        assert result.items[0].type is ItemType.MEETING
        assert result.items[0].date == date(2026, 9, 4)

    def test_join_without_on(self) -> None:
        result = MessageExtractor().extract(
            make_message("Please join the kickoff tomorrow at 15:00 at the training hall.")
        )
        assert len(result.items) == 1
        assert result.items[0].date == date(2026, 9, 2)
        assert result.items[0].time == "15:00"

    def test_reference_is_message_timestamp_not_system_date(self) -> None:
        message = RawMessage(
            message_id="HIST",
            timestamp=datetime(2020, 1, 1, 9, 0, 0),
            sender="Meera",
            message="Please submit the report by tomorrow.",
        )
        item = MessageExtractor().extract(message).items[0]
        assert item.deadline == date(2020, 1, 2)


class TestAmbiguousDates:
    """Vague or conditional scheduling is never turned into a definite date."""

    def test_friday_afternoon_speculation(self) -> None:
        result = MessageExtractor().extract(
            make_message("The review could be Friday afternoon.")
        )
        assert result.method is ExtractorMethod.NONE
        assert result.items == ()

    def test_report_may_be_needed_tomorrow(self) -> None:
        result = MessageExtractor().extract(make_message("The report may be needed tomorrow."))
        assert result.items == ()

    def test_public_holiday_notice(self) -> None:
        result = MessageExtractor().extract(make_message("Tomorrow is a public holiday."))
        assert result.items == ()

    def test_sometime_next_week(self) -> None:
        result = MessageExtractor().extract(make_message("Let us meet sometime next week."))
        assert result.items == ()


class TestMissingTime:
    """Explicit date without an explicit time leaves time as null."""

    def test_reminder_without_time(self) -> None:
        result = MessageExtractor().extract(
            make_message("Reminder: mentor catch-up happens on 2026-09-16 in the city clinic.")
        )
        assert len(result.items) == 1
        item = result.items[0]
        assert item.type is ItemType.REMINDER
        assert item.date == date(2026, 9, 16)
        assert item.time is None

    def test_scheduled_meeting_without_time(self) -> None:
        result = MessageExtractor().extract(
            make_message("The product demo is scheduled for 2026-09-17 in Zoom.")
        )
        assert len(result.items) == 1
        assert result.items[0].type is ItemType.MEETING
        assert result.items[0].time is None

    def test_calendar_event_without_time(self) -> None:
        result = MessageExtractor().extract(
            make_message("Calendar update: family dinner, 2026-09-19, the library.")
        )
        assert len(result.items) == 1
        assert result.items[0].type is ItemType.EVENT
        assert result.items[0].time is None


class TestMultipleTasks:
    """A message with several distinct deadlines yields one task per deadline."""

    def test_two_explicit_deadlines(self) -> None:
        result = MessageExtractor().extract(
            make_message("Submit the report by 2026-09-04 and pay the bill by 2026-09-09.")
        )
        assert len(result.items) == 2
        first, second = result.items
        assert all(item.type is ItemType.TASK for item in result.items)
        assert first.title == "submit the report"
        assert first.deadline == date(2026, 9, 4)
        assert first.item_id == "TASK_T001"
        assert second.title == "pay the bill"
        assert second.deadline == date(2026, 9, 9)
        assert second.item_id == "TASK_T001-2"

    def test_three_deadlines_split_by_commas(self) -> None:
        result = MessageExtractor().extract(
            make_message(
                "Reply to the client by 2026-09-05, renew the library book by 2026-09-07, "
                "and email the signed document by 2026-09-10."
            )
        )
        assert len(result.items) == 3
        assert [item.deadline for item in result.items] == [
            date(2026, 9, 5),
            date(2026, 9, 7),
            date(2026, 9, 10),
        ]

    def test_relative_and_explicit_mix(self) -> None:
        result = MessageExtractor().extract(
            make_message("Send the draft by tomorrow and finish the slides by Friday.")
        )
        assert len(result.items) == 2
        assert result.items[0].deadline == date(2026, 9, 2)
        assert result.items[1].deadline == date(2026, 9, 4)


class TestLLMFallback:
    def test_llm_consulted_when_rules_find_nothing(self) -> None:
        llm = StubLLM(
            payload=items_payload(one_item(title="send the proposal", type="task"))
        )
        extractor = MessageExtractor(llm=llm)
        result = extractor.extract(make_message("Please send the proposal."))
        assert llm.calls
        assert result.method is ExtractorMethod.LLM_FALLBACK
        assert result.items[0].title == "send the proposal"

    def test_llm_not_consulted_when_rule_matches(self) -> None:
        llm = StubLLM(payload=items_payload())
        extractor = MessageExtractor(llm=llm)
        result = extractor.extract(
            make_message("Please submit the report by 2026-09-04.")
        )
        assert llm.calls == []
        assert result.method is ExtractorMethod.RULE_BASED

    def test_llm_failure_returns_empty_result(self) -> None:
        llm = StubLLM(payload=items_payload(), fail=True)
        extractor = MessageExtractor(llm=llm)
        result = extractor.extract(make_message("Please send the proposal."))
        assert llm.calls
        assert result.method is ExtractorMethod.NONE
        assert result.items == ()

    def test_llm_invalid_response_returns_empty_result(self) -> None:
        llm = StubLLM(payload='{"items": [{"type": "bogus"}]}')
        extractor = MessageExtractor(llm=llm)
        result = extractor.extract(make_message("Please send the proposal."))
        assert result.method is ExtractorMethod.NONE
        assert result.items == ()

    def test_llm_exception_is_contained(self) -> None:
        class BoomLLM(MessageExtractorLLM):
            def extract(
                self, *, message_id: str, safe_message: str
            ) -> tuple[ExtractedItem, ...] | None:
                raise RuntimeError("provider down")

        extractor = MessageExtractor(llm=BoomLLM())
        result = extractor.extract(make_message("Please send the proposal."))
        assert result.method is ExtractorMethod.NONE
        assert result.items == ()


class TestLLMSecurity:
    def test_llm_never_receives_raw_sensitive_value(self) -> None:
        llm = StubLLM(payload=items_payload())
        extractor = MessageExtractor(llm=llm)
        extractor.extract(make_message("Your OTP is 482913."))
        assert llm.calls
        _, safe_message = llm.calls[0]
        assert "482913" not in safe_message
        assert "******" in safe_message

    def test_prompt_instructs_about_masking_and_emptiness(self) -> None:
        llm = RecordingLLM()
        prompt = llm.build_prompt(message_id="T001", safe_message="Your OTP is ******.")
        assert "do not invent" in prompt.lower()
        assert "masked" in prompt.lower()
        assert "482913" not in prompt
        assert "T001" in prompt

    def test_no_item_contains_raw_sensitive_value(self) -> None:
        extractor = MessageExtractor()
        result = extractor.extract(make_message("Your OTP is 482913."))
        assert result.items == ()


class TestLLMResponseParsing:
    def test_valid_single_object(self) -> None:
        items = parse_llm_response("T001", items_payload(one_item()))
        assert len(items) == 1
        assert items[0].title == "submit the report"
        assert items[0].type is ItemType.TASK
        assert items[0].deadline == date(2026, 9, 4)
        assert items[0].item_id == "TASK_T001"

    def test_valid_bare_list(self) -> None:
        items = parse_llm_response("T001", json.dumps([one_item()]))
        assert len(items) == 1
        assert items[0].title == "submit the report"

    def test_empty_items_list(self) -> None:
        assert parse_llm_response("T001", items_payload()) == ()

    def test_fenced_json(self) -> None:
        raw = "```json\n" + items_payload(one_item()) + "\n```"
        items = parse_llm_response("T001", raw)
        assert len(items) == 1

    def test_duplicate_type_ids_are_indexed(self) -> None:
        items = parse_llm_response(
            "T001",
            items_payload(one_item(), one_item(title="pay the bill")),
        )
        assert len(items) == 2
        assert [item.item_id for item in items] == ["TASK_T001", "TASK_T001-2"]

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", items_payload(one_item(type="urgent")))

    def test_empty_title_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", items_payload(one_item(title="  ")))

    def test_invalid_date_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", items_payload(one_item(date="soon")))

    def test_invalid_time_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", items_payload(one_item(time="25:00")))

    def test_message_id_mismatch_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", items_payload(message_id="OTHER"))

    def test_missing_items_field_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", json.dumps({"message_id": "T001"}))

    def test_empty_response_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", "")

    def test_non_json_rejected(self) -> None:
        with pytest.raises(LLMResponseError):
            parse_llm_response("T001", "not json at all")

    def test_bad_priority_coerces_to_unknown(self) -> None:
        items = parse_llm_response("T001", items_payload(one_item(priority="very")))
        assert items[0].priority is Priority.UNKNOWN


class TestRulesOnRealDataset:
    def test_all_messages_extract_without_crash(self) -> None:
        from app.config import Settings
        from app.services.loader import load_messages_csv

        dataset = load_messages_csv(Settings().messages_csv_path)
        extractor = MessageExtractor()
        total_items = 0
        messages_with_items = 0
        for message in dataset.messages:
            result = extractor.extract(message)
            assert result.message_id == message.message_id
            assert result.items or result.method is ExtractorMethod.NONE
            for item in result.items:
                assert item.source_message_id == message.message_id
                assert item.title
                if item.type is ItemType.TASK:
                    assert item.date is None
                else:
                    assert item.deadline is None
            total_items += len(result.items)
            messages_with_items += bool(result.items)
        assert total_items > 0
        assert messages_with_items > 0

    def test_item_ids_unique_across_dataset(self) -> None:
        from app.config import Settings
        from app.services.loader import load_messages_csv

        dataset = load_messages_csv(Settings().messages_csv_path)
        extractor = MessageExtractor()
        item_ids = [
            item.item_id
            for message in dataset.messages
            for item in extractor.extract(message).items
        ]
        assert len(item_ids) == len(set(item_ids))

    def test_raw_values_never_reach_llm_on_dataset(self) -> None:
        from app.config import Settings
        from app.services.loader import load_messages_csv
        from app.services.sensitive_detector import SensitiveDetector

        dataset = load_messages_csv(Settings().messages_csv_path)
        detector = SensitiveDetector()
        llm = StubLLM(
            payload=lambda mid: items_payload(
                one_item(title="fallback item"), message_id=mid
            )
        )
        extractor = MessageExtractor(llm=llm)
        for message in dataset.messages:
            extractor.extract(message)
        assert llm.calls
        for (message_id, safe_message), message in zip(llm.calls, dataset.messages):
            for detection in detector.detect(message.message):
                assert detection.matched_value_internal_only not in safe_message, (
                    f"{message_id} leaked into LLM input"
                )
