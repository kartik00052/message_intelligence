"""Unit tests for dataset ingestion and input validation."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings
from app.models.dataset import DatasetStatistics
from app.models.message import RawMessage
from app.services.loader import DatasetLoadingError, dataset_statistics, load_messages_csv
from app.services.validator import DatasetValidationError

COLUMNS = ["message_id", "timestamp", "sender", "message"]

VALID_SENDERS = ["Meera", "Ishaan", "Kabir", "Aarav", "Ananya", "Neha"]


def _valid_rows(n: int = 900) -> list[dict[str, str]]:
    """Generate ``n`` valid, chronologically ordered message rows."""
    base = datetime(2026, 9, 1, 8, 0, 0)
    rows: list[dict[str, str]] = []
    for i in range(n):
        timestamp = (base + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            {
                "message_id": f"MSG_{i + 1:04d}",
                "timestamp": timestamp,
                "sender": VALID_SENDERS[i % len(VALID_SENDERS)],
                "message": f"Message number {i + 1}",
            }
        )
    return rows


def _write_csv(
    path: Path, rows: list[dict[str, str]], *, fieldnames: list[str] | None = None
) -> None:
    """Write rows to ``path`` as a UTF-8 (BOM) CSV, like the real dataset."""
    fields = fieldnames or list(rows[0].keys())
    rows = [{key: value for key, value in row.items() if key in fields} for row in rows]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class TestLoader:
    def test_loads_real_dataset(self) -> None:
        ds = load_messages_csv(Settings().messages_csv_path)
        assert ds.statistics.total_messages == 900
        assert ds.statistics.unique_message_ids == 900
        assert ds.statistics.empty_message_count == 0
        assert ds.statistics.empty_sender_count == 0
        assert ds.statistics.earliest_timestamp < ds.statistics.latest_timestamp

        message_ids = [message.message_id for message in ds.messages]
        assert message_ids[0] == "MSG_0001"
        assert len(message_ids) == 900
        assert len(set(message_ids)) == 900

        timestamps = [message.timestamp for message in ds.messages]
        assert timestamps == sorted(timestamps)
        assert all(isinstance(ts, datetime) for ts in timestamps)
        assert all(message.sender and message.message for message in ds.messages)

    def test_preserves_file_order_and_ids(self, tmp_path: Path) -> None:
        rows = _valid_rows(10)
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)

        ds = load_messages_csv(path, expected_count=10)
        assert [m.message_id for m in ds.messages] == [row["message_id"] for row in rows]
        assert [m.message for m in ds.messages] == [row["message"] for row in rows]

    def test_handles_utf8_bom_and_special_characters(self, tmp_path: Path) -> None:
        rows = _valid_rows(3)
        rows[0]["message"] = "Just checking\u2014please join the call."
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)

        ds = load_messages_csv(path, expected_count=3)
        assert ds.messages[0].message == "Just checking\u2014please join the call."

    def test_missing_file_raises_loading_error(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetLoadingError, match="does not exist"):
            load_messages_csv(tmp_path / "missing.csv")

    def test_directory_raises_loading_error(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetLoadingError, match="not a file"):
            load_messages_csv(tmp_path)

    def test_empty_file_raises_loading_error(self, tmp_path: Path) -> None:
        path = tmp_path / "messages.csv"
        path.write_text("", encoding="utf-8-sig")
        with pytest.raises(DatasetLoadingError):
            load_messages_csv(path)

    def test_malformed_csv_raises_loading_error(self, tmp_path: Path) -> None:
        path = tmp_path / "messages.csv"
        path.write_text('message_id,timestamp,sender,message\n"unclosed', encoding="utf-8-sig")
        with pytest.raises(DatasetLoadingError):
            load_messages_csv(path)

    def test_configurable_expected_count(self, tmp_path: Path) -> None:
        path = tmp_path / "messages.csv"
        _write_csv(path, _valid_rows(10))
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path, expected_count=12)
        assert excinfo.value.issues
        assert excinfo.value.issues[0].code == "unexpected_dataset_size"


class TestValidation:
    def test_valid_dataset_passes(self, tmp_path: Path) -> None:
        rows = _valid_rows(5)
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)
        ds = load_messages_csv(path, expected_count=5)
        assert ds.statistics.total_messages == 5

    def test_missing_column_rejected(self, tmp_path: Path) -> None:
        rows = _valid_rows(5)
        path = tmp_path / "messages.csv"
        _write_csv(path, rows, fieldnames=["message_id", "timestamp", "message"])
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path, expected_count=5)
        assert [i.code for i in excinfo.value.issues] == ["missing_columns"]
        assert "sender" in str(excinfo.value)

    def test_unexpected_size_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "messages.csv"
        _write_csv(path, _valid_rows(901))
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path)
        assert excinfo.value.issues[0].code == "unexpected_dataset_size"
        assert "900" in str(excinfo.value)

        too_few = tmp_path / "too_few.csv"
        _write_csv(too_few, _valid_rows(899))
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(too_few)
        assert excinfo.value.issues[0].code == "unexpected_dataset_size"

    def test_duplicate_message_id_rejected(self, tmp_path: Path) -> None:
        rows = _valid_rows(5)
        rows[3]["message_id"] = rows[0]["message_id"]
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path, expected_count=5)
        assert "duplicate_message_id" in [i.code for i in excinfo.value.issues]
        assert "MSG_0001" in str(excinfo.value)

    def test_empty_message_id_rejected(self, tmp_path: Path) -> None:
        rows = _valid_rows(3)
        rows[1]["message_id"] = ""
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path, expected_count=3)
        assert "empty_message_id" in [i.code for i in excinfo.value.issues]

    def test_malformed_timestamp_rejected(self, tmp_path: Path) -> None:
        rows = _valid_rows(5)
        rows[2]["timestamp"] = "not-a-date"
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path, expected_count=5)
        assert "malformed_timestamp" in [i.code for i in excinfo.value.issues]
        assert "not-a-date" in str(excinfo.value)

    def test_missing_timestamp_rejected(self, tmp_path: Path) -> None:
        rows = _valid_rows(3)
        rows[1]["timestamp"] = ""
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path, expected_count=3)
        assert "malformed_timestamp" in [i.code for i in excinfo.value.issues]

    def test_empty_message_rejected(self, tmp_path: Path) -> None:
        rows = _valid_rows(5)
        rows[4]["message"] = ""
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path, expected_count=5)
        assert "missing_message_content" in [i.code for i in excinfo.value.issues]

    def test_whitespace_message_rejected(self, tmp_path: Path) -> None:
        rows = _valid_rows(3)
        rows[1]["message"] = "   "
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path, expected_count=3)
        assert "missing_message_content" in [i.code for i in excinfo.value.issues]

    def test_empty_sender_rejected(self, tmp_path: Path) -> None:
        rows = _valid_rows(5)
        rows[0]["sender"] = ""
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path, expected_count=5)
        assert "missing_sender" in [i.code for i in excinfo.value.issues]

    def test_out_of_order_rejected(self, tmp_path: Path) -> None:
        rows = _valid_rows(5)
        rows[4]["timestamp"] = rows[0]["timestamp"]
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path, expected_count=5)
        assert "not_chronological" in [i.code for i in excinfo.value.issues]

    def test_multiple_issues_reported_together(self, tmp_path: Path) -> None:
        rows = _valid_rows(5)
        rows[1]["message"] = ""
        rows[2]["message_id"] = rows[0]["message_id"]
        path = tmp_path / "messages.csv"
        _write_csv(path, rows)
        with pytest.raises(DatasetValidationError) as excinfo:
            load_messages_csv(path, expected_count=5)
        codes = [i.code for i in excinfo.value.issues]
        assert "missing_message_content" in codes
        assert "duplicate_message_id" in codes


class TestStatistics:
    def test_statistics_are_computed_correctly(self) -> None:
        messages = [
            RawMessage(
                message_id="MSG_0001",
                timestamp=datetime(2026, 9, 1, 8, 0, 0),
                sender="Meera",
                message="First",
            ),
            RawMessage(
                message_id="MSG_0002",
                timestamp=datetime(2026, 9, 1, 9, 0, 0),
                sender="",
                message="",
            ),
            RawMessage(
                message_id="MSG_0003",
                timestamp=datetime(2026, 9, 1, 10, 0, 0),
                sender="Ishaan",
                message="   ",
            ),
        ]
        stats = dataset_statistics(messages)
        assert stats == DatasetStatistics(
            total_messages=3,
            unique_message_ids=3,
            earliest_timestamp=datetime(2026, 9, 1, 8, 0, 0),
            latest_timestamp=datetime(2026, 9, 1, 10, 0, 0),
            empty_message_count=2,
            empty_sender_count=1,
        )

    def test_statistics_on_real_dataset(self) -> None:
        ds = load_messages_csv(Settings().messages_csv_path)
        stats = ds.statistics
        assert stats.total_messages == 900
        assert stats.unique_message_ids == 900
        assert stats.earliest_timestamp == datetime(2026, 9, 1, 8, 0, 0)
        assert stats.empty_message_count == 0
        assert stats.empty_sender_count == 0

    def test_statistics_reject_empty_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            dataset_statistics([])


class TestSettings:
    def test_default_paths(self) -> None:
        settings = Settings()
        assert settings.messages_csv_path == PROJECT_ROOT / "messages.csv"
        assert settings.mandatory_demo_ids_path == PROJECT_ROOT / "mandatory_demo_ids.csv"
        assert settings.outputs_dir == PROJECT_ROOT / "outputs"
        assert settings.expected_message_count == 900

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        csv_path = tmp_path / "custom.csv"
        ids_path = tmp_path / "custom_ids.csv"
        outputs = tmp_path / "custom_outputs"
        monkeypatch.setenv("MESSAGES_CSV_PATH", str(csv_path))
        monkeypatch.setenv("MANDATORY_DEMO_IDS_PATH", str(ids_path))
        monkeypatch.setenv("OUTPUTS_DIR", str(outputs))
        monkeypatch.setenv("EXPECTED_MESSAGE_COUNT", "950")

        settings = Settings.from_env()
        assert settings.messages_csv_path == csv_path
        assert settings.mandatory_demo_ids_path == ids_path
        assert settings.outputs_dir == outputs
        assert settings.expected_message_count == 950

    def test_invalid_expected_count_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXPECTED_MESSAGE_COUNT", "not-a-number")
        with pytest.raises(ValueError, match="EXPECTED_MESSAGE_COUNT"):
            Settings.from_env()
