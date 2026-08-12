"""Dataset ingestion: load the CSV and produce validated typed messages.

The loader is intentionally independent of FastAPI and any later pipeline
stage. It only reads, validates and models the input data.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.models.dataset import DatasetStatistics
from app.models.message import RawMessage
from app.services.validator import validate_dataset

DEFAULT_ENCODING = "utf-8-sig"
DEFAULT_EXPECTED_COUNT = 900


class DatasetLoadingError(Exception):
    """Raised when the dataset file cannot be read or parsed."""


@dataclass(frozen=True)
class LoadedDataset:
    """Result of loading and validating the input dataset.

    Attributes:
        messages: Validated messages in original file (chronological) order.
        statistics: Summary statistics for the loaded dataset.
        source_path: Path the dataset was loaded from.
    """

    messages: tuple[RawMessage, ...]
    statistics: DatasetStatistics
    source_path: Path


def load_messages_csv(
    path: str | os.PathLike[str],
    *,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
    encoding: str = DEFAULT_ENCODING,
) -> LoadedDataset:
    """Load and validate a message dataset from a CSV file.

    Args:
        path: Path to the CSV file. The file itself is never modified.
        expected_count: Exact number of messages the dataset must contain.
        encoding: Text encoding used to read the file. Defaults to UTF-8 with a
            BOM, which is how ``messages.csv`` is stored.

    Returns:
        A :class:`LoadedDataset` with messages in their original order.

    Raises:
        DatasetLoadingError: If the file is missing, unreadable or cannot be
            parsed as CSV.
        DatasetValidationError: If the loaded rows fail input validation.
    """
    source_path = Path(path)
    if not source_path.exists():
        raise DatasetLoadingError(f"Dataset file does not exist: {source_path}")
    if not source_path.is_file():
        raise DatasetLoadingError(f"Dataset path is not a file: {source_path}")

    try:
        df = pd.read_csv(source_path, encoding=encoding, dtype=str)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetLoadingError(f"Could not parse CSV file {source_path}: {exc}") from exc
    except OSError as exc:
        raise DatasetLoadingError(f"Could not read CSV file {source_path}: {exc}") from exc

    validated = validate_dataset(df, expected_count=expected_count)
    messages = _to_messages(validated)
    statistics = dataset_statistics(messages)
    return LoadedDataset(messages=messages, statistics=statistics, source_path=source_path)


def dataset_statistics(messages: Sequence[RawMessage]) -> DatasetStatistics:
    """Compute summary statistics over validated messages.

    Args:
        messages: The validated messages, in chronological order.

    Returns:
        A :class:`DatasetStatistics` summary.

    Raises:
        ValueError: If ``messages`` is empty.
    """
    if not messages:
        raise ValueError("Cannot compute statistics for an empty dataset.")

    return DatasetStatistics(
        total_messages=len(messages),
        unique_message_ids=len({message.message_id for message in messages}),
        earliest_timestamp=min(message.timestamp for message in messages),
        latest_timestamp=max(message.timestamp for message in messages),
        empty_message_count=sum(1 for message in messages if not message.message.strip()),
        empty_sender_count=sum(1 for message in messages if not message.sender.strip()),
    )


def _to_messages(df: pd.DataFrame) -> tuple[RawMessage, ...]:
    """Convert a validated DataFrame into typed ``RawMessage`` objects."""
    messages: list[RawMessage] = []
    for row in df.to_dict(orient="records"):
        messages.append(
            RawMessage(
                message_id=str(row["message_id"]).strip(),
                timestamp=row["timestamp"],
                sender=str(row["sender"]).strip(),
                message=str(row["message"]).strip(),
            )
        )
    return tuple(messages)
