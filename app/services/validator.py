"""Input validation for the message dataset.

Validates that the raw DataFrame read from the CSV has the required schema,
the expected size, well-formed and unique message IDs, parseable timestamps,
non-empty content, and preserves chronological ordering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("message_id", "timestamp", "sender", "message")


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation problem.

    Attributes:
        code: Machine readable category of the problem.
        detail: Human readable description, including the affected row when
            relevant.
    """

    code: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.detail}"


class DatasetValidationError(Exception):
    """Raised when the dataset fails validation.

    Attributes:
        issues: All problems detected during validation.
    """

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(str(issue) for issue in self.issues))


def validate_dataset(df: pd.DataFrame, *, expected_count: int) -> pd.DataFrame:
    """Validate a raw dataset DataFrame.

    Args:
        df: DataFrame as read from the CSV, in file order.
        expected_count: Exact number of messages the dataset must contain.

    Returns:
        A copy of ``df`` with the ``timestamp`` column converted to ``datetime``.

    Raises:
        DatasetValidationError: If any validation rule is violated. All detected
            issues are reported at once.
    """
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise DatasetValidationError(
            [
                ValidationIssue(
                    "missing_columns",
                    f"Missing required column(s) {missing_columns}; found {list(df.columns)}.",
                )
            ]
        )

    issues: list[ValidationIssue] = []
    _validate_size(df, expected_count, issues)
    _validate_message_ids(df, issues)
    parsed_timestamps = _validate_timestamps(df, issues)
    _validate_text_fields(df, issues)

    if issues:
        raise DatasetValidationError(issues)

    normalized = df.copy()
    normalized["timestamp"] = parsed_timestamps
    return normalized


def _validate_size(df: pd.DataFrame, expected_count: int, issues: list[ValidationIssue]) -> None:
    if len(df) != expected_count:
        issues.append(
            ValidationIssue(
                "unexpected_dataset_size",
                f"Expected exactly {expected_count} messages but found {len(df)}.",
            )
        )


def _validate_message_ids(df: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    message_ids = df["message_id"]
    empty_mask = _empty_mask(message_ids)
    for position in _positions(empty_mask):
        issues.append(
            ValidationIssue(
                "empty_message_id",
                f"Row {position}: message_id is empty.",
            )
        )
    if empty_mask.any():
        return

    duplicated = message_ids[message_ids.duplicated(keep=False)]
    if not duplicated.empty:
        duplicate_values = sorted(str(value) for value in duplicated.unique())
        issues.append(
            ValidationIssue(
                "duplicate_message_id",
                f"Duplicate message_id value(s): {duplicate_values}.",
            )
        )


def _validate_timestamps(df: pd.DataFrame, issues: list[ValidationIssue]) -> pd.Series:
    raw_timestamps = df["timestamp"]
    parsed = pd.to_datetime(raw_timestamps, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(raw_timestamps, errors="coerce")

    bad_mask = parsed.isna()
    for position in _positions(bad_mask):
        issues.append(
            ValidationIssue(
                "malformed_timestamp",
                f"Row {position} (message_id={df['message_id'].iloc[position]!r}): "
                f"cannot parse timestamp {raw_timestamps.iloc[position]!r}.",
            )
        )

    if not bad_mask.any() and not parsed.is_monotonic_increasing:
        first_bad = _first_inversion_position(parsed)
        issues.append(
            ValidationIssue(
                "not_chronological",
                "Timestamps are not in chronological order; "
                f"first inversion between rows {first_bad - 1} and {first_bad}.",
            )
        )
    return parsed


def _validate_text_fields(df: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    for column, code in (
        ("message", "missing_message_content"),
        ("sender", "missing_sender"),
    ):
        empty_mask = _empty_mask(df[column])
        for position in _positions(empty_mask):
            issues.append(
                ValidationIssue(
                    code,
                    f"Row {position} (message_id={df['message_id'].iloc[position]!r}): "
                    f"{column} is empty.",
                )
            )


def _empty_mask(series: pd.Series) -> pd.Series:
    """Mask marking rows whose value is NaN or empty after stripping."""
    return series.fillna("").astype(str).str.strip().eq("")


def _positions(mask: pd.Series) -> list[int]:
    """Row positions (0-based, index independent) where ``mask`` is True."""
    return [position for position, is_bad in enumerate(mask.tolist()) if is_bad]


def _first_inversion_position(values: pd.Series) -> int:
    """Position of the first timestamp smaller than the previous one."""
    for position in range(1, len(values)):
        if values.iloc[position] < values.iloc[position - 1]:
            return position
    raise AssertionError("No inversion found in a non-monotonic series")
