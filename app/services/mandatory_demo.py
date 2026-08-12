"""Mandatory-demo coverage.

The demo must show a fixed, separately maintained set of message IDs (provided
through ``mandatory_demo_ids.csv``, never hardcoded in Python). This module:

- loads the mandatory IDs from CSV (validating count and uniqueness),
- checks them against the dataset and the pipeline outputs,
- returns the complete processed results for exactly those messages.

The mandatory messages are returned in the chronological order of the original
dataset. No fake results are ever produced - if a mandatory message is missing
from the processed outputs the service raises instead of inventing a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from app.models.pipeline import FinalMessageResult


class MandatoryDemoError(Exception):
    """Raised when the mandatory-ID file or its validation fails."""


class MandatoryDemoCheck(BaseModel):
    """Dataset-level validation of the mandatory demo IDs.

    Attributes:
        provided_count: Number of IDs loaded from the file.
        unique_count: Number of distinct IDs.
        duplicate_ids: IDs that appear more than once in the file.
        missing_from_dataset: Mandatory IDs absent from the 900-message dataset.
        not_processed: Mandatory IDs with no pipeline output record.
        not_classified: Mandatory IDs whose output record has no classification.
        ok: True when every check passes.
    """

    model_config = ConfigDict(frozen=True)

    provided_count: int
    unique_count: int
    duplicate_ids: tuple[str, ...] = ()
    missing_from_dataset: tuple[str, ...] = ()
    not_processed: tuple[str, ...] = ()
    not_classified: tuple[str, ...] = ()
    ok: bool


class MandatoryDemoResult(BaseModel):
    """Complete processed results for the mandatory demo messages.

    Attributes:
        requested_ids: The mandatory IDs (in file order).
        results: Processed results in the original dataset chronological order.
        found: Number of mandatory IDs present in the dataset.
        processed: Number of mandatory IDs present in the processed results.
        missing: Mandatory IDs not found in the processed results.
    """

    model_config = ConfigDict(frozen=True)

    requested_ids: tuple[str, ...]
    results: tuple[FinalMessageResult, ...]
    found: int
    processed: int
    missing: tuple[str, ...] = ()


def load_mandatory_ids(
    path: str | Path, *, expected_count: int = 15
) -> tuple[str, ...]:
    """Load the mandatory message IDs from a CSV with a ``message_id`` column.

    Raises:
        MandatoryDemoError: if the file is missing, lacks the ``message_id``
            column, does not provide exactly ``expected_count`` IDs, or contains
            duplicate IDs.
    """
    source = Path(path)
    if not source.is_file():
        raise MandatoryDemoError(f"Mandatory IDs file does not exist: {source}")
    try:
        df = pd.read_csv(source, dtype=str)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MandatoryDemoError(f"Could not parse mandatory IDs file {source}: {exc}") from exc

    if "message_id" not in df.columns:
        raise MandatoryDemoError(
            f"Mandatory IDs file {source} must have a 'message_id' column; "
            f"found {list(df.columns)}."
        )

    ids = tuple(
        str(value).strip()
        for value in df["message_id"].tolist()
        if value is not None and str(value).strip() != ""
    )
    if len(ids) != expected_count:
        raise MandatoryDemoError(
            f"Mandatory IDs file must contain exactly {expected_count} IDs, found {len(ids)}."
        )
    duplicates = tuple(sorted({value for value in ids if ids.count(value) > 1}))
    if duplicates:
        raise MandatoryDemoError(
            f"Mandatory IDs must be unique; duplicate(s): {list(duplicates)}."
        )
    return ids


class MandatoryDemoService:
    """Validates and serves the mandatory demo messages.

    Args:
        ids: The mandatory message IDs (loaded from file by the caller).
    """

    def __init__(self, ids: Sequence[str]) -> None:
        self._ids = tuple(ids)

    @property
    def ids(self) -> tuple[str, ...]:
        """The mandatory message IDs in file order."""
        return self._ids

    def check(
        self,
        *,
        dataset_ids: Sequence[str],
        processed_ids: Sequence[str],
        classified_ids: Sequence[str],
    ) -> MandatoryDemoCheck:
        """Validate the mandatory IDs against the dataset and pipeline outputs."""
        expected = set(self._ids)
        dataset = set(dataset_ids)
        processed = set(processed_ids)
        classified = set(classified_ids)

        missing_from_dataset = tuple(sorted(expected - dataset))
        not_processed = tuple(sorted(expected - processed))
        not_classified = tuple(
            sorted(
                message_id
                for message_id in expected
                if message_id in processed and message_id not in classified
            )
        )
        duplicate_ids = tuple(sorted({value for value in self._ids if self._ids.count(value) > 1}))
        ok = (
            not missing_from_dataset
            and not not_processed
            and not not_classified
            and not duplicate_ids
        )
        return MandatoryDemoCheck(
            provided_count=len(self._ids),
            unique_count=len(set(self._ids)),
            duplicate_ids=duplicate_ids,
            missing_from_dataset=missing_from_dataset,
            not_processed=not_processed,
            not_classified=not_classified,
            ok=ok,
        )

    def build(self, final_results: Sequence[FinalMessageResult]) -> MandatoryDemoResult:
        """Return the complete processed results for the mandatory messages.

        Results are filtered in the order of ``final_results``, which preserves
        the original dataset chronological order.

        Raises:
            MandatoryDemoError: if any mandatory message is missing from the
                processed results - fake results are never generated.
        """
        by_id = {result.message_id: result for result in final_results}
        missing = tuple(sorted(message_id for message_id in self._ids if message_id not in by_id))
        if missing:
            raise MandatoryDemoError(
                "Mandatory message ID(s) missing from processed results "
                f"(no fake results are generated): {list(missing)}."
            )
        wanted = set(self._ids)
        results = tuple(result for result in final_results if result.message_id in wanted)
        return MandatoryDemoResult(
            requested_ids=self._ids,
            results=results,
            found=len(wanted),
            processed=len(results),
            missing=(),
        )
