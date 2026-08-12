"""Repository for the generated pipeline artifacts.

The API reads validated results from the ``outputs/`` JSON artifacts written by
``scripts.run_pipeline``. This module is the only place that touches the
artifact files: route handlers never open files or parse JSON themselves.

Security: the repository exposes only the sanitized Pydantic models
(:class:`FinalMessageResult`, :class:`MessageSensitiveResult`, ...) that the
pipeline already vetted. Raw message text never appears in any artifact.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.models.pipeline import FinalMessageResult, MessageSensitiveResult
from app.models.task_event import ExtractionResult

VALIDATION_REPORT_FILENAME = "validation_report.json"
FINAL_RESULTS_FILENAME = "final_results.json"
SENSITIVE_FILENAME = "sensitive_detections.json"
TASKS_EVENTS_FILENAME = "tasks_events.json"


class OutputRepositoryError(Exception):
    """Raised when a required pipeline artifact is missing or corrupt."""


class OutputRepository:
    """Reads and validates the pipeline artifacts from the outputs directory."""

    def __init__(
        self,
        *,
        outputs_dir: str | Path,
        mandatory_demo_ids_path: str | Path | None = None,
    ) -> None:
        self._outputs_dir = Path(outputs_dir)
        self._mandatory_ids_path = (
            Path(mandatory_demo_ids_path) if mandatory_demo_ids_path else None
        )

    # ------------------------------------------------------------------ reads

    def load_final_results(self) -> tuple[FinalMessageResult, ...]:
        """Load and validate every per-message final result (dataset order)."""
        return tuple(
            FinalMessageResult.model_validate(record)
            for record in self._records(FINAL_RESULTS_FILENAME)
        )

    def load_sensitive_results(self) -> tuple[MessageSensitiveResult, ...]:
        """Load the sanitized sensitive-detection results (dataset order)."""
        return tuple(
            MessageSensitiveResult.model_validate(record)
            for record in self._records(SENSITIVE_FILENAME)
        )

    def load_extractions(self) -> tuple[ExtractionResult, ...]:
        """Load the per-message extraction results (dataset order)."""
        return tuple(
            ExtractionResult.model_validate(record)
            for record in self._records(TASKS_EVENTS_FILENAME)
        )

    def load_validation_document(self) -> dict[str, object]:
        """Load the raw ``validation_report.json`` document."""
        path = self._path(VALIDATION_REPORT_FILENAME)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OutputRepositoryError(
                f"Could not read validation report {path.name}: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise OutputRepositoryError("validation report is not a JSON object")
        return document

    def load_mandatory_ids(self) -> tuple[str, ...]:
        """Load the mandatory demo IDs from their CSV file."""
        if self._mandatory_ids_path is None:
            raise OutputRepositoryError("mandatory demo IDs path is not configured")
        path = self._mandatory_ids_path
        if not path.is_file():
            raise OutputRepositoryError(f"Mandatory IDs file does not exist: {path}")
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as exc:  # noqa: BLE001 - report a stable error, never the raw traceback
            raise OutputRepositoryError(
                f"Could not parse mandatory IDs file {path.name}: {exc}"
            ) from exc
        if "message_id" not in df.columns:
            raise OutputRepositoryError(
                f"Mandatory IDs file {path.name} must have a 'message_id' column."
            )
        return tuple(
            str(value).strip()
            for value in df["message_id"].tolist()
            if value is not None and str(value).strip() != ""
        )

    def has_outputs(self) -> bool:
        """True when the primary artifacts are present."""
        return (
            self._path(FINAL_RESULTS_FILENAME).is_file()
            and self._path(VALIDATION_REPORT_FILENAME).is_file()
        )

    def clear_cache(self) -> None:
        """Drop cached artifacts (used by tests after regenerating outputs)."""
        self._records.cache_clear()

    # ---------------------------------------------------------------- helpers

    def _path(self, filename: str) -> Path:
        return self._outputs_dir / filename

    @lru_cache(maxsize=1)
    def _records(self, filename: str) -> tuple[dict[str, object], ...]:
        path = self._path(filename)
        if not path.is_file():
            raise OutputRepositoryError(
                f"Pipeline artifact not found: {path}. Run "
                "`python -m scripts.run_pipeline` first."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OutputRepositoryError(f"Artifact {path.name} is not valid JSON: {exc}") from exc
        records = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise OutputRepositoryError(f"Artifact {path.name} has no 'messages' list")
        return tuple(record for record in records if isinstance(record, dict))
