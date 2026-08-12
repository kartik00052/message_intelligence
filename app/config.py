"""Application configuration with configurable paths.

Paths default to the repository layout and can be overridden through
environment variables so that nothing is hardcoded to a specific machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MESSAGES_CSV_PATH = PROJECT_ROOT / "messages.csv"
DEFAULT_MANDATORY_DEMO_IDS_PATH = PROJECT_ROOT / "mandatory_demo_ids.csv"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_EXPECTED_MESSAGE_COUNT = 900
DEFAULT_LLM_CONFIDENCE_THRESHOLD = 0.75


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the pipeline.

    Attributes:
        messages_csv_path: Path to the input CSV dataset.
        mandatory_demo_ids_path: Path to the CSV listing message IDs that must
            appear in the demo.
        outputs_dir: Directory where generated JSON artifacts are written.
        expected_message_count: Number of messages the input dataset must
            contain exactly.
        llm_confidence_threshold: Rule-based confidence below which the LLM
            fallback classifier is consulted (must be in ``[0, 1]``).
    """

    messages_csv_path: Path = DEFAULT_MESSAGES_CSV_PATH
    mandatory_demo_ids_path: Path = DEFAULT_MANDATORY_DEMO_IDS_PATH
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR
    expected_message_count: int = DEFAULT_EXPECTED_MESSAGE_COUNT
    llm_confidence_threshold: float = DEFAULT_LLM_CONFIDENCE_THRESHOLD

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        return cls(
            messages_csv_path=_path_from_env("MESSAGES_CSV_PATH", DEFAULT_MESSAGES_CSV_PATH),
            mandatory_demo_ids_path=_path_from_env(
                "MANDATORY_DEMO_IDS_PATH", DEFAULT_MANDATORY_DEMO_IDS_PATH
            ),
            outputs_dir=_path_from_env("OUTPUTS_DIR", DEFAULT_OUTPUTS_DIR),
            expected_message_count=_int_from_env(
                "EXPECTED_MESSAGE_COUNT", DEFAULT_EXPECTED_MESSAGE_COUNT
            ),
            llm_confidence_threshold=_float_from_env(
                "LLM_CONFIDENCE_THRESHOLD", DEFAULT_LLM_CONFIDENCE_THRESHOLD
            ),
        )


def _path_from_env(name: str, default: Path) -> Path:
    """Read an environment variable as a path, or return ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return Path(raw)


def _int_from_env(name: str, default: int) -> int:
    """Read an environment variable as an integer, or return ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _float_from_env(name: str, default: float) -> float:
    """Read an environment variable as a float, or return ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Environment variable {name} must be between 0 and 1, got {value!r}")
    return value
