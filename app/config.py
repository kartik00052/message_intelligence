"""Application configuration with configurable paths.

Paths default to the repository layout and can be overridden through
environment variables so that nothing is hardcoded to a specific machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MESSAGES_CSV_PATH = PROJECT_ROOT / "messages.csv"
DEFAULT_MANDATORY_DEMO_IDS_PATH = PROJECT_ROOT / "mandatory_demo_ids.csv"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_ENCRYPTED_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_KEY_FILE = DEFAULT_ENCRYPTED_DATA_DIR / ".dataset.key"
DEFAULT_EXPECTED_MESSAGE_COUNT = 900
DEFAULT_EXPECTED_MANDATORY_COUNT = 15
DEFAULT_LLM_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_LLM_ENABLED = False
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = ""
DEFAULT_LLM_API_KEY = ""
DEFAULT_LLM_BASE_URL = ""
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0


def _secret(value: str) -> str:
    """Marker type so a secret is clearly not a public value."""
    return value


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
        expected_mandatory_count: Number of mandatory demo IDs that must be
            provided.
        llm_enabled: Whether the LLM fallback is available. When False the
            pipeline runs fully offline using deterministic rules only.
        llm_provider: Name of the LLM provider (default ``openai``).
        llm_model: Model identifier sent to the provider.
        llm_api_key: Secret API key. Never logged or serialized.
        llm_base_url: Optional provider base URL override.
        llm_timeout_seconds: Timeout for a single LLM request.
        llm_confidence_threshold: Rule-based confidence below which the LLM
            fallback classifier is consulted (must be in ``[0, 1]``).
    """

    messages_csv_path: Path = DEFAULT_MESSAGES_CSV_PATH
    mandatory_demo_ids_path: Path = DEFAULT_MANDATORY_DEMO_IDS_PATH
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR
    encrypted_data_dir: Path = DEFAULT_ENCRYPTED_DATA_DIR
    dataset_key_file: Path = DEFAULT_DATASET_KEY_FILE
    expected_message_count: int = DEFAULT_EXPECTED_MESSAGE_COUNT
    expected_mandatory_count: int = DEFAULT_EXPECTED_MANDATORY_COUNT
    llm_confidence_threshold: float = DEFAULT_LLM_CONFIDENCE_THRESHOLD
    llm_enabled: bool = DEFAULT_LLM_ENABLED
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    llm_api_key: str = field(default_factory=lambda: _secret(DEFAULT_LLM_API_KEY))
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        return cls(
            messages_csv_path=_path_from_env("MESSAGES_CSV_PATH", DEFAULT_MESSAGES_CSV_PATH),
            mandatory_demo_ids_path=_path_from_env(
                "MANDATORY_DEMO_IDS_PATH", DEFAULT_MANDATORY_DEMO_IDS_PATH
            ),
            outputs_dir=_path_from_env("OUTPUTS_DIR", DEFAULT_OUTPUTS_DIR),
            encrypted_data_dir=_path_from_env(
                "ENCRYPTED_DATA_DIR", DEFAULT_ENCRYPTED_DATA_DIR
            ),
            dataset_key_file=_path_from_env("DATASET_KEY_FILE", DEFAULT_DATASET_KEY_FILE),
            expected_message_count=_int_from_env(
                "EXPECTED_MESSAGE_COUNT", DEFAULT_EXPECTED_MESSAGE_COUNT
            ),
            expected_mandatory_count=_int_from_env(
                "EXPECTED_MANDATORY_COUNT", DEFAULT_EXPECTED_MANDATORY_COUNT
            ),
            llm_confidence_threshold=_float_from_env(
                "LLM_CONFIDENCE_THRESHOLD", DEFAULT_LLM_CONFIDENCE_THRESHOLD
            ),
            llm_enabled=_bool_from_env("LLM_ENABLED", DEFAULT_LLM_ENABLED),
            llm_provider=_str_from_env("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
            llm_model=_str_from_env("LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_api_key=_str_from_env("LLM_API_KEY", DEFAULT_LLM_API_KEY),
            llm_base_url=_str_from_env("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            llm_timeout_seconds=_float_from_env(
                "LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS
            ),
        )

    @property
    def llm_configured(self) -> bool:
        """True when the LLM fallback is enabled and a provider is available."""
        return self.llm_enabled and bool(self.llm_api_key) and bool(self.llm_model)


def _bool_from_env(name: str, default: bool) -> bool:
    """Read an environment variable as a boolean."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _str_from_env(name: str, default: str) -> str:
    """Read an environment variable as a string."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip()


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
