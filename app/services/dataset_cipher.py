"""Fernet-based encryption for the private demo dataset.

The assignment forbids publishing the (fictional but sensitive-looking)
dataset in a public repository, so the plaintext CSVs are not tracked in git.
Instead the repository commits Fernet blobs (``data/*.enc``) and the decryption
key is supplied at build/start time through the ``DATASET_ENC_KEY`` environment
variable, or - for local development - the gitignored ``data/.dataset.key``
file written by ``scripts.encrypt_dataset``.

Only this module touches the key. ``prepare_datasets`` materializes a plaintext
CSV exactly when it is missing, so a fresh clone can regenerate the dataset
before the pipeline runs without ever printing a raw message value.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from app.config import PROJECT_ROOT, Settings

DATASET_KEY_ENV = "DATASET_ENC_KEY"
KEY_FILE_NAME = ".dataset.key"

MESSAGES_ENC_FILENAME = "messages.csv.enc"
MANDATORY_ENC_FILENAME = "mandatory_demo_ids.csv.enc"

_DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class DatasetEncryptionError(Exception):
    """Raised when the dataset key or an encrypted blob is unusable."""


def generate_key() -> str:
    """Return a fresh URL-safe Fernet key as an ASCII string."""
    return Fernet.generate_key().decode("ascii")


def load_key(secret: str) -> Fernet:
    """Build a Fernet cipher from a ``DATASET_ENC_KEY`` value."""
    try:
        raw = base64.urlsafe_b64decode(secret.encode("ascii") + b"=" * ((4 - len(secret) % 4) % 4))
    except Exception as exc:  # noqa: BLE001 - surface a stable, actionable error
        raise DatasetEncryptionError(
            f"{DATASET_KEY_ENV} must be a base64-encoded Fernet key (32 bytes)."
        ) from exc
    if len(raw) != 32:
        raise DatasetEncryptionError(
            f"{DATASET_KEY_ENV} must decode to 32 bytes, got {len(raw)}. "
            "Generate one with `python -m scripts.encrypt_dataset`."
        )
    return Fernet(base64.urlsafe_b64encode(raw))


def resolve_key(key_file: Path | None = None) -> Fernet:
    """Resolve the dataset key: environment variable first, then key file."""
    secret = os.environ.get(DATASET_KEY_ENV)
    if secret:
        return load_key(secret)
    key_file = key_file or _DEFAULT_DATA_DIR / KEY_FILE_NAME
    if key_file.is_file():
        stored = key_file.read_text(encoding="utf-8").strip()
        if stored:
            return load_key(stored)
    raise DatasetEncryptionError(
        f"No {DATASET_KEY_ENV} environment variable and no key file at {key_file}. "
        "Run `python -m scripts.encrypt_dataset` to generate one."
    )


def encrypt_file(fernet: Fernet, source: Path, destination: Path) -> None:
    """Encrypt ``source`` into a Fernet token at ``destination``."""
    destination.write_bytes(fernet.encrypt(source.read_bytes()))


def decrypt_file(fernet: Fernet, source: Path, destination: Path) -> None:
    """Decrypt a Fernet token ``source`` into ``destination``."""
    try:
        payload = fernet.decrypt(source.read_bytes())
    except InvalidToken as exc:
        raise DatasetEncryptionError(
            f"Cannot decrypt {source.name}: wrong key or corrupt blob."
        ) from exc
    destination.write_bytes(payload)


def prepare_datasets(settings: Settings) -> None:
    """Materialize plaintext dataset CSVs from the committed encrypted blobs.

    No-op when the plaintext files already exist (normal local runs). When a
    plaintext file is missing it is decrypted from ``settings.encrypted_data_dir``
    using the key from ``DATASET_ENC_KEY`` or ``settings.dataset_key_file``.
    """
    jobs = (
        (settings.messages_csv_path, MESSAGES_ENC_FILENAME),
        (settings.mandatory_demo_ids_path, MANDATORY_ENC_FILENAME),
    )
    pending = [plain for plain, _ in jobs if not plain.is_file()]
    if not pending:
        return
    fernet = resolve_key(settings.dataset_key_file)
    for plain, enc_name in jobs:
        if plain.is_file():
            continue
        enc_path = settings.encrypted_data_dir / enc_name
        if not enc_path.is_file():
            raise DatasetEncryptionError(
                f"Encrypted dataset blob not found: {enc_path}. "
                "Run `python -m scripts.encrypt_dataset` to create it."
            )
        decrypt_file(fernet, enc_path, plain)
