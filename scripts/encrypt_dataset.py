"""Encrypt the demo dataset CSVs into committed Fernet blobs.

The plaintext ``messages.csv`` and ``mandatory_demo_ids.csv`` are never
committed (see ``README.md`` §2 and §5). This script turns them into
``data/messages.csv.enc`` and ``data/mandatory_demo_ids.csv.enc`` using a
Fernet key from:

1. the ``DATASET_ENC_KEY`` environment variable, or
2. the gitignored ``data/.dataset.key`` file, or
3. (with ``--new-key``, or when none exists) a freshly generated key that is
   persisted to ``data/.dataset.key``.

Afterwards the same key must be set as a Render environment variable
(``DATASET_ENC_KEY``) so the build can decrypt the blobs back to plaintext.

Usage:
    python -m scripts.encrypt_dataset          # key from env or data/.dataset.key
    python -m scripts.encrypt_dataset --new-key  # always generate a fresh key
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from app.config import Settings
from app.services.dataset_cipher import (
    DATASET_KEY_ENV,
    MANDATORY_ENC_FILENAME,
    MESSAGES_ENC_FILENAME,
    decrypt_file,
    encrypt_file,
    generate_key,
    load_key,
)


def _persist_key(secret: str, key_file: Path) -> None:
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text(secret + "\n", encoding="utf-8")
    try:
        key_file.chmod(0o600)
    except OSError:
        pass  # best effort on platforms without POSIX permissions


def encrypt_datasets(
    *,
    settings: Settings,
    new_key: bool = False,
    key: str | None = None,
) -> str:
    """Encrypt the configured CSVs into ``settings.encrypted_data_dir``.

    Returns the Fernet key used (for display / copying into Render env).
    """
    sources = {
        settings.messages_csv_path: MESSAGES_ENC_FILENAME,
        settings.mandatory_demo_ids_path: MANDATORY_ENC_FILENAME,
    }
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit(f"Plaintext dataset missing; cannot encrypt: {', '.join(missing)}")

    if key is not None:
        fernet = load_key(key)
    elif os.environ.get(DATASET_KEY_ENV):
        fernet = load_key(os.environ[DATASET_KEY_ENV])
        key = os.environ[DATASET_KEY_ENV]
    elif new_key or not settings.dataset_key_file.is_file():
        key = generate_key()
        fernet = load_key(key)
        _persist_key(key, settings.dataset_key_file)
    else:
        key = settings.dataset_key_file.read_text(encoding="utf-8").strip()
        fernet = load_key(key)

    settings.encrypted_data_dir.mkdir(parents=True, exist_ok=True)
    for source, enc_name in sources.items():
        destination = settings.encrypted_data_dir / enc_name
        encrypt_file(fernet, source, destination)
        print(f"Encrypted {source.name} -> {destination}")

    _verify(settings, fernet)
    return key


def _verify(settings: Settings, fernet) -> None:
    """Decrypt each blob to a temp file and require byte-identical round-trip."""
    pairs = (
        (settings.messages_csv_path, MESSAGES_ENC_FILENAME),
        (settings.mandatory_demo_ids_path, MANDATORY_ENC_FILENAME),
    )
    with tempfile.TemporaryDirectory() as tmp:
        for plain, enc_name in pairs:
            restored = Path(tmp) / plain.name
            decrypt_file(fernet, settings.encrypted_data_dir / enc_name, restored)
            if restored.read_bytes() != plain.read_bytes():
                raise SystemExit(f"Round-trip verification failed for {enc_name}")
    print("Round-trip verification OK (blobs decrypt back to the original CSVs).")


def main(argv: list[str] | None = None) -> int:
    new_key = "--new-key" in (argv or sys.argv[1:])
    settings = Settings.from_env()
    key = encrypt_datasets(settings=settings, new_key=new_key)
    print(f"\nKey source: {DATASET_KEY_ENV} environment variable or data/.dataset.key")
    print(f"\nSet this as the Render secret environment variable {DATASET_KEY_ENV}:")
    print(key)
    print("\nKeep the key private. Anyone with the key can decrypt the dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
