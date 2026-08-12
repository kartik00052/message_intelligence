"""Tests for the Fernet dataset cipher and plaintext preparation."""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from app.config import Settings
from app.services.dataset_cipher import (
    DATASET_KEY_ENV,
    MANDATORY_ENC_FILENAME,
    MESSAGES_ENC_FILENAME,
    DatasetEncryptionError,
    decrypt_file,
    encrypt_file,
    generate_key,
    load_key,
    prepare_datasets,
    resolve_key,
)


def make_settings(*, tmp_path: Path) -> Settings:
    return Settings(
        messages_csv_path=tmp_path / "messages.csv",
        mandatory_demo_ids_path=tmp_path / "mandatory_demo_ids.csv",
        outputs_dir=tmp_path / "outputs",
        encrypted_data_dir=tmp_path / "data",
        dataset_key_file=tmp_path / "data" / ".dataset.key",
    )


def test_generate_and_round_trip(tmp_path: Path) -> None:
    fernet = load_key(generate_key())
    source = tmp_path / "original.bin"
    source.write_bytes(b"secret bytes")
    blob = tmp_path / "blob.enc"
    encrypt_file(fernet, source, blob)
    assert blob.read_bytes() != source.read_bytes()
    restored = tmp_path / "restored.bin"
    decrypt_file(fernet, blob, restored)
    assert restored.read_bytes() == source.read_bytes()


def test_load_key_rejects_bad_secret() -> None:
    with pytest.raises(DatasetEncryptionError):
        load_key("not-a-key")
    with pytest.raises(DatasetEncryptionError):
        load_key("c2hvcnQ=")  # decodes to 4 bytes, not 32


def test_decrypt_with_wrong_key_fails(tmp_path: Path) -> None:
    source = tmp_path / "original.bin"
    source.write_bytes(b"data")
    blob = tmp_path / "blob.enc"
    encrypt_file(load_key(generate_key()), source, blob)
    with pytest.raises(DatasetEncryptionError):
        decrypt_file(load_key(generate_key()), blob, tmp_path / "out.bin")


def test_prepare_datasets_decrypts_missing_plaintext(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path=tmp_path)
    plain_messages = "message_id,timestamp,sender,message\n"
    plain_mandatory = "message_id\n"
    settings.messages_csv_path.write_text(plain_messages, encoding="utf-8")
    settings.mandatory_demo_ids_path.write_text(plain_mandatory, encoding="utf-8")

    key = generate_key()
    fernet = load_key(key)
    settings.encrypted_data_dir.mkdir(parents=True, exist_ok=True)
    encrypt_file(
        fernet,
        settings.messages_csv_path,
        settings.encrypted_data_dir / MESSAGES_ENC_FILENAME,
    )
    encrypt_file(
        fernet,
        settings.mandatory_demo_ids_path,
        settings.encrypted_data_dir / MANDATORY_ENC_FILENAME,
    )

    settings.messages_csv_path.unlink()
    settings.mandatory_demo_ids_path.unlink()
    assert not settings.messages_csv_path.exists()

    monkeypatch.setenv(DATASET_KEY_ENV, key)
    prepare_datasets(settings)
    assert settings.messages_csv_path.read_text(encoding="utf-8") == plain_messages
    assert settings.mandatory_demo_ids_path.read_text(encoding="utf-8") == plain_mandatory


def test_prepare_datasets_noop_when_plaintext_present(tmp_path: Path) -> None:
    settings = make_settings(tmp_path=tmp_path)
    settings.messages_csv_path.write_text("already here", encoding="utf-8")
    settings.mandatory_demo_ids_path.write_text("ids", encoding="utf-8")
    prepare_datasets(settings)
    assert settings.messages_csv_path.read_text(encoding="utf-8") == "already here"


def test_prepare_datasets_missing_key_raises(tmp_path: Path) -> None:
    settings = make_settings(tmp_path=tmp_path)
    settings.mandatory_demo_ids_path.write_text("ids", encoding="utf-8")
    with pytest.raises(DatasetEncryptionError):
        prepare_datasets(settings)


def test_prepare_datasets_missing_blob_raises(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path=tmp_path)
    monkeypatch.setenv(DATASET_KEY_ENV, generate_key())
    settings.messages_csv_path.write_text("here", encoding="utf-8")
    with pytest.raises(DatasetEncryptionError):
        prepare_datasets(settings)


def test_resolve_key_prefers_environment(tmp_path: Path, monkeypatch) -> None:
    key_file = tmp_path / "key"
    key_file.write_text(generate_key() + "\n", encoding="utf-8")
    env_key = generate_key()
    monkeypatch.setenv(DATASET_KEY_ENV, env_key)
    token = Fernet(env_key.encode("ascii")).encrypt(b"payload")
    assert resolve_key(key_file).decrypt(token) == b"payload"
