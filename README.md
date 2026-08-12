# Message Intelligence Pipeline

Hybrid message-intelligence pipeline: ingests a 900-message CSV, detects and
masks sensitive information, classifies every message into one of six
categories (deterministic rules first, optional LLM fallback), and later
extracts tasks/events. Output is served through a FastAPI dashboard.

> **Checkpoint document.** This README records every implemented subsystem and
> the current verification status so that work can resume at any time. Read the
> "Implemented subsystems" and "Roadmap" sections before continuing.

---

## 1. Project overview

| Area | Choice |
| --- | --- |
| Language | Python >= 3.14 (repo `pyproject.toml`; assignment requires 3.12+) |
| API | FastAPI |
| Validation | Pydantic v2 |
| Data | pandas |
| Tests | pytest |
| Lint / types | ruff, mypy |
| External services | None required; LLM is an optional, swappable fallback |

The pipeline must never leak sensitive values (see [Security rules](#5-security-rules)).

## 2. Dataset

- `messages.csv` — 900 fictional messages, chronological, UTF-8 (BOM). Columns:
  `message_id`, `timestamp` (`%Y-%m-%d %H:%M:%S`), `sender`, `message`.
  - 900 rows, all IDs unique, no null/empty cells, strictly chronological.
  - Earliest: `2026-09-01 08:00:00` · Latest: `2026-09-24 10:23:00`.
  - 13 senders (e.g. Meera, Ishaan, Kabir, Aarav, Ananya, Neha, Tara, Rohan,
    Vikram, Maya, Promotions, Project Lead, HR Team).
- `mandatory_demo_ids.csv` — 15 message IDs that must appear in the demo.
- **`messages.csv` is intentionally NOT committed to git** (the assignment
  forbids publishing the dataset in a public repository). It is listed in
  `.gitignore`. The pipeline loads it from the working directory.

## 3. Architecture

```
CSV
 ↓
Input Validator          (loader.py + validator.py)
 ↓
Sensitive Data Detector  (sensitive_detector.py)
 ↓
Mask Sensitive Data      (masker.py)
 ↓
Hybrid Classifier        (classifier.py)
   ├── deterministic rules first
   └── LLM fallback only below confidence threshold
 ↓
Task/Event Extractor     (NOT YET IMPLEMENTED - see Roadmap)
 ↓
Pydantic Validation
 ↓
Output Validation
 ↓
JSON artifacts
 ↓
FastAPI dashboard
 ↓
Render
```

## 4. Implemented subsystems (task log)

### 4.1 Task 1 — Sensitive information detection and masking (DONE)

Detects sensitive values **before** any message can reach an external LLM.

**Files**
- `app/models/sensitive.py` — internal vs public representation
- `app/services/sensitive_detector.py` — the detector
- `app/services/masker.py` — the masker
- `tests/test_sensitive.py` — 82 tests

**Models (`app/models/sensitive.py`)**
- `SensitiveType` (14 values): `one_time_password`, `password`, `pin`,
  `authentication_token`, `account_recovery_code`, `payment_card_number`,
  `bank_account_number`, `upi_payment_identifier`, `private_phone_number`,
  `private_email`, `private_address`, `identification_number`,
  `health_information`, `other_sensitive_credential`.
- `RiskLevel`: `low | medium | high`.
- `SensitiveDetection` — **internal** result. The raw matched value is stored in
  a Pydantic `PrivateAttr` (with `start`/`end` span) and is therefore excluded
  from `model_dump()` / `model_dump_json()`.
- `PublicSensitiveDetection` — sanitized result (`detected`, `sensitivity_type`,
  `risk`, `masked_text`, `recommended_action`) with no raw value.
- `SensitiveAnalysis` — per-message analysis: `message_id`, `detections`,
  `safe_message`, `has_detection`.

**Detection strategy (`app/services/sensitive_detector.py`)**
Combination of carefully designed regexes + contextual keyword signals +
validation heuristics (length checks, Luhn checksum for cards, plain-word
rejection, digit-count checks for phone numbers, single-label-domain rejection
for UPI). Detection is deliberately conservative to avoid false positives.

Masking preserves surrounding context:
- Star-masked same length: OTP / password / PIN / recovery code / card / bank /
  UPI (e.g. `Your OTP is 482913` → `Your OTP is ******`).
- Bracket-masked: `[REDACTED_TOKEN]`, `[REDACTED_PHONE]`, `[REDACTED_EMAIL]`,
  `[REDACTED_ADDRESS]`, `[REDACTED_ID]`, `[REDACTED_HEALTH]`, `[REDACTED]`.

**Notable bugs fixed in this task**
1. `_phone_context_re` was compiled without `re.IGNORECASE`, so uppercase
   `"Call me at ..."` / `"Contact Me On ..."` were missed.
2. `_other_re` captured the word `are` instead of the credential for
   `"credentials are admin:Passw0rd"`; added an `are` connector and a
   colon-tolerant value class.
3. `_upi_fmt_re` partial-matched real emails (`help@store` inside
   `help@store.example.com`) causing false-positive UPI detections; added a
   `(?!\.\w)` lookahead.
4. `_is_address_value` re-detected an already-masked `[REDACTED_ADDRESS]`; added
   redaction-marker rejection so masking is idempotent.

**Test coverage**: every sensitive type, multiple sensitive values in one
message, false-positive-ish ordinary numbers, ordinary email/newsletter
messages, already-masked messages (idempotent), case variations, empty input,
security tests (raw values never appear in masked output, serialized detection,
or public model dump), and a full-dataset leak scan.

### 4.2 Task 2 — Hybrid message classification (DONE)

One of exactly six categories per message, with confidence, reason and method.

**Files**
- `app/models/classification.py`
- `app/services/classifier.py`
- `tests/test_classifier.py` — 35 tests

**Categories (`app/models/classification.py`)**
`action_required`, `meeting_or_event`, `personal_information`,
`general_information`, `promotional`, `sensitive_information`.

`ClassificationResult`: `message_id`, `category`, `confidence` (float in
`[0, 1]`), `reason` (short, truncated to 300 chars), `method`
(`rule_based` | `llm_fallback`).

**Pipeline (`app/services/classifier.py`)**
1. Detect sensitive values → mask → the classifier only ever sees the sanitized
   message.
2. `RuleClassifier` — deterministic, context-aware scoring for action / meeting /
   promotional / personal signals, plus a sensitive path when detections exist.
   Scores below `_RULE_CONFIDENCE_FLOOR` fall back to `general_information`.
3. If rule confidence >= `llm_confidence_threshold` the rule result is accepted.
   Otherwise the LLM fallback is consulted; any failure falls back to the rule
   result. **A single LLM failure never crashes the 900-message pipeline.**

**LLM interface**
- `MessageClassifierLLM` (ABC) — swappable provider; receives
  `(message_id, safe_message)` only, returns a `ClassificationResult` or `None`.
- `BaseLLMClassifier` — shared prompt building + robust parsing; subclasses
  implement `_invoke(prompt) -> str`.
- `parse_llm_response` — handles fenced JSON, malformed JSON, unknown category,
  out-of-range / unparseable confidence (clamped), message_id mismatch, empty
  responses.
- The prompt mandates exactly one category, no invented facts, preserved
  message_id, confidence in `[0, 1]`, short reason, and states that sensitive
  values are already masked.

**Notable bugs fixed in this task**
1. **Critical:** the LLM prompt template contained literal JSON braces, so
   `str.format()` raised `KeyError` on every call and the LLM fallback silently
   never ran. Braces were escaped.
2. `MessageClassifier` did not catch provider exceptions; wrapped the LLM
   invocation so one failing call falls back instead of crashing.
3. `max(scores, key=scores.get)` mypy `arg-type` error → replaced with a lambda.

### 4.3 Task 3 — Dataset ingestion + input validation (DONE)

**Files**
- `app/services/loader.py` — safe pandas loading, statistics, typed models
- `app/services/validator.py` — all input validation rules
- `app/models/message.py` — `RawMessage` (frozen)
- `app/models/dataset.py` — `DatasetStatistics`
- `app/config.py` — `Settings`, env-overridable paths, `pathlib`
- `tests/test_validation.py` — 26 tests

**`loader.load_messages_csv(path, *, expected_count=900, encoding='utf-8-sig')`**
- Raises `DatasetLoadingError` for missing/unreadable/malformed files.
- Never modifies the input CSV (read-only).
- Returns `LoadedDataset(messages, statistics, source_path)`.
- Independent of FastAPI.

**`validator.validate_dataset(df, *, expected_count)`**
Validates: required columns; exact size; non-null, unique `message_id`;
parseable timestamps; non-empty `message` and `sender`; chronological order.
All problems are reported at once via `DatasetValidationError` containing
`ValidationIssue(code, detail)` tuples with codes such as
`missing_columns`, `unexpected_dataset_size`, `duplicate_message_id`,
`empty_message_id`, `malformed_timestamp`, `not_chronological`,
`missing_message_content`, `missing_sender`.

**`loader.dataset_statistics(messages)`** returns `DatasetStatistics`:
`total_messages`, `unique_message_ids`, `earliest_timestamp`,
`latest_timestamp`, `empty_message_count`, `empty_sender_count`.

### 4.4 Cross-cutting work

- Enums migrated from `str, Enum` to `StrEnum` (ruff `UP042`).
- Line-length (100) formatting fixes across touched files.
- Full repo passes `ruff check .` and `mypy app tests`.

## 5. Security rules

- Raw sensitive values exist **only** in `SensitiveDetection` private
  attributes; they are excluded from every serialization.
- Never write raw sensitive values to logs, JSON artifacts, API responses,
  frontend, screenshots, or generated reports.
- The LLM (when configured) only ever receives the masked/sanitized message.
- `messages.csv` is excluded from git; `.env` and `outputs/*.json` are
  gitignored. Do not commit API keys or secrets.

## 6. Verification status (latest run)

| Check | Result |
| --- | --- |
| `pytest` | **143 passed** (26 validation + 82 sensitive + 35 classifier) |
| `ruff check .` | All checks passed |
| `mypy app tests` | No issues found in 20 source files |
| Dataset leak scan | 0 leaks across all 900 messages |

Commands:

```bash
python -m pytest
python -m pytest tests/test_validation.py -v
python -m pytest tests/test_sensitive.py -v
python -m pytest tests/test_classifier.py -v
ruff check .
mypy app tests
```

## 7. Configuration

Paths and thresholds are configurable via environment variables (see
`app/config.py`), never hardcoded:

| Env var | Default | Purpose |
| --- | --- | --- |
| `MESSAGES_CSV_PATH` | `./messages.csv` | Input dataset |
| `MANDATORY_DEMO_IDS_PATH` | `./mandatory_demo_ids.csv` | 15 demo-required IDs |
| `OUTPUTS_DIR` | `./outputs` | Generated JSON artifacts |
| `EXPECTED_MESSAGE_COUNT` | `900` | Exact expected dataset size |
| `LLM_CONFIDENCE_THRESHOLD` | `0.75` | Rule confidence below which the LLM fallback is consulted |

## 8. Roadmap (not yet implemented)

Ordered next steps:

1. **Task/Event extraction** — `app/models/task_event.py` + `app/services/extractor.py`
   (item_id, type, title, description, date, deadline, time, person, priority,
   source_message_id). Fields genuinely unavailable must be `null`, never
   fabricated.
2. **Pipeline runner** — `scripts/run_pipeline.py`: load → detect/mask →
   classify → extract → validate → write Pydantic-validated JSON artifacts.
3. **Output validation** — validate every generated structured artifact; fail
   clearly rather than writing corrupt output.
4. **FastAPI dashboard** — `app/main.py` + `app/templates/index.html`: demo
   showing the 15 mandatory message IDs.
5. **Render deployment** — wire `render.yaml`.

Do not hardcode classifications or extractions for individual message IDs.
Preserve original message IDs; never silently drop or duplicate messages.
