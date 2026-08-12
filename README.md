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
- **Privacy rule:** the assignment forbids publishing the dataset in a public
  repository, so the plaintext CSVs are **not** committed. The repository
  instead commits Fernet-encrypted blobs (`data/messages.csv.enc`,
  `data/mandatory_demo_ids.csv.enc`) plus a decryption routine. The pipeline
  materializes the plaintext CSVs on demand from the committed blobs using a
  key supplied via the `DATASET_ENC_KEY` environment variable (production) or
  the gitignored `data/.dataset.key` file (local development) — see
  §4.7 and §7. `scripts/encrypt_dataset.py` produces the blobs.

## 3. Architecture

```
Encrypted dataset blobs  (data/*.enc, committed; key from DATASET_ENC_KEY)
 ↓  prepare_datasets() / python -m scripts.encrypt_dataset (encrypts the other way)
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
Task/Event Extractor     (extractor.py)
   ├── deterministic rules first
   └── LLM fallback when nothing matched
 ↓
 Pydantic Validation
 ↓
JSON artifacts           (run_pipeline.py: classifications / tasks_events /
                          sensitive_detections / final_results)
 ↓
Output Validation + Leak Scan + Mandatory Demo
                         (output_validator.py + leak_scanner.py + mandatory_demo.py)
 ↓
validation_report.json
 ↓
FastAPI dashboard        (app/main.py + app/templates/dashboard.html)
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

### 4.4 Task 4 — Task/event extraction (DONE)

Extracts tasks, meetings, events and reminders from sanitized messages.

**Files**
- `app/models/task_event.py` — `ItemType` (`task | meeting | event | reminder`),
  `Priority` (`low | medium | high | unknown`), `ExtractedItem` (frozen),
  `ExtractorMethod`, `ExtractionResult`.
- `app/services/extractor.py` — `ExtractionRules`, LLM interface, `MessageExtractor`.
- `tests/test_extraction.py` — 84 tests

**Fields** (`ExtractedItem`): `item_id`, `type`, `title`, `description`, `date`,
`deadline`, `time` (24h `HH:MM`), `person`, `priority`, `source_message_id`.
Fields genuinely unavailable are `null` / `unknown` — nothing is fabricated.

**Item IDs** are deterministic and derived from the source message:
`<TYPE>_<message_id>`, e.g. `TASK_MSG_0002`, `EVENT_MSG_0007`. When one message
yields several items of the same type they are indexed with a `-N` suffix
(`TASK_MSG_0002-2`, `TASK_MSG_0002-3`). The format is covered by dedicated unit
tests and asserted across the whole dataset (`make_item_id`).

**Relative-date resolution** — relative expressions (`today`, `tomorrow`,
`yesterday`, bare or qualified weekdays, `in <n> days`) are resolved against
**the message timestamp**, never the current system date
(`resolve_relative_date` + `extract_from_safe(..., reference_date=...)`, where
the reference is `message.timestamp.date()`). Vague or conditional phrasing
(`could be Friday`, `may be needed tomorrow`, `sometime next week`) is never
turned into a definite date. Explicit ISO dates need no reference.

**Deterministic rules** (`ExtractionRules`), evaluated in precedence order:
1. `Calendar update: <title>, <date> at <time>, <loc>` → `event`
2. `Reminder: <title> happens on <date> at <time> in <loc>` → `reminder`
3. `The <title> is scheduled for <date> at <time> in <loc>` → `meeting`
4. `Please join the <title> on <date>, <time> at <loc>` → `event`
5. `Are you available for the <title> at <time> on <date>? Location: ...` → `meeting`
6. **Missing-time variants** of 1–3 (explicit date, no time) → the same item with
   `time=None` — the time is never guessed.
7. Explicit deadline (`by|before|deadline is|is due on|due on` + date) → `task`
   with the deadline captured and a clean action title
8. **Multi-task rule** — a message with several distinct explicit deadlines
   yields one task per deadline (with `-2`, `-3`, ... suffixes); each title is
   split from the text between consecutive deadlines.
9. `Please call <Name>` / `call the <target>` → `task` with `person` where a name
   is present

Times are normalized to 24h `HH:MM` (`9 AM` → `09:00`, `8 pm` → `20:00`).
Priority is `high` only for explicit urgency words, `low` for flexibility
signals, else `unknown`. Message prefixes (`For today:`, `FYI:`, ...) and
polite softeners (`please`, `could you`, `i need you to`, ...) are stripped from
task titles.

**LLM interface** (same pattern as the classifier):
- `MessageExtractorLLM` (ABC) — receives `(message_id, safe_message)` only,
  returns items or `None` on any failure.
- `BaseLLMExtractor` — shared prompt + robust parsing.
- `parse_llm_response` — accepts `{"items": [...]}` or a bare list; validates
  types/dates/times; rejects unknown types, empty titles, bad dates/times and
  `message_id` mismatches; indexes repeated item types (`-2`, `-3`, ...).

**Pipeline** (`MessageExtractor`): mask → deterministic rules; LLM fallback only
when nothing matched; a single LLM failure never crashes the pipeline.

### 4.5 Task 5 — Pipeline runner + output validation + leak scan (DONE)

**Files**
- `app/models/pipeline.py` — `MessageSensitiveResult` (public, `extra="forbid"`),
  `MessagePipelineResult` (carries `timestamp` + `sender`),
  `FinalMessageResult` (sanitized per-message output), `PipelineSummary`
  (failure counter + processing duration), `PipelineRunResult`.
- `app/services/pipeline.py` — `PipelineRunner` (detect → mask → classify →
  extract) with LLM failure counters and processing duration.
- `app/services/output_validator.py` — per-artifact validation +
  `QualityReport` (the `validation_report.json` structure).
- `app/services/leak_scanner.py` — `LeakScanner`, findings never carry raw values.
- `app/services/mandatory_demo.py` — mandatory 15-ID demo coverage.
- `app/services/llm_provider.py` — optional HTTP (OpenAI-compatible) clients.
- `scripts/run_pipeline.py` — writes the five JSON artifacts.
- `tests/test_pipeline.py` (17) · `tests/test_output_validator.py` (32) ·
  `tests/test_mandatory_demo.py` (14)

**`scripts/run_pipeline.py`** (run with `python -m scripts.run_pipeline`) loads
the dataset, runs every stage, and writes to `outputs/`:
1. `classifications.json` — per-message `ClassificationResult`
2. `tasks_events.json` — per-message `ExtractionResult`
3. `sensitive_detections.json` — per-message public detections (masked only)
4. `final_results.json` — sanitized per-message `FinalMessageResult`
   (`message_id`, `timestamp`, `sender`, `classification`, `security`,
   `extracted_items`; deliberately no raw message text)
5. `validation_report.json` — run summary + `QualityReport` + leak scan

Exit code 0 on success, 1 when validation or the leak scan fails.

**Output validation** (`output_validator.py`): every record must parse into its
typed Pydantic model; every message ID preserved exactly once (missing /
duplicate / unknown IDs reported); item IDs unique; item `source_message_id`
matches its message; classification category is one of the six and confidence is
in `[0, 1]`. `build_quality_report` produces the consolidated `QualityReport`:
dataset integrity (900 in / 900 out), valid categories/confidences, task/event
count, sensitive-message count, mandatory demo coverage and the leak check.

**Mandatory demo** (`mandatory_demo.py`): loads the 15 required IDs from
`mandatory_demo_ids.csv` (validating count and uniqueness), checks them against
the dataset and pipeline outputs (`MandatoryDemoCheck`), and serves the complete
processed results for exactly those messages in original dataset chronological
order (`MandatoryDemoService.build`). Missing messages raise instead of ever
fabricating a result.

**Leak scan** (`leak_scanner.py`): re-runs sensitive detection on each original
message and verifies no raw matched value appears in any artifact's JSON text.
`LeakFinding` stores message_id, artifact, sensitivity_type only.

**LLM providers** (`llm_provider.py`): optional OpenAI-compatible HTTP clients
for both classifier and extractor; they only ever receive the masked message.
`build_llm_components(settings)` returns `(None, None)` — fully offline —
unless the LLM fallback is enabled with a key and model configured.

### 4.6 Cross-cutting work

- Enums migrated from `str, Enum` to `StrEnum` (ruff `UP042`).
- Line-length (100) formatting fixes across touched files.
- `app/config.py` extended with configurable LLM settings and an offline mode
  (default: `llm_enabled=False` → fully deterministic, no external calls).
- Full repo passes `ruff check .` and `mypy app tests scripts`.
- New modules follow the frozen-Pydantic / ABC-provider conventions of earlier
  tasks; the `date` model field is annotated via `from datetime import date as
  Date` to avoid class-namespace shadowing.

### 4.7 Dataset encryption (privacy) (DONE)

**Files**
- `app/services/dataset_cipher.py` — Fernet helpers + `prepare_datasets`
- `scripts/encrypt_dataset.py` — encrypts plaintext CSVs into `data/*.enc`
- `tests/test_dataset_cipher.py` — 8 tests

The assignment forbids publishing the dataset in a public repository, so the
plaintext CSVs are never tracked. Instead:
- `scripts/encrypt_dataset.py` encrypts `messages.csv` and
  `mandatory_demo_ids.csv` into `data/messages.csv.enc` and
  `data/mandatory_demo_ids.csv.enc` and verifies a byte-identical round trip.
- The key comes from the `DATASET_ENC_KEY` environment variable or the
  gitignored `data/.dataset.key` file (`--new-key` forces a fresh key).
- `prepare_datasets(settings)` (called at the start of
  `python -m scripts.run_pipeline`) decrypts a plaintext CSV back into place
  exactly when it is missing, so a fresh clone can regenerate the dataset
  before the pipeline runs.
- On Render the key is a dashboard secret (`DATASET_ENC_KEY` in `render.yaml`),
  never committed.

### 4.8 Task 6 — FastAPI dashboard (DONE)

**Files**
- `app/main.py` — FastAPI app: dashboard + sanitized read API
- `app/models/api.py` — response models
- `app/services/api_service.py` — filtering/pagination/stats/mandatory demo
- `app/services/output_repository.py` — typed reads of the `outputs/` artifacts
- `app/templates/dashboard.html`, `app/templates/base.html` — dashboard UI
- `app/static/app.js`, `app/static/styles.css` — client assets
- `tests/test_api.py` — 24 tests

The app is a thin presentation layer over the validated pipeline artifacts:
`GET /` (dashboard), `GET /health`, `/api/stats`, `/api/messages`,
`/api/messages/{id}`, `/api/tasks`, `/api/sensitive`, `/api/demo/mandatory`,
`/api/validation`. Responses only ever serialize the pipeline's sanitized
models; the raw message text and sensitive values never reach the API.
`app = create_app()` at module import; run with `uvicorn app.main:app`.

## 5. Security rules

- Raw sensitive values exist **only** in `SensitiveDetection` private
  attributes; they are excluded from every serialization.
- Never write raw sensitive values to logs, JSON artifacts, API responses,
  frontend, screenshots, or generated reports.
- The LLM (when configured) only ever receives the masked/sanitized message.
- The plaintext dataset CSVs and the Fernet key (`DATASET_ENC_KEY`,
  `data/.dataset.key`) are never committed or logged; only the encrypted
  `data/*.enc` blobs live in the repository.
- `.env` and `outputs/*.json` are gitignored; API keys and secrets must never be
  committed.

## 6. Verification status (latest run)

| Check | Result |
| --- | --- |
| `pytest` | **322 passed** (26 validation + 82 sensitive + 35 classifier + 84 extraction + 32 output validation/leak + 17 pipeline + 14 mandatory demo + 8 dataset cipher + 24 API) |
| `ruff check .` | All checks passed |
| `mypy app tests scripts` | No issues found in 38 source files |
| `python -m scripts.run_pipeline` | 900 messages · 360 items extracted (60 event / 60 meeting / 30 reminder / 210 task) · 100 sensitive messages · 15/15 mandatory demo · 0 validation issues · 0 leaks · ~0.2s · validation PASS |

Commands:

```bash
python -m pytest
python -m pytest tests/test_validation.py -v
python -m pytest tests/test_sensitive.py -v
python -m pytest tests/test_classifier.py -v
python -m pytest tests/test_extraction.py -v
python -m pytest tests/test_output_validator.py -v
python -m pytest tests/test_pipeline.py -v
python -m pytest tests/test_mandatory_demo.py -v
python -m pytest tests/test_dataset_cipher.py -v
python -m pytest tests/test_api.py -v
ruff check .
mypy app tests scripts
python -m scripts.encrypt_dataset        # (re)create data/*.enc from plaintext CSVs
python -m scripts.run_pipeline           # decrypt-on-demand, then process
uvicorn app.main:app                     # local dashboard
```

## 7. Configuration

Paths and thresholds are configurable via environment variables (see
`app/config.py`), never hardcoded:

| Env var | Default | Purpose |
| --- | --- | --- |
| `MESSAGES_CSV_PATH` | `./messages.csv` | Input dataset (decrypted on demand) |
| `MANDATORY_DEMO_IDS_PATH` | `./mandatory_demo_ids.csv` | 15 demo-required IDs |
| `OUTPUTS_DIR` | `./outputs` | Generated JSON artifacts |
| `ENCRYPTED_DATA_DIR` | `./data` | Committed `*.enc` Fernet blobs |
| `DATASET_KEY_FILE` | `./data/.dataset.key` | Local gitignored Fernet key file |
| `DATASET_ENC_KEY` | *(empty)* | Fernet key secret; Render dashboard env var |
| `EXPECTED_MESSAGE_COUNT` | `900` | Exact expected dataset size |
| `EXPECTED_MANDATORY_COUNT` | `15` | Exact expected mandatory-ID count |
| `LLM_ENABLED` | `false` | Offline mode; when false the pipeline is fully deterministic |
| `LLM_PROVIDER` | `openai` | Provider name (OpenAI-compatible) |
| `LLM_MODEL` | *(empty)* | Model identifier sent to the provider |
| `LLM_API_KEY` | *(empty)* | Secret key; never logged or serialized |
| `LLM_BASE_URL` | *(empty)* | Optional provider base URL override |
| `LLM_TIMEOUT_SECONDS` | `30` | Timeout for a single LLM request |
| `LLM_CONFIDENCE_THRESHOLD` | `0.75` | Rule confidence below which the LLM fallback is consulted |

Copy `.env.example` to `.env` for local overrides (the pipeline reads
environment variables directly; `python-dotenv` is available).

## 8. Deployment (Render) — ready to ship

The dashboard and API are implemented and the `render.yaml` Blueprint is wired.

1. The GitHub repository must stay **private** (the assignment forbids
   publishing the dataset; only encrypted blobs are committed).
2. Generate the dataset key once: `python -m scripts.encrypt_dataset`. Copy the
   printed Fernet key into the Render dashboard as the `DATASET_ENC_KEY` secret
   (keep `sync: false` in `render.yaml`; never commit it).
3. Push `main`. Render Blueprint builds with `pip install -r requirements.txt`
   (Python `3.14.3`, from `.python-version`) and starts with
   `python -m scripts.run_pipeline && uvicorn app.main:app --host 0.0.0.0 --port $PORT`,
   which decrypts the dataset, regenerates the artifacts, and serves the
   dashboard. No external services or LLM calls are required.

Do not hardcode classifications or extractions for individual message IDs.
Preserve original message IDs; never silently drop or duplicate messages.
