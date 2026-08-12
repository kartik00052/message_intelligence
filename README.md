# Message Intelligence

A hybrid message-intelligence pipeline for the L1 candidate assignment. It
ingests a 900-message CSV, detects and masks sensitive information, classifies
every message into exactly one of six categories, extracts actionable tasks and
scheduled events, validates every artifact, and serves the results through a
FastAPI dashboard.

The pipeline is fully deterministic and offline by default. An optional LLM
fallback can be enabled for low-confidence classifications and unmatched
extractions; it is disabled unless explicitly configured and only ever receives
masked message text.

> **Privacy requirement (from `README.txt`):** the dataset must not be
> published in a public repository and raw messages must never be sent to
> external AI services. This repository therefore commits only
> Fernet-encrypted dataset blobs and never sends raw text anywhere (see
> [§6 Sensitive Information Detection](#6-sensitive-information-detection) and
> [§22 Privacy/Security Notes](#22-privacysecurity-notes)).

---

## 1. Overview

The assignment: build a system that processes a folder of fictional messages
and turns them into structured, safe-to-share intelligence.

This system implements that end-to-end:

1. **Ingest** a 900-message CSV (fictional but deliberately sensitive-looking
   values), validating schema, size, IDs, timestamps and ordering.
2. **Detect and mask** sensitive values *before* anything else touches the
   message, so no raw credential ever reaches a downstream stage.
3. **Classify** every message into one of six categories with a confidence
   score, a short reason and the method used.
4. **Extract** tasks, meetings, events and reminders with dates, deadlines,
   times and priority - without guessing a single field.
5. **Validate** every output artifact and scan it for sensitive-value leaks.
6. **Serve** the sanitized results through a FastAPI dashboard (including the
   15 messages required for the demo video) and document a Render deployment.

No answer labels are used; every category and extraction is derived from the
message content itself.

## 2. Features

- **Message classification** - every message is classified into exactly one of
  six categories with a confidence score, a reason and the method used.
- **Task/event extraction** - tasks, meetings, events and reminders with
  date/deadline/time/person/priority fields; several explicit deadlines in one
  message yield one task each.
- **Sensitive information detection** - 14 sensitive types (OTPs, passwords,
  PINs, tokens, recovery codes, cards, bank accounts, UPI IDs, phone numbers,
  emails, addresses, ID numbers, health data and other credentials).
- **Masking** - detected values are replaced in-place (star or bracket masks)
  so the safe message keeps its surrounding context.
- **Hybrid rules + LLM fallback** - deterministic rules run first; an optional
  LLM is consulted only for low-confidence classifications or unmatched
  extractions, and only ever sees masked text. Fully offline by default.
- **Structured outputs** - every result is a Pydantic-validated model and is
  written to machine-readable JSON artifacts.
- **Validation** - input validation, output validation (IDs preserved exactly
  once, schemas enforced) and a sensitive-value leak scan.
- **FastAPI dashboard** - a single-page dashboard over the sanitized artifacts
  plus a typed read-only API.
- **Mandatory message demo** - the 15 required message IDs are loaded from a
  CSV (never hardcoded) and shown in dataset order.

## 3. Architecture

The pipeline runs every message through these stages in order:

```
CSV
→ Input Validation
→ Sensitive Detection
→ Masking
→ Hybrid Classification
→ Task/Event Extraction
→ Pydantic Validation
→ Output Validation
→ JSON Artifacts
→ FastAPI Dashboard
→ Render
```

- `app/services/loader.py` + `app/services/validator.py` read and validate the
  input CSV into typed `RawMessage` objects.
- `app/services/sensitive_detector.py` runs before classification/extraction
  and `app/services/masker.py` replaces every detected span in-place.
- `app/services/classifier.py` and `app/services/extractor.py` only ever see
  the masked message. Both implement deterministic rules first with an
  optional LLM fallback.
- `app/services/pipeline.py` orchestrates detection → masking → classification
  → extraction and produces validated `MessagePipelineResult` models.
- `app/services/output_validator.py` and `app/services/leak_scanner.py`
  validate the artifacts and check for sensitive-value leaks.
- `scripts/run_pipeline.py` writes the JSON artifacts to `outputs/`.
- `app/main.py` serves the FastAPI dashboard and read API over the artifacts.
- `render.yaml` deploys the app to Render.

The LLM is optional and swappable (`app/services/llm_provider.py`); the
default configuration returns `(None, None)` so the pipeline is fully offline
and deterministic.

## 4. Classification

Each message is assigned **exactly one** of six categories:

| Category | Meaning |
| --- | --- |
| `action_required` | A request to do something (task verb, direct ask or deadline). |
| `meeting_or_event` | A meeting or event with schedule, date, time or location signals. |
| `personal_information` | A personal fact, preference or profile detail. |
| `general_information` | An informational statement with no action, event or offer. |
| `promotional` | Promotional content: an offer, sale or promo code. |
| `sensitive_information` | A message containing detected sensitive content. |

`ClassificationResult` carries `message_id`, `category`, `confidence` (a float
in `[0, 1]`), `reason` (a short justification, capped at 300 characters) and
`method` (`rule_based` or `llm_fallback`).

**Approach.** The `RuleClassifier` scores four candidate categories with
context-aware signals:

- **Action**: action phrases (`please submit`, `review the`, `don't forget`,
  ...), weak asks (`can you`, `could you`, ...) and deadline signals
  (`by <date>`, `deadline`, `due on`, `asap`, ...).
- **Meeting/event**: strong nouns (`calendar`, `meeting`, `dinner`, `demo`,
  `session`, ...), weak nouns (`meet`, `review`, ...) and context regexes
  (ISO dates, `at <time>`, `tomorrow`, `next week`, locations such as `meeting
  room` or `the library`, ...).
- **Promotional**: offer phrases (`discount`, `sale`, `cashback`, `% off`,
  `use code`, ...) and promo-code patterns (`code SAVE30`, ...).
- **Personal**: preference phrases (`my favourite`, `i prefer`, `i like`, `my
  birthday`, ...).

The category with the highest score wins. Scores below the rule confidence
floor (`0.65`) are classified as `general_information` with a fixed confidence
of `0.5`. When sensitive detections exist the message is classified
`sensitive_information` (confidence `0.97` for high-risk types, else `0.92`).

The `MessageClassifier` masks the message first, applies the rules, and only
calls the LLM when the rule confidence is below `llm_confidence_threshold`
(default `0.75`).

## 5. Task/Event Extraction

`ExtractedItem` is the validated representation of one actionable or scheduled
item. Fields that are genuinely unavailable are `null`/`unknown` - nothing is
fabricated.

**Schema**

| Field | Meaning |
| --- | --- |
| `item_id` | Deterministic ID: `<TYPE>_<message_id>` (e.g. `TASK_MSG_0002`); repeated types get `-2`, `-3` suffixes. |
| `type` | `task`, `meeting`, `event` or `reminder`. |
| `title` | Short, cleaned title derived only from the message. |
| `description` | Optional context (location etc.), else `null`. |
| `date` | Calendar date (`YYYY-MM-DD`) when explicitly stated. |
| `deadline` | Due date for tasks when explicitly stated. |
| `time` | Normalized 24h `HH:MM` when explicitly stated. |
| `person` | A clearly named person, else `null`. |
| `priority` | `low`, `medium`, `high` or `unknown`. |
| `source_message_id` | The message the item was extracted from. |

**Date handling.** Explicit ISO dates (`2026-09-04`) are parsed directly.
Relative expressions are resolved against **the message timestamp**, never the
current system date.

**Relative dates.** `today`, `tomorrow`, `yesterday`, bare or qualified
weekdays (`friday`, `this friday`, `next friday`) and `in <n> days` (digit or
word based) are supported.

**Null/unresolved behavior.** Vague or conditional phrasing (`could be Friday`,
`may be needed tomorrow`, `sometime next week`) is never turned into a definite
date - it resolves to `null`. A missing reference date also yields `null`.

**Priority rules.** `high` for explicit urgency words (`urgent`, `asap`,
`critical`, `immediately`, `as soon as possible`); `low` for flexibility
signals (`no rush`, `whenever`, `when you are free`, `at your convenience`);
otherwise `unknown`.

**No guessing policy.** A field is populated only when it is explicitly stated
in the message. Message prefixes (`For today:`, `FYI:`, ...) and polite
softeners (`please`, `could you`, `i need you to`, ...) are stripped from
titles rather than interpreted as dates or constraints.

The deterministic `ExtractionRules` run in a fixed precedence order (calendar
updates, reminders, scheduled meetings, join requests, availability requests,
missing-time variants, explicit deadlines, call requests). A message with
several distinct explicit deadlines produces one task per deadline. When no
rule matches, the optional LLM fallback is consulted; if nothing matches, the
result is an empty item list with method `none`.

## 6. Sensitive Information Detection

Detection runs before any other stage so a masked value never reaches
classification, extraction, the LLM, the API or the UI.

**Detection strategy.** A combination of:

- **Regex/rule detection** - per-type regular expressions for OTPs, passwords,
  PINs, tokens (including `sk-`/`pk-`/`ghp_` prefixes and JWTs), recovery
  codes, card numbers, bank accounts, UPI IDs, phone numbers, emails,
  addresses, ID numbers (PAN, Aadhaar, passport, generic IDs), health terms and
  other credentials.
- **Contextual checks** - values are only flagged near explicit context
  (`call me at`, `my email is`, `card number`, `password is`, ...).
- **Validation heuristics** - Luhn checksum for card numbers, digit-count
  checks for phone numbers, plain-word rejection, already-masked rejection and
  single-label email-domain rejection for UPI handles.

Detection is deliberately conservative to avoid false positives.

**Masking.** The `Masker` replaces every detected span in-place, preserving all
surrounding context. Two mask styles:

- **Star masks (same length):** OTP, password, PIN, recovery code, card, bank
  account, UPI ID - e.g. `Your OTP is 482913` → `Your OTP is ******`.
- **Bracket masks:** tokens, phone, email, address, ID, health and other
  credentials - `[REDACTED_TOKEN]`, `[REDACTED_PHONE]`, `[REDACTED_EMAIL]`,
  `[REDACTED_ADDRESS]`, `[REDACTED_ID]`, `[REDACTED_HEALTH]`, `[REDACTED]`.

**Risk levels.** Each sensitive type maps to a risk: `high` for credentials
(OTP, password, PIN, token, recovery code, card, bank account, ID, health,
other) and `medium` for contact/profile identifiers (UPI ID, phone, email,
address).

**Recommended actions.** Every detection carries a type-specific
`recommended_action` (e.g. "Rotate the password and enable two-factor
authentication.", "Do not share the one-time password; treat it as compromised
if forwarded.").

> **Sensitive values are never intentionally written to logs, public API
> responses, generated outputs, screenshots, or the demo UI.**

The raw matched value is stored only in a Pydantic `PrivateAttr` inside the
internal `SensitiveDetection`, which is excluded from every serialization;
public models (`PublicSensitiveDetection`, `MessageSensitiveResult`) carry only
masked text. See [§22 Privacy/Security Notes](#22-privacysecurity-notes).

## 7. Hybrid AI Architecture

The pipeline combines deterministic rules with an optional LLM fallback:

- **Deterministic rules** run first for every message (classification scoring
  and extraction regex rules). In the default configuration this is the only
  path used - the pipeline is fully offline.
- **LLM fallback** is used only when the deterministic result is not
  confident enough:
  - *Classifier:* rule confidence below `llm_confidence_threshold` (default
    `0.75`).
  - *Extractor:* no deterministic rule matched.
- **Masked LLM input** - the LLM only ever receives `(message_id,
  safe_message)` where `safe_message` is the fully masked text. The prompt
  explicitly states that sensitive values are already masked and must not be
  reconstructed.
- **Structured LLM output** - the prompt mandates JSON. `parse_llm_response`
  validates category/items, clamps confidence to `[0, 1]`, enforces known
  types/priorities, validates dates and 24h times, checks the returned
  `message_id`, and tolerates fenced or partially malformed JSON.
- **Error handling** - a failed, empty or unusable LLM response returns `None`
  from the provider, and the orchestrator falls back to the deterministic
  result. Failures are counted in the pipeline summary. **A single LLM failure
  never crashes the 900-message run.**

Providers are swappable (`MessageClassifierLLM` / `MessageExtractorLLM` ABCs).
The bundled `OpenAIClassifierLLM` / `OpenAIExtractorLLM` call an
OpenAI-compatible chat-completions endpoint with `temperature=0`. Without an
enabled configuration, `build_llm_components` returns `(None, None)` and the
LLM is never invoked.

## 8. Validation

Validation is strict - problems fail loudly instead of being silently written.

- **Input validation** (`validator.py`): the CSV must have exactly the required
  columns (`message_id`, `timestamp`, `sender`, `message`), exactly the
  expected size (900), unique non-empty message IDs, parseable timestamps, and
  non-empty message/sender content in strictly chronological order. All issues
  are reported at once with machine-readable codes.
- **ID integrity** (`output_validator.py`): every artifact must contain every
  expected message ID exactly once - missing, duplicated or invented IDs are
  reported. Item IDs must be unique and each item's `source_message_id` must
  match its message.
- **Schema validation** - every artifact record must parse into its typed
  Pydantic model; classification categories must be one of the six and
  confidence must be in `[0, 1]`.
- **Privacy leak detection** (`leak_scanner.py`): the scanner re-runs sensitive
  detection on every original message and verifies that no raw matched value
  appears anywhere in the serialized JSON of any artifact. Findings never
  include the raw value - only message ID, artifact and sensitivity type.
- **Mandatory ID validation** (`mandatory_demo.py`): the 15 mandatory IDs are
  loaded from `mandatory_demo_ids.csv` (exact count, uniqueness), checked
  against the dataset and the outputs, and served in dataset order. Missing
  messages raise an error rather than fabricating a result.

The consolidated `QualityReport` (written to `validation_report.json`) reports
dataset integrity, valid categories/confidences, task/event count,
sensitive-message count, mandatory demo coverage and the leak check, with an
overall `validation_status` of `PASS` or `FAIL`.

## 9. Project Structure

```
message_intelligence/
├── app/
│   ├── main.py                    # FastAPI app: dashboard + read-only API
│   ├── config.py                  # Settings, env-overridable paths
│   ├── models/
│   │   ├── api.py                 # API response models
│   │   ├── classification.py      # Category, ClassificationResult
│   │   ├── dataset.py             # DatasetStatistics
│   │   ├── message.py             # RawMessage
│   │   ├── pipeline.py            # MessagePipelineResult, FinalMessageResult
│   │   ├── sensitive.py           # SensitiveType, risk, public/internal models
│   │   └── task_event.py          # ExtractedItem, ItemType, Priority
│   ├── services/
│   │   ├── api_service.py         # Dashboard/API business logic
│   │   ├── classifier.py          # RuleClassifier + MessageClassifierLLM
│   │   ├── dataset_cipher.py      # Fernet encrypt/decrypt + prepare_datasets
│   │   ├── extractor.py           # ExtractionRules + MessageExtractorLLM
│   │   ├── leak_scanner.py        # Sensitive-value leak scan of artifacts
│   │   ├── llm_provider.py        # Optional OpenAI-compatible clients
│   │   ├── loader.py              # CSV loading + typed messages
│   │   ├── mandatory_demo.py      # 15 mandatory IDs validation/serving
│   │   ├── masker.py              # In-place masking
│   │   ├── output_repository.py   # Typed reads of outputs/*.json
│   │   ├── output_validator.py    # Artifact validation + QualityReport
│   │   ├── pipeline.py            # End-to-end orchestrator
│   │   ├── sensitive_detector.py  # Sensitive-value detection
│   │   └── validator.py           # Input dataset validation
│   ├── static/
│   │   ├── app.js                 # Dashboard client logic
│   │   └── styles.css
│   └── templates/
│       ├── base.html
│       └── dashboard.html
├── data/
│   ├── messages.csv.enc           # Encrypted dataset blob (committed)
│   └── mandatory_demo_ids.csv.enc # Encrypted mandatory IDs blob (committed)
├── outputs/
│   └── .gitkeep                   # Generated JSON artifacts (gitignored)
├── scripts/
│   ├── encrypt_dataset.py         # Encrypt plaintext CSVs into data/*.enc
│   └── run_pipeline.py            # Run the full pipeline -> outputs/*.json
├── tests/                         # 322 tests across 9 files
│   ├── test_api.py
│   ├── test_classifier.py
│   ├── test_dataset_cipher.py
│   ├── test_extraction.py
│   ├── test_mandatory_demo.py
│   ├── test_output_validator.py
│   ├── test_pipeline.py
│   ├── test_sensitive.py
│   └── test_validation.py
├── .env.example                   # Documented environment variables
├── .gitignore
├── .python-version                # 3.14.3
├── pyproject.toml                 # Project metadata + dev tooling config
├── render.yaml                    # Render Blueprint
├── requirements.txt               # Runtime dependencies
├── uv.lock                        # Locked dependency tree
├── README.md                      # This document
└── README.txt                     # Assignment brief (unchanged)
```

The plaintext `messages.csv` and `mandatory_demo_ids.csv` and the local key
file `data/.dataset.key` exist in the working copy but are **gitignored** -
only the encrypted blobs are committed.

## 10. Installation

Requirements: **Python 3.14+** (the repo pins `3.14.3`). [uv](https://docs.astral.sh/uv/)
is recommended because the lockfile is included, but plain `pip` works too.

```bash
# Clone / copy the project folder, then from the project root:

# Option A - uv (recommended; installs runtime + dev deps from uv.lock)
uv sync

# Option B - pip
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest ruff mypy   # dev dependencies
```

The dataset key is a secret. The shared working copy already contains the
gitignored plaintext CSVs and the local key file (`data/.dataset.key`). On a
fresh machine, provide the key via the `DATASET_ENC_KEY` environment variable,
or regenerate the blobs with `python -m scripts.encrypt_dataset` (which also
creates the local key file).

## 11. Environment Variables

All settings come from the process environment (see `app/config.py`).
`.env.example` documents every variable. Copy it to `.env` as a reference and
export the values you change; the key must be set as an environment variable or
in the gitignored `data/.dataset.key` file. **Never include real credentials in
the repository.**

| Variable | Default | Purpose |
| --- | --- | --- |
| `MESSAGES_CSV_PATH` | `./messages.csv` | Input dataset (decrypted on demand) |
| `MANDATORY_DEMO_IDS_PATH` | `./mandatory_demo_ids.csv` | 15 demo-required IDs |
| `OUTPUTS_DIR` | `./outputs` | Generated JSON artifacts |
| `ENCRYPTED_DATA_DIR` | `./data` | Committed `*.enc` Fernet blobs |
| `DATASET_KEY_FILE` | `./data/.dataset.key` | Local gitignored Fernet key file |
| `DATASET_ENC_KEY` | *(empty)* | Fernet key secret (Render dashboard env var) |
| `EXPECTED_MESSAGE_COUNT` | `900` | Exact expected dataset size |
| `EXPECTED_MANDATORY_COUNT` | `15` | Exact expected mandatory-ID count |
| `LLM_ENABLED` | `false` | Offline mode; when `false` the pipeline is fully deterministic |
| `LLM_PROVIDER` | `openai` | Provider name (OpenAI-compatible) |
| `LLM_MODEL` | *(empty)* | Model identifier sent to the provider |
| `LLM_API_KEY` | *(empty)* | Secret key; never logged or serialized |
| `LLM_BASE_URL` | *(empty)* | Optional provider base URL override |
| `LLM_TIMEOUT_SECONDS` | `30` | Timeout for a single LLM request |
| `LLM_CONFIDENCE_THRESHOLD` | `0.75` | Rule confidence below which the LLM fallback is consulted |

## 12. Running Locally

```bash
# 1. Ensure the dataset is available (decrypts on demand if missing)
python -m scripts.run_pipeline

# 2. Start the dashboard (serves the sanitized API + UI)
uvicorn app.main:app --reload
# then open http://127.0.0.1:8000
```

Interactive API docs are available at `/docs` and `/redoc`.

## 13. Running the Pipeline

```bash
python -m scripts.run_pipeline
```

This loads and validates the dataset, decrypts the CSVs from `data/*.enc` on
demand if the plaintext files are missing, runs detection → masking →
classification → extraction over all 900 messages, and writes
`outputs/*.json`. The exit code is `0` on success and `1` when validation or
the leak scan fails.

## 14. Running Tests

```bash
python -m pytest            # 322 tests
python -m ruff check .      # lint
python -m mypy app tests scripts   # type checking
```

The test suite covers input validation, sensitive detection (including
security/leak tests), classification, extraction, output validation, pipeline
orchestration, mandatory demo coverage, the dataset cipher, and the API.

## 15. API Endpoints

All responses are sanitized Pydantic models; raw sensitive values never appear.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Single-page dashboard (HTML). |
| `GET` | `/health` | Liveness check: `{"status": "ok"}`. |
| `GET` | `/api/stats` | Aggregate statistics (counts only, no content). |
| `GET` | `/api/messages` | Paginated message list with `search`, `category`, `sensitive`, `limit`, `offset` filters. |
| `GET` | `/api/messages/{message_id}` | Full sanitized detail for one message. |
| `GET` | `/api/tasks` | Extracted items with `type`, `priority`, `date_from`, `date_to`, `limit`, `offset` filters. |
| `GET` | `/api/sensitive` | Sanitized sensitive detections (masked values only). |
| `GET` | `/api/demo/mandatory` | The 15 mandatory demo messages in dataset order. |
| `GET` | `/api/validation` | The validation report (summary, quality report, leak scan). |

Pagination defaults to `limit=100`; message/sensitive lists cap at 900 and the
tasks list at 2000.

## 16. Output Files

`python -m scripts.run_pipeline` writes these files to `outputs/` (gitignored,
regenerated on every run):

- **`classifications.json`** - one `ClassificationResult` per message
  (`message_id`, `category`, `confidence`, `reason`, `method`).
- **`tasks_events.json`** - one `ExtractionResult` per message (`message_id`,
  `items`, `method`, `reason`); `items` is a list of validated
  `ExtractedItem`s.
- **`sensitive_detections.json`** - one sanitized `MessageSensitiveResult` per
  message (`message_id`, `has_detection`, `detections`). Detections contain
  masked text and never a raw value.
- **`final_results.json`** - one sanitized `FinalMessageResult` per message
  (`message_id`, `timestamp`, `sender`, `safe_message`, `classification`,
  `security`, `extracted_items`). This artifact deliberately carries the
  masked message, never raw text.
- **`validation_report.json`** - the run summary, the consolidated
  `QualityReport`, the leak-scan result and the overall `validation_status`.

A `summary_statistics.json` is also written with aggregate counts only
(totals, items by type, classification methods, failures, leak check).

## 17. Deployment

The repository includes a `render.yaml` Blueprint. Render builds from the repo,
runs the pipeline, and serves the dashboard:

```yaml
services:
  - type: web
    name: message-intelligence
    runtime: python
    plan: free
    region: singapore
    buildCommand: pip install -r requirements.txt
    startCommand: python -m scripts.run_pipeline && uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATASET_ENC_KEY
        sync: false
      - key: PYTHON_VERSION
        value: 3.14.3
```

Steps:

1. Keep the GitHub repository **private** (the assignment forbids publishing
   the dataset; only encrypted blobs are committed).
2. Set the Fernet key as the `DATASET_ENC_KEY` secret in the Render dashboard
   (`sync: false` means you enter the value there; it is never committed).
3. Push `main`. Render installs `requirements.txt` (Python `3.14.3`), the start
   command decrypts the dataset, regenerates the artifacts, and serves the app.
   No external services or LLM calls are required.

## 18. Demo

The demo must show the 15 message IDs listed in `mandatory_demo_ids.csv`. The
"Mandatory Demo" section of the dashboard does this automatically:

- The IDs are loaded from the CSV at runtime - **never hardcoded** - and
  validated for count and uniqueness.
- Each mandatory message is shown with its classification (category,
  confidence, method), its extracted task/event, and a masked-content preview.
- Results are displayed in the **original dataset chronological order**.
- If any mandatory message were missing from the outputs, the service raises
  instead of showing a fabricated result.

The same data is available via `GET /api/demo/mandatory`.

## 19. Assumptions

The following assumptions reflect the implementation:

- The input is a UTF-8 (BOM) CSV with exactly the columns `message_id`,
  `timestamp`, `sender`, `message`, exactly 900 rows, unique IDs and strictly
  chronological timestamps (all enforced by validation).
- Each message is processed independently and yields exactly one
  classification and zero or more extracted items.
- The **message timestamp** is the reference for resolving relative dates; the
  system clock is never used.
- A message may yield several tasks only when it contains several explicit
  deadlines; otherwise exactly one item per matching rule.
- Editorial prefixes (`For today:`, `FYI:`, ...) are message prefixes, not date
  constraints, and are stripped from titles.
- A field is populated only when explicitly stated; everything else stays
  `null`/`unknown`.
- Fictional data is handled with the same privacy rigor as real data.
- The mandatory demo IDs come from `mandatory_demo_ids.csv`, never from code.
- The LLM fallback is optional; the default run is fully deterministic
  (rule-based) and offline.

## 20. Limitations

- **Synthetic dataset** - the 900 messages are fictional and hand-crafted, so
  real-world phrasing variety is not fully represented.
- **Regex limitations** - detection and extraction are regex/rule based.
  Novel phrasings may be missed; the conservative design avoids false positives
  at the cost of some under-detection.
- **LLM ambiguity** - when enabled, free-text LLM responses are parsed
  strictly; malformed or inconsistent responses are discarded and the
  deterministic result is kept.
- **Relative date ambiguity** - vague or conditional phrases (`could be
  Friday`, `sometime next week`) are intentionally not resolved into definite
  dates and stay `null`.
- **Contextual sensitivity detection** - only values that appear near explicit
  context or in exact formats are flagged (e.g. a bare phone number without any
  context is not detected). Detection is best-effort and not guaranteed to be
  exhaustive.

## 21. AI Tool Usage Declaration

- **AI coding tools** were used to develop this project: to design and
  implement the pipeline modules, write and run tests, review code, and produce
  documentation. All generated code was executed, tested and verified as part
  of the repository's test suite.
- **LLM APIs (pipeline feature):** the pipeline includes an optional LLM
  fallback (classifier/extractor). It is **disabled by default** (`LLM_ENABLED`
  defaults to `false`) and was **not used** during development or runs. When
  enabled it would receive only the **masked** message text - never raw
  sensitive values - via an OpenAI-compatible endpoint configured by the user.
- **Dataset privacy:** the assignment dataset was never uploaded to or
  processed by any external AI service. All pipeline runs were fully offline
  and deterministic.

## 22. Privacy/Security Notes

- **Masking policy** - sensitive values are detected and masked *before* any
  downstream stage: classification, extraction, the LLM, the API, the UI and
  the JSON artifacts only ever see masked text.
- **Internal-only raw values** - the raw matched value lives in a Pydantic
  `PrivateAttr` of the internal `SensitiveDetection`, which is excluded from
  `model_dump()`/`model_dump_json()`. Public models and API responses never
  contain a raw value; `MessageSensitiveResult` uses `extra="forbid"` so a
  stray raw field cannot sneak into an artifact.
- **Leak scan** - every run re-detects sensitive values in the original
  messages and verifies none appear in any artifact's JSON; findings record
  only message ID, artifact and sensitivity type.
- **Dataset at rest** - the plaintext CSVs and the Fernet key are gitignored;
  only encrypted `data/*.enc` blobs are committed. The key is supplied at
  build/start time via `DATASET_ENC_KEY` or the local `data/.dataset.key`.
- **Logging/reporting** - no pipeline stage logs or serializes raw sensitive
  values; `final_results.json` and the API serve only the masked message.
- **Secrets** - API keys and secrets are never committed, logged or included in
  responses.
