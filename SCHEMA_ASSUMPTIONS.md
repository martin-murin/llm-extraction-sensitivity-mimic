# Schema Assumptions

This code assumes that a credentialed user has loaded MIMIC-IV v3.1 and
MIMIC-IV-Note v2.2 into a PostgreSQL database. This repository does not include
MIMIC data or derived artifacts. A user must either load MIMIC into a database matching
the assumptions below, or adapt `src/db/connection.py` and `src/db/queries.py`.

## Database Engine and Configuration

- Engine: PostgreSQL, accessed through SQLAlchemy using `psycopg`.
- Connection setting: `MIMIC_PG_URI`.
- Optional model setting used elsewhere in the pipeline: `OPENAI_API_KEY` and
  `OPENAI_MODEL`.
- Configuration is read by `src/config.py` from environment variables or a local,
  untracked `.env` file. Use placeholders such as:

```bash
MIMIC_PG_URI=postgresql+psycopg://<USER>:<PASSWORD>@<HOST>:<PORT>/<DATABASE>
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
OPENAI_MODEL=<MODEL_SNAPSHOT>
```

## Table Discovery

`src/db/connection.py` does not hard-code schema names. It scans
`information_schema.tables` and `information_schema.columns`, ignores
`information_schema` and `pg_catalog`, and selects the first suitable table by
name. Schemas whose names start with `mimic` are preferred.

The resolved keys used by the code are:

| Code key | Candidate table names | Required for discovery | Required later by queries | Notes |
|---|---|---|---|---|
| `discharge_notes` | `discharge`, `discharge_note` | `hadm_id`, `text` | `hadm_id`, `subject_id`, `text`, PostgreSQL `ctid` | MIMIC-IV-Note v2.2 stock table is commonly `mimiciv_note.discharge`; this matches candidate `discharge`. If a local load renamed it to `discharge_note`, that is also accepted. |
| `radiology_notes` | `radiology`, `radiology_note` | `hadm_id`, `text` | only counted/explored if present | Optional. Not part of the paper's discharge-summary extraction pipeline. |
| `admissions` | `admissions` | none beyond table existence | `hadm_id` | Expected to be the MIMIC-IV hospital admissions table, usually `mimiciv_hosp.admissions`. |
| `diagnoses_icd` | `diagnoses_icd` | none beyond table existence | `hadm_id`, `icd_code`, `icd_version`, `seq_num` | Expected to be the MIMIC-IV hospital diagnosis-code table, usually `mimiciv_hosp.diagnoses_icd`. |
| `d_icd_diagnoses` | `d_icd_diagnoses` | none beyond table existence | `icd_code`, `icd_version`, `long_title` | Expected to be the MIMIC-IV ICD dictionary table, usually `mimiciv_hosp.d_icd_diagnoses`. |
| `patients` | `patients` | none beyond table existence | none in released query helpers except exploration/counting context | Expected to be the MIMIC-IV hospital patients table, usually `mimiciv_hosp.patients`. |

## Query-Level Column Assumptions

`src/db/queries.py` uses these columns and behaviors:

### Discharge notes table: `discharge` or `discharge_note`

Expected columns:

- `hadm_id`: admission identifier; used as the primary join key.
- `subject_id`: patient identifier; used when building split manifests.
- `text`: discharge summary text.

PostgreSQL-specific behavior:

- `pull_split_candidates()` uses `ORDER BY dn.ctid` to choose one deterministic
  row when multiple discharge-note rows share a `hadm_id`. `ctid` is a PostgreSQL
  physical tuple identifier, not a MIMIC column.

Functions reading this table:

- `count_notes()`
- `note_length_stats()`
- `join_cardinality()`
- `sample_redaction_excerpts()`
- `fetch_notes_by_hadm_ids()`
- `fetch_all_discharge_hadm_ids()`
- `fetch_all_discharge_notes()`
- `pull_split_candidates()`

Important caution: the source code includes `sample_redaction_excerpts()` and
`scripts/01_explore_mimic_notes.py`, which can print short note excerpts into a
local report. Those reports are derived from MIMIC and must not be published.

### Admissions table: `admissions`

Expected columns:

- `hadm_id`: used to verify discharge notes join to real admissions and to filter
  split candidates to admissions present in MIMIC-IV hosp.

Functions reading this table:

- `join_cardinality()`
- `pull_split_candidates()`

### Diagnoses table: `diagnoses_icd`

Expected columns:

- `hadm_id`
- `icd_code`
- `icd_version`
- `seq_num`

`seq_num` is cast to text and then parsed numerically when possible. Primary ICD
selection uses `seq_num = 1` when available; otherwise it orders by numeric
`seq_num`, then `icd_code`, then `icd_version`.

Functions reading this table:

- `top_primary_icds()`
- `top_any_position_icds()`
- `fetch_icd_codes_by_hadm_ids()`
- `fetch_primary_icd_by_hadm_ids()`
- `pull_split_candidates()`

### ICD dictionary table: `d_icd_diagnoses`

Expected columns:

- `icd_code`
- `icd_version`
- `long_title`

Functions reading this table:

- `top_primary_icds()`
- `top_any_position_icds()`

### Radiology table: `radiology` or `radiology_note`

Expected columns for discovery:

- `hadm_id`
- `text`

The table is optional and is not used by the paper extraction pipeline. It is
only included in exploration counts if present.

## Derived Local Artifacts

The code writes and later rereads restricted artifacts under local paths such as:

- `data/splits/*.csv`
- `data/raw_responses/<run_id>/*.json`
- `data/raw_responses/<run_id>/results.jsonl`
- `data/production/parquet/*.parquet`
- `data/optimization/*.jsonl`
- `data/cache/embeddings/**`
- `codex_outputs/*.md` and `codex_outputs/*.json`

These artifacts are generated from MIMIC and are not included in this repository.
Credentialed users regenerate them locally after loading MIMIC into a database
matching the assumptions above.
