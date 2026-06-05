from __future__ import annotations

# Release documentation:
# Summarizes final production post-run QA.
#
# Reads: data/splits, codex_outputs/40_postrun_qa.md, codex_outputs/40_parse_failures.md, codex_outputs/40_api_errors.md, data/raw_responses.
# Writes: data/splits, codex_outputs/40_postrun_qa.md, codex_outputs/40_parse_failures.md, codex_outputs/40_api_errors.md, data/raw_responses.
# Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
# Usage: `python scripts/40_postrun_qa.py` unless the script's argparse help says otherwise.

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.connection import get_engine
from src.db.icd_utils import icd10_chapter_from_code
from src.db.queries import fetch_notes_by_hadm_ids, fetch_primary_icd_by_hadm_ids
from src.llm.extractor import extract_content_from_raw_response
from src.schema.fields import LLMNoteFeatures

EXPECTED_ATTEMPTS = 331_793
DEFAULT_RUN_ID = "production_v1"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body: list[str] = []
    for row in rows:
        rendered: list[str] = []
        for col in columns:
            value = row.get(col, "")
            text = str(value)
            rendered.append(text.replace("|", "\\|"))
        body.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *body])


def _pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100:.2f}%"


def _bucket_input_tokens(tokens: int) -> str:
    if tokens <= 4_000:
        return "<=4k"
    if tokens <= 6_000:
        return "4k-6k"
    if tokens <= 8_000:
        return "6k-8k"
    if tokens <= 10_000:
        return "8k-10k"
    if tokens <= 12_000:
        return "10k-12k"
    return ">12k"


def _structure_signature(parse_error: str, raw_text: str) -> str:
    lower = parse_error.lower()
    if not raw_text:
        if "no response content" in lower:
            return "empty_response"
        return "no_raw_payload"
    stripped = raw_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return "json_like"
    if stripped.startswith("{") and not stripped.endswith("}"):
        return "truncated_json_like"
    if "```json" in stripped.lower():
        return "markdown_fenced_json"
    return "non_json_text"


def _error_taxonomy(parse_error: str) -> str:
    lower = parse_error.lower()
    if lower.startswith("api_error:"):
        if "ratelimit" in lower or "429" in lower:
            return "api_error.rate_limit"
        if "timeout" in lower:
            return "api_error.timeout"
        if "connection" in lower:
            return "api_error.connection"
        if "budget" in lower:
            return "api_error.budget"
        return "api_error.other"
    if "jsondecodeerror" in lower:
        if "unterminated" in lower:
            return "parse.json_unterminated"
        if "expecting ',' delimiter" in lower:
            return "parse.json_missing_delimiter"
        return "parse.json_decode"
    if "validationerror" in lower:
        if "dominant_admission_reason" in lower:
            return "parse.validation_dominant_not_in_tags"
        if "admission_reason_tags" in lower and "non-empty" in lower:
            return "parse.validation_empty_tags"
        if "literal" in lower:
            return "parse.validation_literal"
        return "parse.validation_other"
    if "no response content" in lower:
        return "parse.empty_content"
    return "parse.other"


@dataclass
class FailureRow:
    hadm_id: int
    parse_error: str
    taxonomy: str
    structure: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    raw_excerpt: str
    chapter: str
    note_char_len: int | None


def _load_split_context() -> dict[int, tuple[str, int | None]]:
    context: dict[int, tuple[str, int | None]] = {}
    split_dir = Path("data/splits")
    for path in sorted(split_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path, usecols=["hadm_id", "chapter", "note_char_len"])
        except Exception:
            continue
        for _, row in df.iterrows():
            hadm_id = int(row["hadm_id"])
            if hadm_id in context:
                continue
            chapter = str(row.get("chapter", "")).strip() or "unknown"
            length_raw = row.get("note_char_len")
            note_len: int | None = None if pd.isna(length_raw) else int(length_raw)
            context[hadm_id] = (chapter, note_len)
    return context


def _load_detailed_payload(run_dir: Path, hadm_id: int) -> dict[str, Any]:
    path = run_dir / f"{hadm_id}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def _enrich_failures_with_db(failures: list[FailureRow]) -> None:
    if not failures:
        return
    hadm_ids = [row.hadm_id for row in failures]
    try:
        engine = get_engine()
        primary = fetch_primary_icd_by_hadm_ids(engine, hadm_ids)
        notes = fetch_notes_by_hadm_ids(engine, hadm_ids)
    except Exception:
        return

    chapter_by_hadm: dict[int, str] = {}
    for hadm_id, value in primary.items():
        code, version = value
        chapter_by_hadm[hadm_id] = icd10_chapter_from_code(code, version)

    for row in failures:
        chapter = chapter_by_hadm.get(row.hadm_id)
        if chapter:
            row.chapter = chapter
        note = notes.get(row.hadm_id)
        if note is not None:
            row.note_char_len = len(note)


def _failure_summary_rows(rows: list[FailureRow]) -> list[dict[str, Any]]:
    grouped = Counter(row.taxonomy for row in rows)
    total = len(rows)
    output: list[dict[str, Any]] = []
    for taxonomy, count in grouped.most_common():
        output.append({"taxonomy": taxonomy, "count": count, "pct": _pct(count, total)})
    return output


def _cluster_rows(rows: list[FailureRow], field: str) -> list[dict[str, Any]]:
    values: Counter[str] = Counter()
    for row in rows:
        value = getattr(row, field)
        values[str(value)] += 1
    total = len(rows)
    output: list[dict[str, Any]] = []
    for key, count in values.most_common():
        output.append({"cluster": key, "count": count, "pct": _pct(count, total)})
    return output


def _note_length_clusters(rows: list[FailureRow]) -> list[dict[str, Any]]:
    b: Counter[str] = Counter()
    for row in rows:
        if row.note_char_len is None:
            b["unknown"] += 1
            continue
        note_len = row.note_char_len
        if note_len <= 2_000:
            key = "<=2k chars"
        elif note_len <= 5_000:
            key = "2k-5k chars"
        elif note_len <= 10_000:
            key = "5k-10k chars"
        elif note_len <= 20_000:
            key = "10k-20k chars"
        else:
            key = ">20k chars"
        b[key] += 1
    total = len(rows)
    out: list[dict[str, Any]] = []
    for key, count in b.most_common():
        out.append({"cluster": key, "count": count, "pct": _pct(count, total)})
    return out


def _write_failure_report(path: Path, title: str, rows: list[FailureRow]) -> None:
    taxonomy = _failure_summary_rows(rows)
    by_chapter = _cluster_rows(rows, "chapter")
    by_input_bucket = Counter(_bucket_input_tokens(r.input_tokens) for r in rows)
    by_input_rows = [
        {"cluster": key, "count": count, "pct": _pct(count, len(rows))}
        for key, count in by_input_bucket.most_common()
    ]
    by_structure = _cluster_rows(rows, "structure")
    by_note_len = _note_length_clusters(rows)

    examples: list[dict[str, Any]] = []
    for row in rows[:30]:
        examples.append(
            {
                "hadm_id": row.hadm_id,
                "taxonomy": row.taxonomy,
                "chapter": row.chapter,
                "input_tokens": row.input_tokens,
                "parse_error": row.parse_error[:140],
                "raw_excerpt": row.raw_excerpt[:160].replace("\n", " "),
            }
        )

    lines = [
        f"# {title}",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        f"- Total rows: **{len(rows):,}**",
        "",
        "## Error taxonomy",
        _markdown_table(taxonomy, ["taxonomy", "count", "pct"]),
        "",
        "## Clustering: ICD chapter",
        _markdown_table(by_chapter, ["cluster", "count", "pct"]),
        "",
        "## Clustering: note length (input token bucket)",
        _markdown_table(by_input_rows, ["cluster", "count", "pct"]),
        "",
        "## Clustering: note length (DB char length when available)",
        _markdown_table(by_note_len, ["cluster", "count", "pct"]),
        "",
        "## Clustering: response structure",
        _markdown_table(by_structure, ["cluster", "count", "pct"]),
        "",
        "## Example rows",
        _markdown_table(
            examples,
            ["hadm_id", "taxonomy", "chapter", "input_tokens", "parse_error", "raw_excerpt"],
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-run QA for production extraction results.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--expected-attempts", type=int, default=EXPECTED_ATTEMPTS)
    parser.add_argument("--output", default="codex_outputs/40_postrun_qa.md")
    parser.add_argument("--parse-failures-output", default="codex_outputs/40_parse_failures.md")
    parser.add_argument("--api-errors-output", default="codex_outputs/40_api_errors.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path("data/raw_responses") / args.run_id
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    split_context = _load_split_context()

    attempted = 0
    parse_ok = 0
    parse_fail = 0
    api_error = 0
    schema_invalid = 0
    hadm_seen: set[int] = set()
    duplicate_hadm: set[int] = set()

    failures_parse: list[FailureRow] = []
    failures_api: list[FailureRow] = []

    parse_input_tokens: list[int] = []

    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            attempted += 1
            payload = json.loads(line)
            hadm_id = int(payload.get("hadm_id", 0) or 0)
            if hadm_id in hadm_seen:
                duplicate_hadm.add(hadm_id)
            hadm_seen.add(hadm_id)

            ok = bool(payload.get("parse_ok", False))
            in_tokens = int(payload.get("input_tokens", 0) or 0)
            out_tokens = int(payload.get("output_tokens", 0) or 0)
            latency = float(payload.get("latency_seconds", 0.0) or 0.0)
            parse_input_tokens.append(in_tokens)

            chapter, note_len = split_context.get(hadm_id, ("unknown", None))

            if ok:
                parse_ok += 1
                features_json = payload.get("features_json")
                if not isinstance(features_json, dict):
                    schema_invalid += 1
                    continue
                try:
                    LLMNoteFeatures.model_validate(features_json)
                except ValidationError:
                    schema_invalid += 1
                continue

            parse_error_text = str(payload.get("parse_error", ""))
            detailed = _load_detailed_payload(run_dir, hadm_id)
            raw_response = detailed.get("raw_response")
            raw_text = ""
            if isinstance(raw_response, dict):
                raw_text = extract_content_from_raw_response(raw_response)

            row = FailureRow(
                hadm_id=hadm_id,
                parse_error=parse_error_text,
                taxonomy=_error_taxonomy(parse_error_text),
                structure=_structure_signature(parse_error_text, raw_text),
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_seconds=latency,
                raw_excerpt=raw_text[:800],
                chapter=chapter,
                note_char_len=note_len,
            )

            if parse_error_text.lower().startswith("api_error:"):
                api_error += 1
                failures_api.append(row)
            else:
                parse_fail += 1
                failures_parse.append(row)

    _enrich_failures_with_db(failures_parse)
    _enrich_failures_with_db(failures_api)

    _write_failure_report(
        Path(args.parse_failures_output),
        "Production Parse Failures",
        failures_parse,
    )
    _write_failure_report(Path(args.api_errors_output), "Production API Errors", failures_api)

    p50_tokens = float(np.percentile(np.array(parse_input_tokens, dtype=np.float64), 50))
    p95_tokens = float(np.percentile(np.array(parse_input_tokens, dtype=np.float64), 95))

    checks: list[dict[str, Any]] = [
        {
            "check": "attempted_total",
            "expected": args.expected_attempts,
            "observed": attempted,
            "status": "PASS" if attempted == args.expected_attempts else "FAIL",
        },
        {
            "check": "unique_hadm_ids",
            "expected": attempted,
            "observed": len(hadm_seen),
            "status": "PASS" if len(hadm_seen) == attempted else "FAIL",
        },
        {
            "check": "duplicate_hadm_ids",
            "expected": 0,
            "observed": len(duplicate_hadm),
            "status": "PASS" if not duplicate_hadm else "FAIL",
        },
        {
            "check": "schema_invalid_rows",
            "expected": 0,
            "observed": schema_invalid,
            "status": "PASS" if schema_invalid == 0 else "FAIL",
        },
        {
            "check": "failure_partition",
            "expected": attempted,
            "observed": parse_ok + parse_fail + api_error,
            "status": "PASS" if (parse_ok + parse_fail + api_error) == attempted else "FAIL",
        },
    ]
    banner = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"

    summary_rows: list[dict[str, Any]] = [
        {"metric": "attempted", "value": attempted},
        {"metric": "parse_ok", "value": parse_ok},
        {"metric": "parse_fail", "value": parse_fail},
        {"metric": "api_error", "value": api_error},
        {"metric": "schema_invalid", "value": schema_invalid},
        {"metric": "duplicate_hadm_ids", "value": len(duplicate_hadm)},
        {"metric": "success_rate", "value": _pct(parse_ok, attempted)},
        {"metric": "failure_rate", "value": _pct(parse_fail + api_error, attempted)},
        {"metric": "p50_input_tokens", "value": f"{p50_tokens:.1f}"},
        {"metric": "p95_input_tokens", "value": f"{p95_tokens:.1f}"},
    ]

    lines = [
        "# Production Post-run QA",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        f"## Banner: {banner}",
        "",
        "## Core checks",
        _markdown_table(checks, ["check", "expected", "observed", "status"]),
        "",
        "## Run summary",
        _markdown_table(summary_rows, ["metric", "value"]),
        "",
        "## Failure artifacts",
        f"- Parse failures report: `{args.parse_failures_output}`",
        f"- API errors report: `{args.api_errors_output}`",
        "",
        "## Notes",
        (
            "- ICD chapter and note-length clustering are enriched from DB when available, "
            "with CSV fallback context where possible."
        ),
        (
            "- Response-structure clustering is derived from `raw_response` payloads in "
            "per-note JSON files."
        ),
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Banner: {banner}")
    print(
        "attempted="
        f"{attempted} "
        f"parse_ok={parse_ok} "
        f"parse_fail={parse_fail} "
        f"api_error={api_error}"
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {args.parse_failures_output}")
    print(f"Wrote {args.api_errors_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
