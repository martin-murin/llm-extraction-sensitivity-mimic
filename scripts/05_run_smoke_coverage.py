"""
Runs smoke-sample extraction and coverage checks.

Reads: configs/optimization.yaml.
Writes: local reports/artifacts determined by CLI defaults.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/05_run_smoke_coverage.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from src import config
from src.db.connection import discover_schemas, get_engine
from src.db.queries import (
    fetch_all_discharge_hadm_ids,
    fetch_all_discharge_notes,
    fetch_notes_by_hadm_ids,
)
from src.llm.batch_runner import BatchSummary, run_batch
from src.llm.client import LLMClient
from src.llm.extractor import extract_content_from_raw_response
from src.schema.vocabulary import ADMISSION_REASON_TAGS, CHAPTER_TO_PLAUSIBLE_TAGS

TOTAL_NOTES_PROJECTION = 331_793

TRISTATE_FIELDS = [
    "aki_present",
    "cardiac_rehab_referred",
    "cognitive_impairment",
    "discharge_delayed_reason",
    "dnr_dni_documented",
    "fall_risk_documented",
    "financial_hardship",
    "goals_of_care_flag",
    "home_health_ordered",
    "hospital_acquired_complication",
    "infection_as_trigger",
    "lives_alone",
    "palliative_care_consult",
    "shock_present",
    "social_support_absent",
    "substance_use_active",
    "unresolved_diagnosis_at_discharge",
]

ENUM_FIELDS = [
    "functional_status",
    "mental_status",
    "discharge_condition_category",
]

COUNT_FIELDS = [
    "new_meds_started_count",
    "meds_stopped_count",
]

logger = logging.getLogger("scripts.05_run_smoke_coverage")


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if value is None:
        return ""
    return str(value)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        rendered: list[str] = []
        for column in columns:
            value = _format_number(row.get(column, ""))
            rendered.append(value.replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *lines])


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _quote_qualified_name(qualified_name: str) -> str:
    schema, table = qualified_name.split(".", maxsplit=1)
    escaped_schema = schema.replace('"', '""')
    escaped_table = table.replace('"', '""')
    return f'"{escaped_schema}"."{escaped_table}"'


def _fetch_icd_descriptions(
    engine: Engine,
    split_frame: pd.DataFrame,
) -> dict[tuple[str, str], str]:
    schemas = discover_schemas(engine)
    dictionary = schemas["d_icd_diagnoses"]
    if not dictionary:
        return {}

    code_series = split_frame["primary_icd_code"].astype(str)
    unique_codes = sorted(code_series.unique().tolist())
    if not unique_codes:
        return {}

    query = text(
        f"""
        SELECT
            icd_code::text AS icd_code,
            CAST(icd_version AS text) AS icd_version,
            COALESCE(long_title, '') AS long_title
        FROM {_quote_qualified_name(dictionary)}
        WHERE icd_code IN :codes
        """
    ).bindparams(bindparam("codes", expanding=True))

    rows = pd.read_sql_query(query, engine, params={"codes": unique_codes})
    valid_pairs = {
        (str(code), str(version))
        for code, version in zip(
            split_frame["primary_icd_code"].astype(str),
            split_frame["primary_icd_version"].astype(str),
            strict=False,
        )
    }
    descriptions: dict[tuple[str, str], str] = {}
    for _, row in rows.iterrows():
        key = (str(row["icd_code"]), str(row["icd_version"]))
        if key in valid_pairs and key not in descriptions:
            descriptions[key] = str(row["long_title"])
    return descriptions


def _load_result_payloads(output_dir: Path, hadm_ids: list[int]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for hadm_id in hadm_ids:
        path = output_dir / f"{hadm_id}.json"
        if not path.exists():
            continue
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payloads.append(loaded)
    return payloads


def _load_run_metadata(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "run_metadata.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _distribution_rows(series: pd.Series) -> list[dict[str, Any]]:
    if series.empty:
        return []
    counts = series.value_counts(dropna=False).sort_values(ascending=False)
    total = int(counts.sum())
    rows = []
    for value, count in counts.items():
        rows.append(
            {
                "value": str(value),
                "count": int(count),
                "pct": _percent(int(count) / total),
            }
        )
    return rows


def _count_stats(values: pd.Series) -> dict[str, Any]:
    null_rate = float(values.isna().mean()) if len(values) else 0.0
    non_null = pd.to_numeric(values.dropna(), errors="coerce").dropna()
    if non_null.empty:
        return {
            "null_rate": _percent(null_rate),
            "min": "",
            "median": "",
            "max": "",
            "p95": "",
        }
    return {
        "null_rate": _percent(null_rate),
        "min": int(non_null.min()),
        "median": float(non_null.median()),
        "max": int(non_null.max()),
        "p95": float(np.percentile(non_null.to_numpy(dtype=np.float64), 95)),
    }


def _collect_preflight(engine: Engine, split_path: Path, sample_hadm_id: int) -> list[str]:
    lines: list[str] = []
    lines.append("### Pre-flight checks")

    with engine.connect() as conn:
        value = int(conn.execute(text("SELECT 1")).scalar_one())
    lines.append(f"- DB connectivity: PASS (`SELECT 1` -> {value})")

    if not split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_path}")
    lines.append(f"- Split file: PASS ({split_path})")

    sample_note = fetch_notes_by_hadm_ids(engine, [sample_hadm_id])
    if sample_hadm_id not in sample_note:
        raise RuntimeError(
            f"Failed pre-flight note fetch: hadm_id {sample_hadm_id} not found in discharge notes."
        )
    lines.append(f"- Note fetch test: PASS (hadm_id {sample_hadm_id})")
    return lines


def _collect_preflight_full(engine: Engine) -> list[str]:
    lines: list[str] = []
    lines.append("### Pre-flight checks")

    with engine.connect() as conn:
        value = int(conn.execute(text("SELECT 1")).scalar_one())
    lines.append(f"- DB connectivity: PASS (`SELECT 1` -> {value})")

    hadm_ids = fetch_all_discharge_hadm_ids(engine)
    if not hadm_ids:
        raise RuntimeError("No discharge notes found for full split pre-flight.")
    lines.append(f"- Full split source query: PASS (`n_hadm_ids={len(hadm_ids):,}`)")

    sample_hadm_id = hadm_ids[0]
    sample_note = fetch_notes_by_hadm_ids(engine, [sample_hadm_id])
    if sample_hadm_id not in sample_note:
        raise RuntimeError(
            f"Failed full pre-flight note fetch: hadm_id {sample_hadm_id} not found."
        )
    lines.append(f"- Note fetch test: PASS (hadm_id {sample_hadm_id})")
    return lines


def _build_report(
    *,
    output_path: Path,
    summary: BatchSummary,
    split_frame: pd.DataFrame,
    result_payloads: list[dict[str, Any]],
    run_id: str,
    split_name: str,
    variant: str,
    include_reasoning: bool,
    budget_cap_usd: float,
    preflight_lines: list[str],
    engine: Engine,
    run_metadata: dict[str, Any],
) -> None:
    parsed_payloads = [
        payload for payload in result_payloads if bool(payload.get("parse_ok", False))
    ]
    failed_payloads = [
        payload for payload in result_payloads if not bool(payload.get("parse_ok", False))
    ]
    n_attempted = summary.n_successful_parse + summary.n_failed_parse + summary.n_api_error

    input_cost = (summary.total_input_tokens / 1_000_000) * config.INPUT_PRICE_PER_MILLION_USD
    output_cost = (summary.total_output_tokens / 1_000_000) * config.OUTPUT_PRICE_PER_MILLION_USD
    total_cost = input_cost + output_cost
    average_per_note_cost = (total_cost / n_attempted) if n_attempted else 0.0
    projected_full_run_cost = average_per_note_cost * TOTAL_NOTES_PROJECTION

    split_by_hadm = split_frame.set_index("hadm_id", drop=False)
    parsed_rows: list[dict[str, Any]] = []
    for payload in parsed_payloads:
        hadm_id = int(payload["hadm_id"])
        features_json = payload.get("features_json")
        if not isinstance(features_json, dict):
            continue
        parsed_rows.append({"hadm_id": hadm_id, **features_json})

    parsed_df = pd.DataFrame(parsed_rows)
    n_parsed = len(parsed_df)

    admission_rows: list[dict[str, Any]] = []
    if n_parsed:
        tags_series = parsed_df["admission_reason_tags"]
        tag_lengths = tags_series.apply(lambda value: len(value) if isinstance(value, list) else 0)
        sole_other = tags_series.apply(lambda value: isinstance(value, list) and value == ["other"])
        contains_other = tags_series.apply(
            lambda value: isinstance(value, list) and ("other" in value)
        )
        admission_rows = [
            {"metric": "min_list_len", "value": int(tag_lengths.min())},
            {"metric": "median_list_len", "value": float(tag_lengths.median())},
            {"metric": "max_list_len", "value": int(tag_lengths.max())},
            {"metric": "pct_exactly_1_tag", "value": _percent(float((tag_lengths == 1).mean()))},
            {
                "metric": "pct_with_2_or_more_tags",
                "value": _percent(float((tag_lengths >= 2).mean())),
            },
            {"metric": "pct_sole_other", "value": _percent(float(sole_other.mean()))},
            {"metric": "pct_other_anywhere", "value": _percent(float(contains_other.mean()))},
        ]

    dominant_distribution_rows: list[dict[str, Any]] = []
    if n_parsed:
        dominant_distribution_rows = _distribution_rows(parsed_df["dominant_admission_reason"])

    tristate_rows: list[dict[str, Any]] = []
    if n_parsed:
        for field_name in sorted(TRISTATE_FIELDS):
            series = parsed_df[field_name].astype(str)
            tristate_rows.append(
                {
                    "field": field_name,
                    "yes_pct": _percent(float((series == "yes").mean())),
                    "no_pct": _percent(float((series == "no").mean())),
                    "not_documented_pct": _percent(float((series == "not_documented").mean())),
                }
            )

    enum_section_tables: list[tuple[str, str]] = []
    if n_parsed:
        for field_name in ENUM_FIELDS:
            rows = _distribution_rows(parsed_df[field_name].astype(str))
            enum_section_tables.append(
                (
                    field_name,
                    _markdown_table(rows, ["value", "count", "pct"]),
                )
            )

    count_rows: list[dict[str, Any]] = []
    if n_parsed:
        for field_name in COUNT_FIELDS:
            stats = _count_stats(parsed_df[field_name])
            count_rows.append({"field": field_name, **stats})

    primary_diagnosis_len_rows: list[dict[str, Any]] = []
    if n_parsed:
        lengths = parsed_df["primary_diagnosis_text"].astype(str).str.len().astype(np.int64)
        primary_diagnosis_len_rows = [
            {"metric": "min", "value": int(lengths.min())},
            {"metric": "median", "value": float(np.median(lengths.to_numpy(dtype=np.float64)))},
            {
                "metric": "p95",
                "value": float(np.percentile(lengths.to_numpy(dtype=np.float64), 95)),
            },
            {"metric": "max", "value": int(lengths.max())},
        ]

    coverage_rate = 0.0
    dominant_counts = {tag: 0 for tag in ADMISSION_REASON_TAGS}
    sole_other_examples: list[dict[str, Any]] = []
    if n_parsed:
        successful_non_other = 0
        for _, row in parsed_df.iterrows():
            tags = row["admission_reason_tags"]
            dominant = str(row["dominant_admission_reason"])
            if dominant in dominant_counts:
                dominant_counts[dominant] += 1

            sole_other = isinstance(tags, list) and tags == ["other"]
            if dominant != "other" and not sole_other:
                successful_non_other += 1

            if sole_other and len(sole_other_examples) < 10:
                sole_other_examples.append(
                    {
                        "hadm_id": int(row["hadm_id"]),
                        "primary_diagnosis_text": str(row["primary_diagnosis_text"]),
                    }
                )
        coverage_rate = successful_non_other / n_parsed

    per_tag_rows = [
        {"tag": tag, "count": dominant_counts.get(tag, 0)}
        for tag in ADMISSION_REASON_TAGS
    ]

    chapter_rows: list[dict[str, Any]] = []
    parsed_dominant_by_hadm: dict[int, str] = {}
    if n_parsed:
        parsed_dominant_by_hadm = {
            int(row["hadm_id"]): str(row["dominant_admission_reason"])
            for _, row in parsed_df[["hadm_id", "dominant_admission_reason"]].iterrows()
        }

    for chapter, chapter_frame in split_frame.groupby("chapter", sort=True):
        hadm_ids = chapter_frame["hadm_id"].astype(int).tolist()
        plausible = CHAPTER_TO_PLAUSIBLE_TAGS.get(str(chapter), set())
        if not plausible:
            chapter_rows.append(
                {
                    "chapter": chapter,
                    "n_notes": len(hadm_ids),
                    "plausible_tags": "N/A",
                    "concordant_fraction": "N/A",
                }
            )
            continue

        concordant = 0
        for hadm_id in hadm_ids:
            dominant = parsed_dominant_by_hadm.get(hadm_id)
            if dominant in plausible:
                concordant += 1
        chapter_rows.append(
            {
                "chapter": chapter,
                "n_notes": len(hadm_ids),
                "plausible_tags": len(plausible),
                "concordant_fraction": _percent(concordant / len(hadm_ids) if hadm_ids else 0.0),
            }
        )

    icd_descriptions = _fetch_icd_descriptions(engine, split_frame)
    sample_sections: list[str] = []
    sample_hadm_ids = sorted(parsed_dominant_by_hadm.keys())[:5]
    for hadm_id in sample_hadm_ids:
        split_row = split_by_hadm.loc[hadm_id]
        code = str(split_row["primary_icd_code"])
        version = str(split_row["primary_icd_version"])
        description = icd_descriptions.get((code, version), "")
        feature_row = parsed_df[parsed_df["hadm_id"] == hadm_id].iloc[0].to_dict()
        sample_sections.extend(
            [
                f"### hadm_id {hadm_id}",
                (
                    f"- primary_icd_code: `{code}` (version {version})  "
                    f"- primary_icd_description: {description or 'N/A'}"
                ),
                "```json",
                json.dumps(feature_row, ensure_ascii=True, indent=2),
                "```",
                "",
            ]
        )

    parse_failure_rows: list[dict[str, Any]] = []
    for payload in failed_payloads[:10]:
        raw_response = payload.get("raw_response")
        raw_text = ""
        if isinstance(raw_response, dict):
            raw_text = extract_content_from_raw_response(raw_response)
        parse_failure_rows.append(
            {
                "hadm_id": int(payload.get("hadm_id", 0) or 0),
                "parse_error": str(payload.get("parse_error", "")),
                "raw_response_excerpt": raw_text[:500],
            }
        )

    run_metadata_rows = [
        {"field": "timestamp_utc", "value": datetime.now(tz=UTC).isoformat()},
        {"field": "run_id", "value": run_id},
        {"field": "split", "value": split_name},
        {"field": "variant", "value": variant},
        {"field": "include_reasoning", "value": include_reasoning},
        {"field": "n_attempted", "value": n_attempted},
        {"field": "n_successful_parse", "value": summary.n_successful_parse},
        {"field": "n_failed_parse", "value": summary.n_failed_parse},
        {"field": "n_api_error", "value": summary.n_api_error},
        {"field": "total_cost_usd", "value": total_cost},
        {"field": "budget_cap_usd", "value": budget_cap_usd},
        {"field": "elapsed_seconds", "value": summary.elapsed_seconds},
        {"field": "median_latency", "value": summary.median_latency_seconds},
        {"field": "p95_latency", "value": summary.p95_latency_seconds},
        {
            "field": "max_concurrency_requested",
            "value": run_metadata.get("max_concurrency_requested", ""),
        },
        {
            "field": "client_semaphore_limit",
            "value": run_metadata.get("client_semaphore_limit", ""),
        },
        {"field": "max_retries", "value": run_metadata.get("max_retries", "")},
        {"field": "retry_policy", "value": run_metadata.get("retry_policy", "")},
    ]

    cost_rows = [
        {"metric": "total_input_tokens", "value": summary.total_input_tokens},
        {"metric": "total_output_tokens", "value": summary.total_output_tokens},
        {"metric": "input_cost_usd", "value": input_cost},
        {"metric": "output_cost_usd", "value": output_cost},
        {"metric": "total_cost_usd", "value": total_cost},
        {"metric": "projected_total_cost_usd_full_331793", "value": projected_full_run_cost},
    ]

    lines: list[str] = [
        "# Coverage Report",
        "",
        "## Run metadata",
        _markdown_table(run_metadata_rows, ["field", "value"]),
        "",
        "## Parse failure audit",
        _markdown_table(parse_failure_rows, ["hadm_id", "parse_error", "raw_response_excerpt"]),
        "",
        "## Cost breakdown",
        _markdown_table(cost_rows, ["metric", "value"]),
        "",
        "## Field-by-field coverage analysis",
        "### admission_reason_tags",
        _markdown_table(admission_rows, ["metric", "value"]),
        "",
        "### dominant_admission_reason distribution",
        _markdown_table(dominant_distribution_rows, ["value", "count", "pct"]),
        "",
        "### TriState fields",
        _markdown_table(tristate_rows, ["field", "yes_pct", "no_pct", "not_documented_pct"]),
        "",
        "### Domain enum fields",
    ]

    for field_name, table in enum_section_tables:
        lines.extend([f"#### {field_name}", table, ""])

    lines.extend(
        [
            "### Count fields",
            _markdown_table(count_rows, ["field", "null_rate", "min", "median", "max", "p95"]),
            "",
            "### primary_diagnosis_text length",
            _markdown_table(primary_diagnosis_len_rows, ["metric", "value"]),
            "",
            "## Vocabulary coverage assessment",
            f"- coverage_rate: {_percent(coverage_rate)} (target: >= 90%)",
            "",
            _markdown_table(per_tag_rows, ["tag", "count"]),
            "",
            "### Notes with admission_reason_tags == ['other']",
            _markdown_table(sole_other_examples, ["hadm_id", "primary_diagnosis_text"]),
            "",
            "## Base-rate comparison against ICD chapter distribution",
            _markdown_table(
                chapter_rows,
                ["chapter", "n_notes", "plausible_tags", "concordant_fraction"],
            ),
            "",
            "## Sample extractions for review",
            *sample_sections,
            "## Questions and assumptions",
            (
                "- Concordance mapping uses `CHAPTER_TO_PLAUSIBLE_TAGS` as a heuristic "
                "diagnostic, not a ground-truth label."
            ),
            (
                "- Chapter rows with `plausible_tags = N/A` indicate intentionally-unmapped "
                "chapters "
                "and are excluded from concordance interpretation."
            ),
            "- Cost projection scales observed smoke-run average per-note cost to 331,793 notes.",
            "",
            "## Pre-flight logs",
            *preflight_lines,
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_full_report(
    *,
    output_path: Path,
    summary: BatchSummary,
    run_id: str,
    variant: str,
    include_reasoning: bool,
    budget_cap_usd: float,
    preflight_lines: list[str],
    run_metadata: dict[str, Any],
) -> None:
    n_attempted = summary.n_successful_parse + summary.n_failed_parse + summary.n_api_error
    input_cost = (summary.total_input_tokens / 1_000_000) * config.INPUT_PRICE_PER_MILLION_USD
    output_cost = (summary.total_output_tokens / 1_000_000) * config.OUTPUT_PRICE_PER_MILLION_USD
    total_cost = input_cost + output_cost
    avg_per_note = (total_cost / n_attempted) if n_attempted else 0.0

    run_metadata_rows = [
        {"field": "timestamp_utc", "value": datetime.now(tz=UTC).isoformat()},
        {"field": "run_id", "value": run_id},
        {"field": "split", "value": "full"},
        {"field": "variant", "value": variant},
        {"field": "include_reasoning", "value": include_reasoning},
        {"field": "n_attempted", "value": n_attempted},
        {"field": "n_successful_parse", "value": summary.n_successful_parse},
        {"field": "n_failed_parse", "value": summary.n_failed_parse},
        {"field": "n_api_error", "value": summary.n_api_error},
        {"field": "total_cost_usd", "value": total_cost},
        {"field": "avg_cost_per_note_usd", "value": avg_per_note},
        {"field": "budget_cap_usd", "value": budget_cap_usd},
        {"field": "elapsed_seconds", "value": summary.elapsed_seconds},
        {"field": "median_latency", "value": summary.median_latency_seconds},
        {"field": "p95_latency", "value": summary.p95_latency_seconds},
        {
            "field": "max_concurrency_requested",
            "value": run_metadata.get("max_concurrency_requested", ""),
        },
        {
            "field": "client_semaphore_limit",
            "value": run_metadata.get("client_semaphore_limit", ""),
        },
        {"field": "max_retries", "value": run_metadata.get("max_retries", "")},
        {"field": "retry_policy", "value": run_metadata.get("retry_policy", "")},
    ]

    lines = [
        "# Coverage Report (Full Split)",
        "",
        "## Run metadata",
        _markdown_table(run_metadata_rows, ["field", "value"]),
        "",
        "## Token and cost totals",
        _markdown_table(
            [
                {"metric": "total_input_tokens", "value": summary.total_input_tokens},
                {"metric": "total_output_tokens", "value": summary.total_output_tokens},
                {"metric": "input_cost_usd", "value": input_cost},
                {"metric": "output_cost_usd", "value": output_cost},
                {"metric": "total_cost_usd", "value": total_cost},
            ],
            ["metric", "value"],
        ),
        "",
        "## Notes",
        "- Full split mode emits a lightweight run report and skips per-field chapter diagnostics.",
        "",
        "## Pre-flight logs",
        *preflight_lines,
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run coverage extraction on a configured split.")
    parser.add_argument("--run-id", default="coverage_v1")
    parser.add_argument(
        "--split",
        choices=[
            "refinement",
            "holdout",
            "smoke",
            "methodology_1k",
            "methodology_5k",
            "methodology_5k_audit_500",
            "full",
        ],
        default="smoke",
    )
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--variant", default="a")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include-reasoning", dest="include_reasoning", action="store_true")
    group.add_argument("--no-reasoning", dest="include_reasoning", action="store_false")
    group.add_argument("--no-include-reasoning", dest="include_reasoning", action="store_false")
    parser.set_defaults(include_reasoning=True)

    parser.add_argument("--config", default="configs/optimization.yaml")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--budget-cap-usd",
        type=float,
        default=None,
        help="Optional override for run budget cap in USD.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config.load_env()

    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    budget_cap_usd = (
        float(args.budget_cap_usd)
        if args.budget_cap_usd is not None
        else float(settings["max_budget_usd_coverage"])
    )
    max_concurrency = int(settings.get("max_concurrent_requests", config.MAX_CONCURRENT_REQUESTS))
    checkpoint_every = int(settings.get("checkpoint_every", 50))

    if not config.SETTINGS.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run smoke coverage extraction.")

    engine = get_engine()
    split_sizes = {
        "refinement": int(settings.get("refinement_split_size", 150)),
        "holdout": int(settings.get("holdout_split_size", 150)),
        "smoke": int(settings.get("smoke_split_size", 200)),
        "methodology_1k": int(settings.get("methodology_1k_split_size", 1000)),
        "methodology_5k": int(settings.get("methodology_5k_split_size", 5000)),
        "methodology_5k_audit_500": int(
            settings.get("methodology_5k_audit_500_split_size", 500)
        ),
        "full": int(settings.get("full_split_size", TOTAL_NOTES_PROJECTION)),
    }
    split_size = split_sizes[args.split]

    if args.split == "full":
        hadm_ids = fetch_all_discharge_hadm_ids(engine)
        if not hadm_ids:
            raise RuntimeError("Full split query returned zero hadm_ids.")
        if args.n_samples is not None:
            hadm_ids = hadm_ids[: min(max(int(args.n_samples), 1), len(hadm_ids))]
        split_frame = pd.DataFrame({"hadm_id": hadm_ids})
        preflight_lines = _collect_preflight_full(engine)
        for line in preflight_lines:
            logger.info(line)
        notes = fetch_all_discharge_notes(engine)
        missing_hadm = [hadm_id for hadm_id in hadm_ids if hadm_id not in notes]
        if missing_hadm:
            raise RuntimeError(f"Missing discharge notes for hadm_ids: {missing_hadm[:10]}")
        ordered_notes = {hadm_id: notes[hadm_id] for hadm_id in hadm_ids}
    else:
        split_path = config.SPLITS_DIR / f"{args.split}_{split_size}.csv"
        if not split_path.exists():
            fallback_path = config.SPLITS_DIR / f"{args.split}.csv"
            if fallback_path.exists():
                split_path = fallback_path
        if not split_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_path}")

        split_frame = pd.read_csv(split_path)
        split_frame["hadm_id"] = pd.to_numeric(split_frame["hadm_id"], errors="coerce").astype(
            "int64"
        )
        split_frame = split_frame.sort_values("hadm_id", kind="mergesort").reset_index(drop=True)

        if split_frame.empty:
            raise RuntimeError("Selected split CSV is empty; cannot run coverage extraction.")

        if args.n_samples is not None:
            n_take = min(max(int(args.n_samples), 1), len(split_frame))
            split_frame = split_frame.head(n_take).copy()

        hadm_ids = split_frame["hadm_id"].astype(int).tolist()
        preflight_lines = _collect_preflight(engine, split_path, hadm_ids[0])
        for line in preflight_lines:
            logger.info(line)

        notes = fetch_notes_by_hadm_ids(engine, hadm_ids)
        missing_hadm = [hadm_id for hadm_id in hadm_ids if hadm_id not in notes]
        if missing_hadm:
            raise RuntimeError(f"Missing discharge notes for hadm_ids: {missing_hadm[:10]}")
        ordered_notes = {hadm_id: notes[hadm_id] for hadm_id in hadm_ids}
    output_dir = config.RAW_RESPONSES_DIR / args.run_id
    resume = not args.no_resume

    client = LLMClient(
        semaphore_limit=max_concurrency,
        run_id=args.run_id,
        max_budget_usd=budget_cap_usd,
    )

    summary = asyncio.run(
        run_batch(
            notes=ordered_notes,
            client=client,
            run_id=args.run_id,
            output_dir=output_dir,
            variant=args.variant,
            include_reasoning=args.include_reasoning,
            max_concurrency=max_concurrency,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )
    )

    result_payloads = _load_result_payloads(output_dir, hadm_ids)
    run_metadata = _load_run_metadata(output_dir)
    if args.output:
        report_path = Path(args.output)
    else:
        report_path = config.CODEX_OUTPUTS_DIR / f"{args.run_id}_coverage_report.md"
    if args.split == "full":
        _build_full_report(
            output_path=report_path,
            summary=summary,
            run_id=args.run_id,
            variant=args.variant,
            include_reasoning=args.include_reasoning,
            budget_cap_usd=budget_cap_usd,
            preflight_lines=preflight_lines,
            run_metadata=run_metadata,
        )
    else:
        _build_report(
            output_path=report_path,
            summary=summary,
            split_frame=split_frame,
            result_payloads=result_payloads,
            run_id=args.run_id,
            split_name=args.split,
            variant=args.variant,
            include_reasoning=args.include_reasoning,
            budget_cap_usd=budget_cap_usd,
            preflight_lines=preflight_lines,
            engine=engine,
            run_metadata=run_metadata,
        )

    print(f"Wrote coverage report to {report_path}")
    print(
        "Run summary: "
        f"n_total={summary.n_total}, "
        f"n_successful_parse={summary.n_successful_parse}, "
        f"n_failed_parse={summary.n_failed_parse}, "
        f"n_api_error={summary.n_api_error}, "
        f"total_cost_usd={summary.total_cost_usd:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
