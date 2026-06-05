"""
Explores local MIMIC note availability, joins, note lengths, and ICD distributions.

Reads: codex_outputs/01_mimic_exploration.md.
Writes: codex_outputs/01_mimic_exploration.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/01_explore_mimic_notes.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import platform
import socket
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd
import tiktoken
from sqlalchemy import Engine, text
from src import config
from src.db.connection import discover_schemas, get_engine
from src.db.queries import (
    count_notes,
    join_cardinality,
    note_length_stats,
    sample_redaction_excerpts,
    top_any_position_icds,
    top_primary_icds,
)
from src.utils.logging import get_logger


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _quote_qualified_name(qualified_name: str) -> str:
    schema, table = qualified_name.split(".", maxsplit=1)
    schema = schema.replace('"', '""')
    table = table.replace('"', '""')
    return f'"{schema}"."{table}"'


def _resolve_encoding(logger: Any) -> tuple[tiktoken.Encoding, str]:
    for model_name in (config.MODEL_ID, "gpt-5.4-nano"):
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            return encoding, model_name
        except KeyError:
            continue

    logger.info(
        "Using tiktoken encoding `o200k_base` as fallback for model `%s`.",
        config.MODEL_ID,
    )
    return tiktoken.get_encoding("o200k_base"), "o200k_base"


def _sample_token_lengths(
    engine: Engine,
    discharge_table: str,
    sample_size: int,
    encoding: tiktoken.Encoding,
) -> np.ndarray:
    query = text(
        f"""
        SELECT COALESCE(text, '') AS note_text
        FROM {_quote_qualified_name(discharge_table)}
        ORDER BY random()
        LIMIT :sample_size
        """
    )

    with engine.connect() as conn:
        texts = [row.note_text for row in conn.execute(query, {"sample_size": sample_size})]

    return np.asarray([len(encoding.encode(note_text)) for note_text in texts], dtype=np.int64)


def _series_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"

    body_lines = []
    for row in rows:
        rendered = []
        for column in columns:
            text_value = _format_number(row.get(column, ""))
            text_value = text_value.replace("\n", " ").replace("|", "\\|")
            rendered.append(text_value)
        body_lines.append("| " + " | ".join(rendered) + " |")

    return "\n".join([header, divider, *body_lines])


def _dataframe_to_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [dict(record) for record in frame.to_dict(orient="records")]


def _count_table(engine: Engine, qualified_name: str) -> int:
    query = text(f"SELECT COUNT(*) AS n FROM {_quote_qualified_name(qualified_name)}")
    with engine.connect() as conn:
        return int(conn.execute(query).scalar_one())


def _render_join_commentary(cardinality: dict[str, int]) -> str:
    orphan_count = cardinality["n_discharge_notes_orphan"]
    unique_hadm = cardinality["n_unique_hadm_ids_in_notes"]
    admissions_with_notes = cardinality["n_admissions_with_discharge_note"]

    if orphan_count == 0 and unique_hadm > 0 and admissions_with_notes > 0:
        return (
            "`hadm_id` appears to be a reliable join key in this installation: no orphan discharge "
            "notes were detected."
        )

    return (
        "`hadm_id` requires caution on this installation: orphan discharge notes were detected "
        f"({orphan_count})."
    )


def build_report(run_id: str, output_path: Path) -> Path:
    config.load_env()
    logger = get_logger("scripts.01_explore_mimic_notes")

    engine = get_engine()
    schemas = discover_schemas(engine)

    for logical_name, qualified_name in schemas.items():
        logger.info(
            "Schema discovery",
            extra={"logical_table": logical_name, "table": qualified_name},
        )

    note_count = count_notes(engine)
    note_lengths = note_length_stats(engine, sample_size=10000)
    primary_icd = top_primary_icds(engine, n=20)
    any_position_icd = top_any_position_icds(engine, n=20)
    cardinality = join_cardinality(engine)
    redaction_samples = sample_redaction_excerpts(engine, n=3, max_chars=400)

    encoding, encoding_label = _resolve_encoding(logger)
    token_lengths = _sample_token_lengths(
        engine,
        discharge_table=str(schemas["discharge_notes"]),
        sample_size=len(note_lengths),
        encoding=encoding,
    )

    char_stats = _series_summary(note_lengths["char_len"].to_numpy(dtype=np.int64))
    token_stats = _series_summary(token_lengths)

    n_admissions = max(cardinality["n_admissions"], 1)
    primary_icd = primary_icd.copy()
    primary_icd["pct_of_admissions"] = (primary_icd["count"] / n_admissions) * 100

    any_position_icd = any_position_icd.copy()
    any_position_icd["pct_of_admissions"] = (any_position_icd["count"] / n_admissions) * 100

    table_counts: list[dict[str, Any]] = []
    for logical_name in [
        "discharge_notes",
        "radiology_notes",
        "admissions",
        "diagnoses_icd",
        "d_icd_diagnoses",
        "patients",
    ]:
        qualified_name = schemas.get(logical_name)
        if qualified_name is None:
            row_count: Any = "not-found"
        else:
            row_count = _count_table(engine, qualified_name)
        table_counts.append(
            {
                "logical_table": logical_name,
                "schema_table": qualified_name or "not-found",
                "row_count": row_count,
            }
        )

    redaction_hits = sum("___" in str(sample.get("excerpt", "")) for sample in redaction_samples)

    mean_input_tokens = float(token_lengths.mean()) if token_lengths.size > 0 else 0.0
    projected_input_tokens = note_count * mean_input_tokens
    projected_output_tokens = note_count * 600
    projected_input_cost = projected_input_tokens / 1_000_000 * config.INPUT_PRICE_PER_MILLION_USD
    projected_output_cost = (
        projected_output_tokens / 1_000_000 * config.OUTPUT_PRICE_PER_MILLION_USD
    )
    projected_total_cost = projected_input_cost + projected_output_cost

    run_timestamp = datetime.now(tz=UTC).isoformat()
    run_meta_rows = [
        {"field": "run_id", "value": run_id},
        {"field": "timestamp_utc", "value": run_timestamp},
        {"field": "hostname", "value": socket.gethostname()},
        {"field": "python_version", "value": platform.python_version()},
        {"field": "openai_version", "value": _package_version("openai")},
        {"field": "pydantic_version", "value": _package_version("pydantic")},
        {"field": "sqlalchemy_version", "value": _package_version("sqlalchemy")},
        {"field": "tiktoken_version", "value": _package_version("tiktoken")},
        {"field": "tiktoken_encoding", "value": encoding_label},
    ]

    schema_rows = [
        {"logical_table": key, "schema_table": value or "not-found"}
        for key, value in schemas.items()
    ]

    length_rows = []
    for label in ["min", "p10", "p25", "median", "p75", "p90", "p99", "max"]:
        length_rows.append(
            {
                "stat": label,
                "char_len": char_stats[label],
                "token_len": token_stats[label],
            }
        )

    join_rows = [{"metric": key, "value": value} for key, value in cardinality.items()]

    report = "\n".join(
        [
            "# MIMIC Exploration Report",
            "",
            "## Run metadata",
            _markdown_table(run_meta_rows, ["field", "value"]),
            "",
            "## Schema discovery",
            _markdown_table(schema_rows, ["logical_table", "schema_table"]),
            "",
            "## Table counts",
            _markdown_table(table_counts, ["logical_table", "schema_table", "row_count"]),
            "",
            "## Discharge note length distribution",
            f"Sample size: {len(note_lengths):,}",
            "",
            _markdown_table(length_rows, ["stat", "char_len", "token_len"]),
            "",
            "## Top 20 primary ICDs",
            _markdown_table(
                _dataframe_to_rows(
                    primary_icd[
                        ["icd_code", "icd_version", "count", "pct_of_admissions", "description"]
                    ]
                ),
                ["icd_code", "icd_version", "count", "pct_of_admissions", "description"],
            ),
            "",
            "## Top 20 any-position ICDs",
            _markdown_table(
                _dataframe_to_rows(
                    any_position_icd[
                        ["icd_code", "icd_version", "count", "pct_of_admissions", "description"]
                    ]
                ),
                ["icd_code", "icd_version", "count", "pct_of_admissions", "description"],
            ),
            "",
            "## Join cardinality",
            _markdown_table(join_rows, ["metric", "value"]),
            "",
            _render_join_commentary(cardinality),
            "",
            "## Redaction pattern samples",
            f"`___` occurrences in sampled excerpts: {redaction_hits} / {len(redaction_samples)}",
            "",
            _markdown_table(redaction_samples, ["hadm_id", "excerpt"]),
            "",
            "## Estimated cost envelope",
            _markdown_table(
                [
                    {
                        "metric": "n_discharge_notes",
                        "value": note_count,
                    },
                    {
                        "metric": "mean_input_tokens_per_note_observed",
                        "value": mean_input_tokens,
                    },
                    {
                        "metric": "assumed_output_tokens_per_note",
                        "value": 600,
                    },
                    {
                        "metric": "projected_total_input_tokens",
                        "value": projected_input_tokens,
                    },
                    {
                        "metric": "projected_total_output_tokens",
                        "value": projected_output_tokens,
                    },
                    {
                        "metric": "projected_input_cost_usd",
                        "value": projected_input_cost,
                    },
                    {
                        "metric": "projected_output_cost_usd",
                        "value": projected_output_cost,
                    },
                    {
                        "metric": "projected_total_cost_usd",
                        "value": projected_total_cost,
                    },
                ],
                ["metric", "value"],
            ),
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore MIMIC discharge notes and emit a markdown report."
    )
    parser.add_argument(
        "--run-id",
        default="explore_v1",
        help="Identifier for this exploration run.",
    )
    parser.add_argument(
        "--output",
        default="codex_outputs/01_mimic_exploration.md",
        help="Path to the markdown report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    report_path = build_report(run_id=args.run_id, output_path=output_path)
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
