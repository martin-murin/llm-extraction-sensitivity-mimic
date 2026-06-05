"""
Checks section-parser coverage on local notes.

Reads: codex_outputs/07_section_coverage.md.
Writes: codex_outputs/07_section_coverage.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/07_smoke_section_coverage.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from src import config
from src.db.connection import get_engine
from src.db.queries import (
    fetch_icd_codes_by_hadm_ids,
    fetch_notes_by_hadm_ids,
    fetch_primary_icd_by_hadm_ids,
)
from src.labeling_functions.base import LFInput, Vote
from src.labeling_functions.icd_lf import build_all_icd_lfs
from src.labeling_functions.section_parser import SECTION_ALIASES, coverage_report, parse_sections

logger = logging.getLogger("scripts.07_smoke_section_coverage")

MARTIN_SECTION_COVERAGE_PCT: dict[str, float] = {
    "Discharge Condition": 99.9,
    "Pertinent Results": 99.0,
    "Discharge Medications": 98.9,
    "History of Present Illness": 98.2,
    "Past Medical History": 98.2,
    "Discharge Instructions": 99.6,
    "Discharge Disposition": 98.7,
    "Physical Exam": 96.1,
    "Chief Complaint": 97.0,
    "Discharge Diagnosis": 94.0,
    "Brief Hospital Course": 83.2,
}


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}" if not value.is_integer() else f"{int(value)}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        values = [
            _format_number(row.get(column, "")).replace("\n", " ").replace("|", "\\|")
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *lines])


def _load_coverage_v2_features() -> dict[int, dict[str, Any]]:
    results_path = config.RAW_RESPONSES_DIR / "coverage_v2" / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"coverage_v2 results missing: {results_path}")

    features_by_hadm: dict[int, dict[str, Any]] = {}
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("parse_ok"):
                continue
            features = row.get("features_json")
            if isinstance(features, dict):
                features_by_hadm[int(row["hadm_id"])] = features
    return features_by_hadm


def _build_alias_resolution_rows(
    parsed_sections: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for canonical_name, aliases in SECTION_ALIASES.items():
        canonical_hits = 0
        alias_hits = 0
        for sections in parsed_sections.values():
            if canonical_name in sections:
                canonical_hits += 1
                continue
            for alias in aliases:
                if alias == canonical_name:
                    continue
                if alias in sections:
                    alias_hits += 1
                    break

        total_present = canonical_hits + alias_hits
        alias_rate = (alias_hits / total_present * 100.0) if total_present else 0.0
        rows.append(
            {
                "canonical_section": canonical_name,
                "canonical_hits": canonical_hits,
                "alias_hits": alias_hits,
                "total_present": total_present,
                "alias_hit_rate_pct": alias_rate,
            }
        )
    return sorted(rows, key=lambda row: row["canonical_section"])


def _build_section_coverage_rows(report_df: pd.DataFrame) -> list[dict[str, Any]]:
    report_by_section = {
        str(row["canonical_section"]): row
        for _, row in report_df.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for section_name, martin_pct in MARTIN_SECTION_COVERAGE_PCT.items():
        row = report_by_section.get(section_name)
        our_pct = float(row["coverage_pct"]) if row is not None else 0.0
        n_present = int(row["n_present"]) if row is not None else 0
        n_absent = int(row["n_absent"]) if row is not None else 0
        median_len = float(row["median_length_chars"]) if row is not None else 0.0
        rows.append(
            {
                "canonical_section": section_name,
                "martin_coverage_pct": martin_pct,
                "our_coverage_pct": our_pct,
                "delta_pp": our_pct - martin_pct,
                "n_present": n_present,
                "n_absent": n_absent,
                "median_length_chars": median_len,
            }
        )
    return rows


def _build_lf_firing_rows(
    engine: Engine,
    hadm_ids: list[int],
    notes: dict[int, str],
    parsed_sections: dict[int, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, set[int]]]:
    icd_codes_map = fetch_icd_codes_by_hadm_ids(engine, hadm_ids)
    primary_map = fetch_primary_icd_by_hadm_ids(engine, hadm_ids)

    lfs = build_all_icd_lfs()
    positive_hadm_by_lf: dict[str, set[int]] = {lf.name: set() for lf in lfs}

    for hadm_id in hadm_ids:
        primary_code, primary_version = primary_map.get(hadm_id, (None, None))
        inputs = LFInput(
            hadm_id=hadm_id,
            note_text=notes.get(hadm_id, ""),
            icd_codes=icd_codes_map.get(hadm_id, []),
            primary_icd_code=primary_code,
            primary_icd_version=primary_version,
            sections=parsed_sections.get(hadm_id),
        )
        for lf in lfs:
            output = lf(inputs)
            if output.vote == Vote.POSITIVE:
                positive_hadm_by_lf[lf.name].add(hadm_id)

    rows = [
        {
            "lf_name": lf.name,
            "target_field": lf.target_field,
            "target_value": lf.target_value,
            "positive_votes": len(positive_hadm_by_lf[lf.name]),
        }
        for lf in lfs
    ]
    rows = sorted(rows, key=lambda row: (-row["positive_votes"], row["lf_name"]))
    return rows, positive_hadm_by_lf


def _build_sanity_rows(
    positive_hadm_by_lf: dict[str, set[int]],
    llm_features: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    sanity_rows: list[dict[str, Any]] = []

    hf_hadm_ids = positive_hadm_by_lf.get("icd_hf_admission", set())
    hf_agree = 0
    for hadm_id in hf_hadm_ids:
        tags = llm_features.get(hadm_id, {}).get("admission_reason_tags", [])
        if isinstance(tags, list) and "cardiac_hf" in tags:
            hf_agree += 1
    hf_pct = (hf_agree / len(hf_hadm_ids) * 100.0) if hf_hadm_ids else 0.0
    sanity_rows.append(
        {
            "lf_name": "icd_hf_admission",
            "lf_positive_notes": len(hf_hadm_ids),
            "llm_match_definition": "LLM admission_reason_tags contains cardiac_hf",
            "agreement_pct": hf_pct,
        }
    )

    aki_hadm_ids = positive_hadm_by_lf.get("icd_aki_primary", set())
    aki_agree = 0
    for hadm_id in aki_hadm_ids:
        aki_value = llm_features.get(hadm_id, {}).get("aki_present")
        if aki_value == "yes":
            aki_agree += 1
    aki_pct = (aki_agree / len(aki_hadm_ids) * 100.0) if aki_hadm_ids else 0.0
    sanity_rows.append(
        {
            "lf_name": "icd_aki_primary",
            "lf_positive_notes": len(aki_hadm_ids),
            "llm_match_definition": "LLM aki_present == yes",
            "agreement_pct": aki_pct,
        }
    )
    return sanity_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3a smoke diagnostics for sections and ICD LFs."
    )
    parser.add_argument("--output", default="codex_outputs/07_section_coverage.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config.load_env()

    split_path = config.SPLITS_DIR / "smoke_200.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Smoke split CSV missing: {split_path}")

    split_df = pd.read_csv(split_path)
    split_df["hadm_id"] = pd.to_numeric(split_df["hadm_id"], errors="coerce").astype("int64")
    hadm_ids = sorted(split_df["hadm_id"].astype(int).tolist())

    engine = get_engine()
    notes = fetch_notes_by_hadm_ids(engine, hadm_ids)
    missing = [hadm_id for hadm_id in hadm_ids if hadm_id not in notes]
    if missing:
        raise RuntimeError(f"Missing notes for hadm_ids: {missing[:10]}")

    parsed_sections = {hadm_id: parse_sections(notes[hadm_id]) for hadm_id in hadm_ids}
    no_section_hadm_ids = [
        hadm_id
        for hadm_id, sections in parsed_sections.items()
        if list(sections.keys()) == ["__full_note__"]
    ]

    section_report = coverage_report(notes)
    section_rows = _build_section_coverage_rows(section_report)
    alias_rows = _build_alias_resolution_rows(parsed_sections)
    lf_firing_rows, positive_hadm_by_lf = _build_lf_firing_rows(
        engine,
        hadm_ids,
        notes,
        parsed_sections,
    )
    llm_features = _load_coverage_v2_features()
    sanity_rows = _build_sanity_rows(positive_hadm_by_lf, llm_features)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# Phase 3a Smoke Section Coverage",
                "",
                "## Run metadata",
                _markdown_table(
                    [
                        {"field": "timestamp_utc", "value": datetime.now(tz=UTC).isoformat()},
                        {
                            "field": "notes_source",
                            "value": (
                                "DB via fetch_notes_by_hadm_ids(smoke split hadm_ids)"
                            ),
                        },
                        {"field": "n_notes", "value": len(hadm_ids)},
                    ],
                    ["field", "value"],
                ),
                "",
                "## Section coverage",
                _markdown_table(
                    section_rows,
                    [
                        "canonical_section",
                        "martin_coverage_pct",
                        "our_coverage_pct",
                        "delta_pp",
                        "n_present",
                        "n_absent",
                        "median_length_chars",
                    ],
                ),
                "",
                "## Alias resolution hits",
                _markdown_table(
                    alias_rows,
                    [
                        "canonical_section",
                        "canonical_hits",
                        "alias_hits",
                        "total_present",
                        "alias_hit_rate_pct",
                    ],
                ),
                "",
                "## No-section notes",
                (
                    "_None_"
                    if not no_section_hadm_ids
                    else _markdown_table(
                        [{"hadm_id": hadm_id} for hadm_id in no_section_hadm_ids],
                        ["hadm_id"],
                    )
                ),
                "",
                "## ICD LF firing rates",
                _markdown_table(
                    lf_firing_rows,
                    ["lf_name", "target_field", "target_value", "positive_votes"],
                ),
                "",
                "## Sanity check",
                _markdown_table(
                    sanity_rows,
                    ["lf_name", "lf_positive_notes", "llm_match_definition", "agreement_pct"],
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Wrote report to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
