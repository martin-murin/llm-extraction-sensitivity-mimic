"""
Evaluates regex labeling-function smoke coverage.

Reads: src/labeling_functions/patterns, data/raw_responses/coverage_v2/results.jsonl, codex_outputs/09_regex_smoke_coverage.md, codex_outputs/09b_regex_smoke_coverage_post_review.md.
Writes: data/raw_responses/coverage_v2/results.jsonl, codex_outputs/09_regex_smoke_coverage.md, codex_outputs/09b_regex_smoke_coverage_post_review.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/09_smoke_regex_coverage.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src import config
from src.db.connection import get_engine
from src.db.queries import fetch_notes_by_hadm_ids
from src.labeling_functions.base import LFInput, Vote
from src.labeling_functions.regex_lf import (
    RegexLabelingFunction,
    build_all_regex_lfs,
    eval_compound_pattern,
    is_negated,
)
from src.labeling_functions.section_parser import get_section, parse_sections
from src.schema.section_map import FIELD_SECTION_MAP

logger = logging.getLogger("scripts.09_smoke_regex_coverage")

ORIGINAL_V1_FIELDS: tuple[str, ...] = (
    "dnr_dni_documented",
    "home_health_ordered",
    "palliative_care_consult",
    "substance_use_active",
)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = []
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *lines])


def _pattern_flags(pattern: str) -> re.RegexFlag:
    if "(?-i)" in pattern:
        return re.NOFLAG
    return re.IGNORECASE


def _extract_matched_pattern(evidence: str | None) -> str | None:
    if not evidence:
        return None
    prefix = "regex match: "
    if not evidence.startswith(prefix):
        return None
    return evidence[len(prefix):]


def _extract_matched_compound(evidence: str | None) -> str | None:
    if not evidence:
        return None
    prefix = "compound match: "
    if not evidence.startswith(prefix):
        return None
    return evidence[len(prefix):]


def _build_search_text_for_field(sections: dict[str, str], field_name: str, note_text: str) -> str:
    full_note = sections.get("__full_note__")
    if isinstance(full_note, str) and full_note:
        return full_note

    chunks: list[str] = []
    for canonical in FIELD_SECTION_MAP[field_name]:
        section_text = get_section(sections, canonical)
        if section_text:
            chunks.append(section_text)

    if chunks:
        return "\n---\n".join(chunks)
    return ""


def _extract_excerpt(search_text: str, pattern: str, radius: int = 80) -> str:
    if not search_text:
        return ""
    compiled = re.compile(pattern, flags=_pattern_flags(pattern))
    for match in compiled.finditer(search_text):
        if is_negated(search_text, match.start()):
            continue
        start = max(0, match.start() - radius)
        end = min(len(search_text), match.end() + radius)
        excerpt = search_text[start:end].replace("\n", " ")
        return excerpt.strip()
    return ""


def _load_coverage_features(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"coverage_v2 results not found: {path}")

    features_by_hadm: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not bool(payload.get("parse_ok", False)):
                continue
            features = payload.get("features_json")
            if not isinstance(features, dict):
                continue
            features_by_hadm[int(payload["hadm_id"])] = features
    return features_by_hadm


def _parse_markdown_table_for_section(report_text: str, section_title: str) -> list[dict[str, str]]:
    marker = f"## {section_title}"
    section_start = report_text.find(marker)
    if section_start < 0:
        return []

    remainder = report_text[section_start + len(marker):]
    lines = [line.rstrip("\n") for line in remainder.splitlines()]
    table_lines: list[str] = []
    saw_table = False
    for line in lines:
        if line.startswith("## "):
            break
        if line.strip().startswith("|"):
            table_lines.append(line.strip())
            saw_table = True
            continue
        if saw_table and not line.strip():
            break

    if len(table_lines) < 2:
        return []
    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for row_line in table_lines[2:]:
        cells = [cell.strip() for cell in row_line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells, strict=False)))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regex LF smoke coverage diagnostics.")
    parser.add_argument(
        "--patterns-dir",
        type=Path,
        default=Path("src/labeling_functions/patterns"),
        help="Directory containing regex pattern YAML files",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/raw_responses/coverage_v2/results.jsonl"),
        help="Path to coverage_v2 results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("codex_outputs/09_regex_smoke_coverage.md"),
        help="Output markdown report path",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("codex_outputs/09b_regex_smoke_coverage_post_review.md"),
        help="Baseline regex report for v1-v2 comparison.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    config.load_env()

    split_path = config.SPLITS_DIR / "smoke_200.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Smoke split file not found: {split_path}")

    split_df = pd.read_csv(split_path)
    if "hadm_id" not in split_df.columns:
        raise RuntimeError(f"Smoke split missing hadm_id column: {split_path}")

    hadm_ids = sorted(int(value) for value in split_df["hadm_id"].tolist())
    engine = get_engine()
    notes = fetch_notes_by_hadm_ids(engine, hadm_ids)

    parsed_sections = {hadm_id: parse_sections(notes.get(hadm_id, "")) for hadm_id in hadm_ids}
    regex_lfs = build_all_regex_lfs(args.patterns_dir)
    features_by_hadm = _load_coverage_features(args.results)

    per_lf_positive: dict[str, list[dict[str, Any]]] = {lf.name: [] for lf in regex_lfs}
    for hadm_id in hadm_ids:
        note_text = notes.get(hadm_id, "")
        sections = parsed_sections[hadm_id]
        lf_input = LFInput(hadm_id=hadm_id, note_text=note_text, sections=sections)

        for lf in regex_lfs:
            output = lf(lf_input)
            if output.vote != Vote.POSITIVE:
                continue

            matched_pattern = _extract_matched_pattern(output.evidence)
            matched_compound = _extract_matched_compound(output.evidence)

            search_text = _build_search_text_for_field(sections, str(lf.target_field), note_text)
            excerpt = ""
            if matched_pattern is not None:
                excerpt = _extract_excerpt(search_text, matched_pattern)
            elif matched_compound is not None and isinstance(lf, RegexLabelingFunction):
                for compound_pattern in lf.compound_patterns:
                    description_raw = compound_pattern.get("description")
                    description = (
                        str(description_raw)
                        if description_raw is not None
                        else str(compound_pattern["all_of"])
                    )
                    if description != matched_compound:
                        continue
                    match_info = eval_compound_pattern(
                        search_text,
                        [str(item) for item in compound_pattern["all_of"]],
                        int(compound_pattern["window_chars"]),
                    )
                    if match_info is None:
                        continue
                    _, _, excerpt = match_info
                    break

            per_lf_positive[lf.name].append(
                {
                    "hadm_id": hadm_id,
                    "field": lf.target_field,
                    "target_value": lf.target_value,
                    "matched_pattern": matched_pattern or "",
                    "compound_description": matched_compound or "",
                    "match_evidence": output.evidence or "",
                    "excerpt": excerpt,
                    "pattern_file": getattr(lf, "pattern_source", ""),
                }
            )

    all_fields = sorted({str(lf.target_field) for lf in regex_lfs})

    firing_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    compound_firing_rows: list[dict[str, Any]] = []
    fp_examples: list[dict[str, Any]] = []
    miss_examples: list[dict[str, Any]] = []
    regex_metrics_by_field: dict[str, dict[str, float | int]] = {}

    for lf in regex_lfs:
        positives = per_lf_positive.get(lf.name, [])
        regex_positive_ids = {int(row["hadm_id"]) for row in positives}
        llm_yes_ids = {
            hadm_id
            for hadm_id, features in features_by_hadm.items()
            if features.get(lf.target_field) == lf.target_value
        }
        intersection = regex_positive_ids.intersection(llm_yes_ids)

        firing_rows.append(
            {
                "pattern_file": getattr(lf, "pattern_source", ""),
                "lf_name": lf.name,
                "field": lf.target_field,
                "target_value": lf.target_value,
                "n_positive_votes": len(regex_positive_ids),
                "pct_of_notes": f"{(len(regex_positive_ids) / len(hadm_ids) * 100):.2f}%",
            }
        )

        agreement_pct = (
            (len(intersection) / len(regex_positive_ids)) * 100 if regex_positive_ids else 0.0
        )
        recall_from_llm_pct = (
            (len(intersection) / len(llm_yes_ids)) * 100 if llm_yes_ids else 0.0
        )
        agreement_rows.append(
            {
                "lf_name": lf.name,
                "field": lf.target_field,
                "regex_positive_n": len(regex_positive_ids),
                "llm_yes_n": len(llm_yes_ids),
                "intersection_n": len(intersection),
                "agreement_pct": f"{agreement_pct:.2f}%",
                "recall_from_llm_pct": f"{recall_from_llm_pct:.2f}%",
            }
        )
        regex_metrics_by_field[str(lf.target_field)] = {
            "positive_n": len(regex_positive_ids),
            "agreement_pct": float(agreement_pct),
            "recall_pct": float(recall_from_llm_pct),
        }

        if len(fp_examples) < 10:
            for row in positives:
                if int(row["hadm_id"]) in llm_yes_ids:
                    continue
                fp_examples.append(
                    {
                        "hadm_id": int(row["hadm_id"]),
                        "field": row["field"],
                        "pattern": row["matched_pattern"] or row["compound_description"],
                        "excerpt": row["excerpt"][:300],
                    }
                )
                if len(fp_examples) >= 10:
                    break

        if len(miss_examples) < 10:
            missing_ids = sorted(llm_yes_ids.difference(regex_positive_ids))
            for missing_id in missing_ids:
                features = features_by_hadm.get(missing_id, {})
                reasoning = str(features.get("reasoning") or "")
                miss_examples.append(
                    {
                        "hadm_id": missing_id,
                        "field": lf.target_field,
                        "llm_reasoning": reasoning[:300],
                    }
                )
                if len(miss_examples) >= 10:
                    break

        compound_groups: dict[str, list[dict[str, Any]]] = {}
        for row in positives:
            description = str(row.get("compound_description") or "")
            if not description:
                continue
            compound_groups.setdefault(description, []).append(row)
        for description, rows in compound_groups.items():
            unique_hadm = {int(item["hadm_id"]) for item in rows}
            example_excerpt = ""
            for item in rows:
                excerpt = str(item.get("excerpt") or "").strip()
                if excerpt:
                    example_excerpt = excerpt[:200]
                    break
            compound_firing_rows.append(
                {
                    "field": lf.target_field,
                    "description": description,
                    "n_notes_matched": len(unique_hadm),
                    "example_excerpt": example_excerpt,
                }
            )

    section_rows: list[dict[str, Any]] = []
    for field_name in all_fields:
        required_sections = FIELD_SECTION_MAP[field_name]
        for section_name in required_sections:
            n_present = 0
            for hadm_id in hadm_ids:
                section_text = get_section(parsed_sections[hadm_id], section_name)
                if section_text is not None:
                    n_present += 1
            section_rows.append(
                {
                    "field": field_name,
                    "required_section": section_name,
                    "n_present": n_present,
                    "pct_of_notes": f"{(n_present / len(hadm_ids) * 100):.2f}%",
                }
            )

    baseline_rows: list[dict[str, str]] = []
    if args.baseline_report.exists():
        baseline_rows = _parse_markdown_table_for_section(
            args.baseline_report.read_text(encoding="utf-8"),
            "LLM agreement sanity check",
        )
    baseline_by_field: dict[str, dict[str, str]] = {
        row.get("field", ""): row for row in baseline_rows if row.get("field")
    }

    v2_vs_v1_rows: list[dict[str, Any]] = []
    for field_name in sorted(
        all_fields,
        key=lambda value: (value not in ORIGINAL_V1_FIELDS, value),
    ):
        current = regex_metrics_by_field.get(field_name, {})
        baseline = baseline_by_field.get(field_name)
        if baseline is None:
            v2_vs_v1_rows.append(
                {
                    "field": field_name,
                    "v1_regex_positive_n": "n/a",
                    "v2_regex_positive_n": int(current.get("positive_n", 0)),
                    "v1_agreement_pct": "n/a",
                    "v2_agreement_pct": f"{float(current.get('agreement_pct', 0.0)):.2f}%",
                    "v1_recall_pct": "n/a",
                    "v2_recall_pct": f"{float(current.get('recall_pct', 0.0)):.2f}%",
                    "note": "first_run_in_v2",
                }
            )
            continue

        v2_vs_v1_rows.append(
            {
                "field": field_name,
                "v1_regex_positive_n": baseline.get("regex_positive_n", "n/a"),
                "v2_regex_positive_n": int(current.get("positive_n", 0)),
                "v1_agreement_pct": baseline.get("agreement_pct", "n/a"),
                "v2_agreement_pct": f"{float(current.get('agreement_pct', 0.0)):.2f}%",
                "v1_recall_pct": baseline.get("recall_from_llm_pct", "n/a"),
                "v2_recall_pct": f"{float(current.get('recall_pct', 0.0)):.2f}%",
                "note": "original_4"
                if field_name in ORIGINAL_V1_FIELDS
                else "baseline_present",
            }
        )

    lines: list[str] = []
    lines.append("# 09 Regex Smoke Coverage")
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "patterns_dir": str(args.patterns_dir),
                    "coverage_results": str(args.results),
                    "baseline_report": str(args.baseline_report),
                    "n_notes": len(hadm_ids),
                    "n_regex_lfs": len(regex_lfs),
                }
            ],
            [
                "timestamp_utc",
                "patterns_dir",
                "coverage_results",
                "baseline_report",
                "n_notes",
                "n_regex_lfs",
            ],
        )
    )
    lines.append("")

    lines.append("## Per-LF firing rates")
    lines.append("")
    lines.append(
        _markdown_table(
            firing_rows,
            [
                "pattern_file",
                "lf_name",
                "field",
                "target_value",
                "n_positive_votes",
                "pct_of_notes",
            ],
        )
    )
    lines.append("")

    lines.append("## LLM agreement sanity check")
    lines.append("")
    lines.append(
        _markdown_table(
            agreement_rows,
            [
                "lf_name",
                "field",
                "regex_positive_n",
                "llm_yes_n",
                "intersection_n",
                "agreement_pct",
                "recall_from_llm_pct",
            ],
        )
    )
    lines.append("")

    lines.append("## Compound pattern firings")
    lines.append("")
    lines.append(
        _markdown_table(
            compound_firing_rows,
            ["field", "description", "n_notes_matched", "example_excerpt"],
        )
        if compound_firing_rows
        else "_No compound pattern matches observed._"
    )
    lines.append("")

    lines.append("## Notes where regex voted POSITIVE but LLM did not")
    lines.append("")
    lines.append(
        _markdown_table(fp_examples, ["hadm_id", "field", "pattern", "excerpt"])
        if fp_examples
        else "_No examples._"
    )
    lines.append("")

    lines.append("## Notes where LLM voted yes but regex abstained")
    lines.append("")
    lines.append(
        _markdown_table(miss_examples, ["hadm_id", "field", "llm_reasoning"])
        if miss_examples
        else "_No examples._"
    )
    lines.append("")

    lines.append("## Section availability")
    lines.append("")
    lines.append(
        _markdown_table(section_rows, ["field", "required_section", "n_present", "pct_of_notes"])
    )
    lines.append("")

    lines.append("## Regex v1 vs v2 comparison")
    lines.append("")
    lines.append(
        _markdown_table(
            v2_vs_v1_rows,
            [
                "field",
                "v1_regex_positive_n",
                "v2_regex_positive_n",
                "v1_agreement_pct",
                "v2_agreement_pct",
                "v1_recall_pct",
                "v2_recall_pct",
                "note",
            ],
        )
        if v2_vs_v1_rows
        else "_No comparison rows available._"
    )
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logger.info("Wrote regex smoke coverage report to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
