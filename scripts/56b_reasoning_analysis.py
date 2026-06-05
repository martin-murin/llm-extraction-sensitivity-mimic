"""
Analyzes reasoning-instruction comparison outputs.

Reads: codex_outputs/56_reasoning_comparison.md.
Writes: codex_outputs/56_reasoning_comparison.md.
Backs Figure 2 and reasoning-instruction comparison claims.
Usage: `python scripts/56b_reasoning_analysis.py` unless the script's argparse help says otherwise.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.schema.fields import LLMNoteFeatures

RUN_OFF_DEFAULT = "methodology_1k_a"
RUN_ON_DEFAULT = "reasoning_on_methodology_1k_a"
RUN_GOLD_DEFAULT = "paired_gold_methodology_1k_a"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        vals = [str(row.get(col, "")).replace("|", "\\|").replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider, *lines])


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return tuple(sorted(set(value)))
        return tuple(value)
    return value


def _load_features(run_id: str) -> dict[int, dict[str, Any]]:
    path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    parsed: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if bool(payload.get("parse_ok", False)) and isinstance(
                payload.get("features_json"), dict
            ):
                parsed[int(payload["hadm_id"])] = payload["features_json"]
    return parsed


def _field_names() -> list[str]:
    fields = [field for field in LLMNoteFeatures.model_fields if field != "reasoning"]
    return sorted(fields)


def _agreement_vs_reference(
    run_features: dict[int, dict[str, Any]],
    ref_features: dict[int, dict[str, Any]],
    fields: list[str],
) -> tuple[int, list[dict[str, Any]]]:
    overlap = sorted(set(run_features).intersection(ref_features))
    if not overlap:
        return 0, []
    n = len(overlap)
    rows: list[dict[str, Any]] = []
    for field in fields:
        disagree = 0
        for hadm_id in overlap:
            left = _normalize(run_features[hadm_id].get(field))
            right = _normalize(ref_features[hadm_id].get(field))
            if left != right:
                disagree += 1
        rows.append(
            {
                "field": field,
                "n_overlap": n,
                "n_disagree": disagree,
                "disagree_rate_pct": (disagree / n * 100.0) if n else 0.0,
                "agree_rate_pct": ((n - disagree) / n * 100.0) if n else 0.0,
            }
        )
    rows.sort(key=lambda row: float(row["disagree_rate_pct"]), reverse=True)
    return n, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze reasoning ON vs OFF outputs for variant A on methodology_1k."
    )
    parser.add_argument("--run-off", default=RUN_OFF_DEFAULT)
    parser.add_argument("--run-on", default=RUN_ON_DEFAULT)
    parser.add_argument("--run-gold", default=RUN_GOLD_DEFAULT)
    parser.add_argument("--output", default="codex_outputs/56_reasoning_comparison.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fields = _field_names()

    off = _load_features(args.run_off)
    on = _load_features(args.run_on)
    n_overlap, on_vs_off_rows = _agreement_vs_reference(on, off, fields)

    if n_overlap == 0:
        raise RuntimeError(
            f"No overlap between {args.run_off} and {args.run_on}. Cannot compute comparison."
        )

    total_cells = n_overlap * len(fields)
    total_disagree = sum(int(row["n_disagree"]) for row in on_vs_off_rows)
    total_disagree_rate = (total_disagree / total_cells * 100.0) if total_cells else 0.0
    fields_over_5 = [row for row in on_vs_off_rows if float(row["disagree_rate_pct"]) > 5.0]

    gold_path = config.RAW_RESPONSES_DIR / args.run_gold / "results.jsonl"
    gold_available = gold_path.exists()
    on_vs_gold_rows: list[dict[str, Any]] = []
    off_vs_gold_rows: list[dict[str, Any]] = []
    n_gold_overlap = 0

    if gold_available:
        gold = _load_features(args.run_gold)
        n_gold_overlap, on_vs_gold_rows = _agreement_vs_reference(on, gold, fields)
        _, off_vs_gold_rows = _agreement_vs_reference(off, gold, fields)

    overview_rows = [
        {"metric": "timestamp_utc", "value": datetime.now(tz=UTC).isoformat()},
        {"metric": "run_off", "value": args.run_off},
        {"metric": "run_on", "value": args.run_on},
        {"metric": "n_overlap_notes", "value": n_overlap},
        {"metric": "n_fields_compared", "value": len(fields)},
        {"metric": "total_cells_compared", "value": total_cells},
        {"metric": "n_cell_disagreements", "value": total_disagree},
        {"metric": "overall_disagreement_rate_pct", "value": f"{total_disagree_rate:.3f}"},
        {"metric": "fields_over_5pct_disagreement", "value": len(fields_over_5)},
        {"metric": "paired_gold_available", "value": gold_available},
        {"metric": "paired_gold_run_id", "value": args.run_gold},
        {"metric": "paired_gold_overlap_notes", "value": n_gold_overlap if gold_available else 0},
    ]

    comparison_rows = [
        {
            "field": row["field"],
            "n_overlap": row["n_overlap"],
            "n_disagree": row["n_disagree"],
            "disagree_rate_pct": f"{row['disagree_rate_pct']:.3f}",
            "agree_rate_pct": f"{row['agree_rate_pct']:.3f}",
        }
        for row in on_vs_off_rows
    ]

    lines = [
        "# Phase 9 Reasoning ON vs OFF Comparison (Variant A, methodology_1k)",
        "",
        "## Overview",
        _markdown_table(overview_rows, ["metric", "value"]),
        "",
        "## Per-field ON vs OFF disagreement",
        _markdown_table(
            comparison_rows,
            ["field", "n_overlap", "n_disagree", "disagree_rate_pct", "agree_rate_pct"],
        ),
        "",
        "## Fields with disagreement > 5%",
        _markdown_table(
            [
                {
                    "field": row["field"],
                    "disagree_rate_pct": f"{row['disagree_rate_pct']:.3f}",
                    "n_disagree": row["n_disagree"],
                    "n_overlap": row["n_overlap"],
                }
                for row in fields_over_5
            ],
            ["field", "disagree_rate_pct", "n_disagree", "n_overlap"],
        ),
        "",
    ]

    if gold_available:
        gold_table_rows: list[dict[str, Any]] = []
        on_map = {row["field"]: row for row in on_vs_gold_rows}
        off_map = {row["field"]: row for row in off_vs_gold_rows}
        for field in fields:
            on_row = on_map[field]
            off_row = off_map[field]
            gold_table_rows.append(
                {
                    "field": field,
                    "on_agree_vs_gold_pct": f"{on_row['agree_rate_pct']:.3f}",
                    "off_agree_vs_gold_pct": f"{off_row['agree_rate_pct']:.3f}",
                    "delta_on_minus_off_pp": (
                        f"{(on_row['agree_rate_pct'] - off_row['agree_rate_pct']):+.3f}"
                    ),
                    "n_overlap": on_row["n_overlap"],
                }
            )
        gold_table_rows.sort(
            key=lambda row: abs(float(row["delta_on_minus_off_pp"])),
            reverse=True,
        )
        lines.extend(
            [
                "## ON/OFF agreement vs paired gold",
                "_Paired gold run detected; comparison included._",
                "",
                _markdown_table(
                    gold_table_rows,
                    [
                        "field",
                        "on_agree_vs_gold_pct",
                        "off_agree_vs_gold_pct",
                        "delta_on_minus_off_pp",
                        "n_overlap",
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## ON/OFF agreement vs paired gold",
                (
                    "_Pending_: paired gold run "
                    f"`{args.run_gold}` not found at "
                    f"`{gold_path}`."
                ),
                "",
            ]
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
