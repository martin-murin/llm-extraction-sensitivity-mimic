"""
Selects the production prompt variant from comparison reports.

Reads: data/methodology_1k/predictions.parquet, codex_outputs/22_production_variant_selection.md.
Writes: data/methodology_1k/predictions.parquet, codex_outputs/22_production_variant_selection.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/21_select_production_variant.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src import config
from src.labeling_functions.llm_lf import FieldType, SNORKEL_TARGET_FIELD_VALUE_PAIRS

TOTAL_NOTES_PROJECTION_DEFAULT = 331_793


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        vals = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider, *lines])


def _load_features(run_id: str) -> dict[int, dict[str, Any]]:
    results_path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    out: dict[int, dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not bool(payload.get("parse_ok", False)):
            continue
        features = payload.get("features_json")
        if isinstance(features, dict):
            out[int(payload["hadm_id"])] = features
    return out


def _load_result_rows(run_id: str) -> list[dict[str, Any]]:
    results_path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    return [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _variant_vote(
    *,
    features: dict[str, Any],
    target_field: str,
    target_value: str,
    field_type: FieldType,
) -> str:
    if field_type == FieldType.ADMISSION_TAG_MEMBERSHIP:
        tags = features.get("admission_reason_tags", [])
        if isinstance(tags, list) and target_value in tags:
            return "positive"
        return "abstain"

    if field_type == FieldType.TRISTATE:
        value = str(features.get(target_field, "not_documented"))
        if value == "not_documented":
            return "abstain"
        if target_value == "yes":
            return "positive" if value == "yes" else "negative"
        if target_value == "no":
            return "positive" if value == "no" else "negative"
        return "abstain"

    return "abstain"


def _snorkel_label(fit_status: str, prob_positive: float) -> str:
    if fit_status == "no_votes":
        return "abstain"
    return "positive" if prob_positive >= 0.5 else "negative"


def select_variant_by_score(
    *,
    scores: dict[str, float],
    median_input_tokens: dict[str, float],
    tie_threshold_pp: float = 1.0,
) -> tuple[str, bool, float]:
    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranking) < 2:
        return ranking[0][0], False, 100.0

    top_variant, top_score = ranking[0]
    _, second_score = ranking[1]
    delta_pp = (top_score - second_score) * 100.0
    if delta_pp <= tie_threshold_pp:
        tied_variants = [
            variant
            for variant, score in ranking
            if ((top_score - score) * 100.0) <= tie_threshold_pp
        ]
        selected = min(tied_variants, key=lambda variant: median_input_tokens[variant])
        return selected, True, delta_pp
    return top_variant, False, delta_pp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select production variant from methodology 1k.")
    parser.add_argument(
        "--run-ids",
        nargs=3,
        default=["methodology_1k_a", "methodology_1k_b", "methodology_1k_c"],
        metavar=("RUN_A", "RUN_B", "RUN_C"),
    )
    parser.add_argument(
        "--predictions-parquet",
        default="data/methodology_1k/predictions.parquet",
    )
    parser.add_argument(
        "--output",
        default="codex_outputs/22_production_variant_selection.md",
    )
    parser.add_argument(
        "--total-notes-projection",
        type=int,
        default=TOTAL_NOTES_PROJECTION_DEFAULT,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    run_map = {"A": args.run_ids[0], "B": args.run_ids[1], "C": args.run_ids[2]}
    feature_maps = {label: _load_features(run_id) for label, run_id in run_map.items()}
    row_maps = {label: _load_result_rows(run_id) for label, run_id in run_map.items()}
    target_field_types = {
        (field, value): field_type
        for field, value, field_type in SNORKEL_TARGET_FIELD_VALUE_PAIRS
    }

    pred_path = Path(args.predictions_parquet)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing Snorkel predictions parquet: {pred_path}")
    pred_df = pd.read_parquet(pred_path)

    if pred_df.empty:
        raise RuntimeError("Predictions parquet is empty; cannot select production variant.")

    pred_df["target_field"] = pred_df["target_field"].astype(str)
    pred_df["target_value"] = pred_df["target_value"].astype(str)
    pred_df["fit_status"] = pred_df["fit_status"].astype(str)
    pred_df["snorkel_prob_positive"] = pd.to_numeric(
        pred_df["snorkel_prob_positive"], errors="coerce"
    ).fillna(0.5)

    per_target_rows: list[dict[str, Any]] = []
    per_variant_scores: dict[str, float] = {}

    for label in ["A", "B", "C"]:
        weighted_numerator = 0.0
        weighted_denominator = 0.0

        for (target_field, target_value), group in pred_df.groupby(
            ["target_field", "target_value"], sort=True
        ):
            field_type = target_field_types.get((target_field, target_value))
            if field_type is None:
                continue

            support = int((group["fit_status"] != "no_votes").sum())
            if support == 0:
                continue

            matches = 0
            total = 0
            for _, row in group.iterrows():
                hadm_id = int(row["hadm_id"])
                features = feature_maps[label].get(hadm_id)
                if features is None:
                    continue
                variant_vote = _variant_vote(
                    features=features,
                    target_field=target_field,
                    target_value=target_value,
                    field_type=field_type,
                )
                snorkel_vote = _snorkel_label(
                    fit_status=str(row["fit_status"]),
                    prob_positive=float(row["snorkel_prob_positive"]),
                )
                total += 1
                if variant_vote == snorkel_vote:
                    matches += 1

            agreement = (matches / total) if total else 0.0
            weighted_numerator += agreement * support
            weighted_denominator += support

            per_target_rows.append(
                {
                    "variant": label,
                    "target_field": target_field,
                    "target_value": target_value,
                    "support": support,
                    "agreement_pct": f"{agreement * 100.0:.2f}",
                }
            )

        per_variant_scores[label] = (
            weighted_numerator / weighted_denominator if weighted_denominator else 0.0
        )

    token_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for label in ["A", "B", "C"]:
        rows = row_maps[label]
        inputs = [float(row.get("input_tokens", 0) or 0) for row in rows]
        outputs = [float(row.get("output_tokens", 0) or 0) for row in rows]
        median_input = float(np.median(np.asarray(inputs, dtype=np.float64))) if inputs else 0.0
        median_output = (
            float(np.median(np.asarray(outputs, dtype=np.float64))) if outputs else 0.0
        )
        per_note_cost = (
            (median_input / 1_000_000.0) * config.INPUT_PRICE_PER_MILLION_USD
            + (median_output / 1_000_000.0) * config.OUTPUT_PRICE_PER_MILLION_USD
        )
        projected_cost = per_note_cost * int(args.total_notes_projection)
        token_rows.append(
            {
                "variant": label,
                "run_id": run_map[label],
                "median_input_tokens": f"{median_input:.1f}",
                "median_output_tokens": f"{median_output:.1f}",
                "per_note_cost_usd": f"{per_note_cost:.6f}",
            }
        )
        projection_rows.append(
            {
                "variant": label,
                "projected_cost_usd": f"{projected_cost:.2f}",
            }
        )

    median_tokens = {
        row["variant"]: float(row["median_input_tokens"]) for row in token_rows
    }
    selection_variant, tie_break_used, delta_pp = select_variant_by_score(
        scores=per_variant_scores,
        median_input_tokens=median_tokens,
        tie_threshold_pp=1.0,
    )

    ranking_rows = sorted(
        [
            {
                "variant": label,
                "run_id": run_map[label],
                "mean_weighted_agreement_with_snorkel_pct": per_variant_scores[label] * 100.0,
                "median_input_tokens": median_tokens[label],
            }
            for label in ["A", "B", "C"]
        ],
        key=lambda row: float(row["mean_weighted_agreement_with_snorkel_pct"]),
        reverse=True,
    )

    selection_run_id = run_map[selection_variant]
    selection_projection = next(
        row for row in projection_rows if row["variant"] == selection_variant
    )["projected_cost_usd"]

    lines = [
        "# Production Variant Selection (Methodology 1k)",
        "",
        "## Run metadata",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_ids": ", ".join(args.run_ids),
                    "predictions_parquet": str(pred_path),
                    "total_notes_projection": int(args.total_notes_projection),
                }
            ],
            ["timestamp_utc", "run_ids", "predictions_parquet", "total_notes_projection"],
        ),
        "",
        "## Per-variant agreement with Snorkel aggregate",
        _markdown_table(
            ranking_rows,
            [
                "variant",
                "run_id",
                "mean_weighted_agreement_with_snorkel_pct",
                "median_input_tokens",
            ],
        ),
        "",
        "## Token and per-note cost stats",
        _markdown_table(
            token_rows,
            [
                "variant",
                "run_id",
                "median_input_tokens",
                "median_output_tokens",
                "per_note_cost_usd",
            ],
        ),
        "",
        "## Projected 332k-scale cost by variant",
        _markdown_table(projection_rows, ["variant", "projected_cost_usd"]),
        "",
        "## Selection",
        _markdown_table(
            [
                {
                    "selected_variant": selection_variant,
                    "selected_run_id": selection_run_id,
                    "tie_break_used": tie_break_used,
                    "top_minus_second_delta_pp": f"{delta_pp:.4f}",
                    "selected_projected_cost_usd": selection_projection,
                }
            ],
            [
                "selected_variant",
                "selected_run_id",
                "tie_break_used",
                "top_minus_second_delta_pp",
                "selected_projected_cost_usd",
            ],
        ),
        "",
        (
            "Selection rule: highest mean weighted per-field agreement with Snorkel aggregate; "
            "if top variants are within 1 pp, choose lower median input tokens."
        ),
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote production variant selection report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
