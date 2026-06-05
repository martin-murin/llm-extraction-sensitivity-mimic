"""
Compares refinement and holdout agreement behavior.

Reads: codex_outputs/16c_iter2_kappa.md.json, codex_outputs/21_holdout_kappa_report.md.json, codex_outputs/21_refinement_vs_holdout_comparison.md.
Writes: codex_outputs/16c_iter2_kappa.md.json, codex_outputs/21_holdout_kappa_report.md.json, codex_outputs/21_refinement_vs_holdout_comparison.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/17_refinement_vs_holdout.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src import config

TRISTATE_FIELDS: set[str] = {
    "shock_present",
    "infection_as_trigger",
    "aki_present",
    "lives_alone",
    "social_support_absent",
    "financial_hardship",
    "substance_use_active",
    "fall_risk_documented",
    "cognitive_impairment",
    "goals_of_care_flag",
    "palliative_care_consult",
    "dnr_dni_documented",
    "home_health_ordered",
    "cardiac_rehab_referred",
    "discharge_delayed_reason",
    "hospital_acquired_complication",
    "unresolved_diagnosis_at_discharge",
}

ENUM_FIELDS: set[str] = {
    "functional_status",
    "mental_status",
    "discharge_condition_category",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare refinement vs holdout kappa sidecars.")
    parser.add_argument(
        "--refinement-kappa",
        default="codex_outputs/16c_iter2_kappa.md.json",
    )
    parser.add_argument(
        "--holdout-kappa",
        default="codex_outputs/21_holdout_kappa_report.md.json",
    )
    parser.add_argument(
        "--output",
        default="codex_outputs/21_refinement_vs_holdout_comparison.md",
    )
    return parser.parse_args()


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.4f}"
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None:
        return ""
    return str(value)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        vals = [_format_number(row.get(col, "")).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider, *lines])


def _load_sidecar(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Kappa sidecar not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("kappa_results"), dict):
        raise RuntimeError(f"Invalid sidecar format: {path}")
    return payload


def _field_class(field_key: str) -> str:
    if field_key.startswith("admission_reason_tags::"):
        return "admission_tags"
    if field_key == "dominant_admission_reason":
        return "dominant_admission_reason"
    if field_key in TRISTATE_FIELDS:
        return "tristates"
    if field_key in ENUM_FIELDS:
        return "enums"
    return "other"


def _load_features_by_run_id(run_id: str) -> dict[int, dict[str, Any]]:
    results_path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not results_path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not bool(payload.get("parse_ok")):
            continue
        features = payload.get("features_json")
        if isinstance(features, dict):
            out[int(payload["hadm_id"])] = features
    return out


def _tri_state_base_rate_rows(
    *,
    refinement_sidecar: dict[str, Any],
    holdout_sidecar: dict[str, Any],
) -> list[dict[str, Any]]:
    ref_run_ids = list(refinement_sidecar.get("run_ids", []))
    hold_run_ids = list(holdout_sidecar.get("run_ids", []))
    if len(ref_run_ids) != 3 or len(hold_run_ids) != 3:
        return []

    ref_maps = {
        "A": _load_features_by_run_id(ref_run_ids[0]),
        "B": _load_features_by_run_id(ref_run_ids[1]),
        "C": _load_features_by_run_id(ref_run_ids[2]),
    }
    hold_maps = {
        "A": _load_features_by_run_id(hold_run_ids[0]),
        "B": _load_features_by_run_id(hold_run_ids[1]),
        "C": _load_features_by_run_id(hold_run_ids[2]),
    }

    rows: list[dict[str, Any]] = []
    for field in sorted(TRISTATE_FIELDS):
        for variant in ["A", "B", "C"]:
            ref_features = ref_maps[variant]
            hold_features = hold_maps[variant]
            if not ref_features or not hold_features:
                continue

            def _rate(values: list[str], target: str) -> float:
                if not values:
                    return 0.0
                return sum(1 for value in values if value == target) / len(values)

            ref_vals = [
                str(payload.get(field, "not_documented")) for payload in ref_features.values()
            ]
            hold_vals = [
                str(payload.get(field, "not_documented")) for payload in hold_features.values()
            ]

            ref_yes = _rate(ref_vals, "yes")
            hold_yes = _rate(hold_vals, "yes")
            ref_no = _rate(ref_vals, "no")
            hold_no = _rate(hold_vals, "no")
            ref_nd = _rate(ref_vals, "not_documented")
            hold_nd = _rate(hold_vals, "not_documented")
            rows.append(
                {
                    "field": field,
                    "variant": variant,
                    "ref_yes": _pct(ref_yes),
                    "hold_yes": _pct(hold_yes),
                    "delta_yes_pp": (hold_yes - ref_yes) * 100.0,
                    "ref_no": _pct(ref_no),
                    "hold_no": _pct(hold_no),
                    "delta_no_pp": (hold_no - ref_no) * 100.0,
                    "ref_not_documented": _pct(ref_nd),
                    "hold_not_documented": _pct(hold_nd),
                    "delta_not_documented_pp": (hold_nd - ref_nd) * 100.0,
                }
            )
    rows.sort(key=lambda row: abs(float(row["delta_not_documented_pp"])), reverse=True)
    return rows


def main() -> int:
    args = _parse_args()
    config.load_env()

    refinement_path = Path(args.refinement_kappa)
    holdout_path = Path(args.holdout_kappa)
    output_path = Path(args.output)

    refinement = _load_sidecar(refinement_path)
    holdout = _load_sidecar(holdout_path)

    ref_results = refinement["kappa_results"]
    hold_results = holdout["kappa_results"]

    shared_keys = sorted(set(ref_results.keys()) & set(hold_results.keys()))
    refinement_only = sorted(set(ref_results.keys()) - set(hold_results.keys()))
    holdout_only = sorted(set(hold_results.keys()) - set(ref_results.keys()))

    comparison_rows: list[dict[str, Any]] = []
    for key in shared_keys:
        ref_row = ref_results[key]
        hold_row = hold_results[key]
        comparison_rows.append(
            {
                "field": key,
                "field_class": _field_class(key),
                "kappa_mean_refinement": float(ref_row["kappa_mean"]),
                "kappa_mean_holdout": float(hold_row["kappa_mean"]),
                "delta_kappa_mean": float(hold_row["kappa_mean"]) - float(ref_row["kappa_mean"]),
                "kappa_a_b_refinement": float(ref_row["kappa_A_B"]),
                "kappa_a_b_holdout": float(hold_row["kappa_A_B"]),
                "delta_kappa_a_b": float(hold_row["kappa_A_B"]) - float(ref_row["kappa_A_B"]),
                "kappa_a_c_refinement": float(ref_row["kappa_A_C"]),
                "kappa_a_c_holdout": float(hold_row["kappa_A_C"]),
                "delta_kappa_a_c": float(hold_row["kappa_A_C"]) - float(ref_row["kappa_A_C"]),
                "kappa_b_c_refinement": float(ref_row["kappa_B_C"]),
                "kappa_b_c_holdout": float(hold_row["kappa_B_C"]),
                "delta_kappa_b_c": float(hold_row["kappa_B_C"]) - float(ref_row["kappa_B_C"]),
                "low_base_rate_refinement": bool(ref_row.get("low_base_rate_flag", False)),
                "low_base_rate_holdout": bool(hold_row.get("low_base_rate_flag", False)),
            }
        )

    comparison_rows.sort(
        key=lambda row: abs(float(row["delta_kappa_mean"])),
        reverse=True,
    )

    filtered_rows = [
        row
        for row in comparison_rows
        if (not row["low_base_rate_refinement"]) and (not row["low_base_rate_holdout"])
    ]
    notable_degradation_rows = [
        row for row in filtered_rows if float(row["delta_kappa_mean"]) < -0.05
    ]

    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(np.median(np.asarray(values, dtype=np.float64)))

    class_rows: list[dict[str, Any]] = []
    for class_name in sorted({row["field_class"] for row in filtered_rows}):
        rows = [row for row in filtered_rows if row["field_class"] == class_name]
        class_rows.append(
            {
                "field_class": class_name,
                "n_fields": len(rows),
                "median_refinement_kappa": _median(
                    [float(row["kappa_mean_refinement"]) for row in rows]
                ),
                "median_holdout_kappa": _median(
                    [float(row["kappa_mean_holdout"]) for row in rows]
                ),
                "median_delta_kappa_pp": _median(
                    [float(row["delta_kappa_mean"]) * 100.0 for row in rows]
                ),
            }
        )

    ref_filtered_median = float(
        refinement.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
    )
    hold_filtered_median = float(
        holdout.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
    )
    filtered_delta = hold_filtered_median - ref_filtered_median
    if filtered_delta >= -0.02:
        interpretation = "good generalization"
    elif filtered_delta >= -0.05:
        interpretation = "modest degradation"
    else:
        interpretation = "concerning degradation"

    base_rate_rows = _tri_state_base_rate_rows(
        refinement_sidecar=refinement,
        holdout_sidecar=holdout,
    )

    lines = [
        "# Refinement vs Holdout Kappa Comparison",
        "",
        "## Top-line",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "refinement_filtered_median_kappa": ref_filtered_median,
                    "holdout_filtered_median_kappa": hold_filtered_median,
                    "delta_holdout_minus_refinement_pp": filtered_delta * 100.0,
                    "interpretation": interpretation,
                    "n_shared_fields": len(shared_keys),
                    "n_filtered_fields": len(filtered_rows),
                }
            ],
            [
                "timestamp_utc",
                "refinement_filtered_median_kappa",
                "holdout_filtered_median_kappa",
                "delta_holdout_minus_refinement_pp",
                "interpretation",
                "n_shared_fields",
                "n_filtered_fields",
            ],
        ),
        "",
        "## Per-field-class summary",
        _markdown_table(
            class_rows,
            [
                "field_class",
                "n_fields",
                "median_refinement_kappa",
                "median_holdout_kappa",
                "median_delta_kappa_pp",
            ],
        ),
        "",
        (
            "## Per-field detail (well-supported only: both refinement and holdout "
            "low_base_rate_flag=False)"
        ),
        _markdown_table(
            filtered_rows,
            [
                "field",
                "field_class",
                "kappa_mean_refinement",
                "kappa_mean_holdout",
                "delta_kappa_mean",
                "delta_kappa_a_b",
                "delta_kappa_a_c",
                "delta_kappa_b_c",
            ],
        ),
        "",
        "## Notable degradation flags (delta < -5 pp)",
        _markdown_table(
            notable_degradation_rows,
            [
                "field",
                "field_class",
                "kappa_mean_refinement",
                "kappa_mean_holdout",
                "delta_kappa_mean",
            ],
        ),
        "",
        "## Fields present only in one sidecar",
        _markdown_table(
            [
                {
                    "refinement_only_count": len(refinement_only),
                    "holdout_only_count": len(holdout_only),
                    "refinement_only_sample": ", ".join(refinement_only[:10]),
                    "holdout_only_sample": ", ".join(holdout_only[:10]),
                }
            ],
            [
                "refinement_only_count",
                "holdout_only_count",
                "refinement_only_sample",
                "holdout_only_sample",
            ],
        ),
        "",
        "## Per-variant TriState base-rate stability",
        _markdown_table(
            base_rate_rows,
            [
                "field",
                "variant",
                "ref_yes",
                "hold_yes",
                "delta_yes_pp",
                "ref_no",
                "hold_no",
                "delta_no_pp",
                "ref_not_documented",
                "hold_not_documented",
                "delta_not_documented_pp",
            ],
        ),
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote refinement-vs-holdout comparison report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
