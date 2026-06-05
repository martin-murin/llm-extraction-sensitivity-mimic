"""
Compares three-way kappa across available samples.

Reads: codex_outputs/16c_iter2_kappa.md.json, codex_outputs/21_holdout_kappa_report.md.json, codex_outputs/22_methodology_1k_kappa_report.md.json, codex_outputs/22_kappa_3way_comparison.md.
Writes: codex_outputs/16c_iter2_kappa.md.json, codex_outputs/21_holdout_kappa_report.md.json, codex_outputs/22_methodology_1k_kappa_report.md.json, codex_outputs/22_kappa_3way_comparison.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/22_kappa_3way_comparison.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np


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
ENUM_FIELDS: set[str] = {"functional_status", "mental_status", "discharge_condition_category"}


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


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sidecar: {path}")
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


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare refinement/holdout/methodology_1k kappa.")
    parser.add_argument("--refinement", default="codex_outputs/16c_iter2_kappa.md.json")
    parser.add_argument("--holdout", default="codex_outputs/21_holdout_kappa_report.md.json")
    parser.add_argument(
        "--methodology",
        default="codex_outputs/22_methodology_1k_kappa_report.md.json",
    )
    parser.add_argument("--output", default="codex_outputs/22_kappa_3way_comparison.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refinement = _load(Path(args.refinement))
    holdout = _load(Path(args.holdout))
    methodology = _load(Path(args.methodology))

    ref_results = refinement["kappa_results"]
    hold_results = holdout["kappa_results"]
    meth_results = methodology["kappa_results"]

    shared_keys = sorted(set(ref_results) & set(hold_results) & set(meth_results))
    filtered_shared = [
        key
        for key in shared_keys
        if (
            not bool(ref_results[key].get("low_base_rate_flag", False))
            and not bool(hold_results[key].get("low_base_rate_flag", False))
            and not bool(meth_results[key].get("low_base_rate_flag", False))
        )
    ]

    top_rows: list[dict[str, float | str]] = [
        {
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "refinement_filtered_median": float(
                refinement.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
            ),
            "holdout_filtered_median": float(
                holdout.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
            ),
            "methodology_1k_filtered_median": float(
                methodology.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
            ),
        }
    ]
    meth_median = float(top_rows[0]["methodology_1k_filtered_median"])
    ref_median = float(top_rows[0]["refinement_filtered_median"])
    hold_median = float(top_rows[0]["holdout_filtered_median"])
    top_rows[0]["delta_1k_vs_refinement_pp"] = (meth_median - ref_median) * 100.0
    top_rows[0]["delta_1k_vs_holdout_pp"] = (meth_median - hold_median) * 100.0

    class_names = sorted({_field_class(key) for key in filtered_shared})
    class_rows: list[dict[str, Any]] = []
    for class_name in class_names:
        class_keys = [key for key in filtered_shared if _field_class(key) == class_name]
        ref_med = _median([float(ref_results[key]["kappa_mean"]) for key in class_keys])
        hold_med = _median([float(hold_results[key]["kappa_mean"]) for key in class_keys])
        meth_med = _median([float(meth_results[key]["kappa_mean"]) for key in class_keys])
        class_rows.append(
            {
                "field_class": class_name,
                "n_fields": len(class_keys),
                "refinement_median": f"{ref_med:.4f}",
                "holdout_median": f"{hold_med:.4f}",
                "methodology_1k_median": f"{meth_med:.4f}",
                "delta_1k_vs_refinement_pp": f"{(meth_med - ref_med) * 100.0:.4f}",
                "delta_1k_vs_holdout_pp": f"{(meth_med - hold_med) * 100.0:.4f}",
            }
        )

    regression_rows: list[dict[str, Any]] = []
    for key in filtered_shared:
        ref_k = float(ref_results[key]["kappa_mean"])
        hold_k = float(hold_results[key]["kappa_mean"])
        meth_k = float(meth_results[key]["kappa_mean"])
        delta_ref = (meth_k - ref_k) * 100.0
        delta_hold = (meth_k - hold_k) * 100.0
        if delta_ref < -5.0 or delta_hold < -5.0:
            regression_rows.append(
                {
                    "field": key,
                    "field_class": _field_class(key),
                    "refinement_kappa": f"{ref_k:.4f}",
                    "holdout_kappa": f"{hold_k:.4f}",
                    "methodology_1k_kappa": f"{meth_k:.4f}",
                    "delta_1k_vs_refinement_pp": f"{delta_ref:.4f}",
                    "delta_1k_vs_holdout_pp": f"{delta_hold:.4f}",
                }
            )
    regression_rows.sort(
        key=lambda row: min(
            float(row["delta_1k_vs_refinement_pp"]),
            float(row["delta_1k_vs_holdout_pp"]),
        )
    )

    lines = [
        "# Kappa 3-Way Comparison (Refinement vs Holdout vs Methodology 1k)",
        "",
        "## Top-line filtered medians",
        _markdown_table(
            top_rows,
            [
                "timestamp_utc",
                "refinement_filtered_median",
                "holdout_filtered_median",
                "methodology_1k_filtered_median",
                "delta_1k_vs_refinement_pp",
                "delta_1k_vs_holdout_pp",
            ],
        ),
        "",
        "## Per-field-class summary (filtered shared fields)",
        _markdown_table(
            class_rows,
            [
                "field_class",
                "n_fields",
                "refinement_median",
                "holdout_median",
                "methodology_1k_median",
                "delta_1k_vs_refinement_pp",
                "delta_1k_vs_holdout_pp",
            ],
        ),
        "",
        "## Potential scale-related regressions (1k >5 pp below refinement OR holdout)",
        _markdown_table(
            regression_rows,
            [
                "field",
                "field_class",
                "refinement_kappa",
                "holdout_kappa",
                "methodology_1k_kappa",
                "delta_1k_vs_refinement_pp",
                "delta_1k_vs_holdout_pp",
            ],
        ),
        "",
        "## Coverage",
        _markdown_table(
            [
                {
                    "shared_fields_total": len(shared_keys),
                    "filtered_shared_fields": len(filtered_shared),
                }
            ],
            ["shared_fields_total", "filtered_shared_fields"],
        ),
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote 3-way kappa comparison report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
