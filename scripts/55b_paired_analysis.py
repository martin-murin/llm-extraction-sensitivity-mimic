"""
Analyzes paired model-size extraction outputs.

Reads: codex_outputs/55_paired_accuracy.md, codex_outputs/55_paired_framing_vs_scale.md, codex_outputs/55_verification.md.
Writes: codex_outputs/55_paired_accuracy.md, codex_outputs/55_paired_framing_vs_scale.md, docs/figures/55_field_level_model_size_effect.png, codex_outputs/55_verification.md.
Backs Figure 2 and model-size paired-comparison claims.
Usage: `python scripts/55b_paired_analysis.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from src import config
from src.schema.fields import LLMNoteFeatures
from src.schema.vocabulary import ADMISSION_REASON_TAGS
from src.utils.threeway_kappa import low_base_rate_flag

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
SKIP_FIELDS: set[str] = {
    "reasoning",
    "primary_diagnosis_text",
    "new_meds_started_count",
    "meds_stopped_count",
}


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in rows:
        vals = [str(row.get(col, "")).replace("|", "\\|").replace("\n", " ") for col in columns]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider, *body])


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _safe_kappa(left: list[int], right: list[int]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    if len(set(left)) == 1 and len(set(right)) == 1 and left[0] == right[0]:
        return 1.0
    try:
        value = float(cohen_kappa_score(left, right))
    except Exception:
        return 0.0
    if np.isnan(value):
        return 0.0
    return value


def _encode_tristate(value: str) -> int:
    v = str(value).strip().lower()
    if v == "yes":
        return 1
    if v == "no":
        return -1
    return 0


def _field_names() -> list[str]:
    return [f for f in LLMNoteFeatures.model_fields if f not in SKIP_FIELDS]


def _load_results(run_id: str) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    parsed: dict[int, dict[str, Any]] = {}
    attempted = 0
    input_tokens = 0
    output_tokens = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            attempted += 1
            row = json.loads(line)
            input_tokens += int(row.get("input_tokens", 0) or 0)
            output_tokens += int(row.get("output_tokens", 0) or 0)
            if bool(row.get("parse_ok", False)) and isinstance(row.get("features_json"), dict):
                parsed[int(row["hadm_id"])] = row["features_json"]
    meta_path = config.RAW_RESPONSES_DIR / run_id / "run_metadata.json"
    meta: dict[str, Any] = {
        "run_id": run_id,
        "attempted": attempted,
        "parsed": len(parsed),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost_usd": 0.0,
    }
    if meta_path.exists():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["total_cost_usd"] = float(payload.get("total_cost_usd", 0.0) or 0.0)
    return parsed, meta


def _ids_from_split(path: Path) -> set[int]:
    frame = pd.read_csv(path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").dropna().astype("int64").tolist())


def _paired_accuracy_rows(
    *,
    variant: str,
    gold_parsed: dict[int, dict[str, Any]],
    nano_parsed: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    overlap = sorted(set(gold_parsed) & set(nano_parsed))
    rows: list[dict[str, Any]] = []
    for field in _field_names():
        if field == "admission_reason_tags":
            for tag in ADMISSION_REASON_TAGS:
                left = [
                    1 if tag in list(gold_parsed[h].get("admission_reason_tags", [])) else 0
                    for h in overlap
                ]
                right = [
                    1 if tag in list(nano_parsed[h].get("admission_reason_tags", [])) else 0
                    for h in overlap
                ]
                agree = int(
                    sum(
                        1
                        for left_vote, right_vote in zip(left, right, strict=False)
                        if left_vote == right_vote
                    )
                )
                ci_lo, ci_hi = _wilson_ci(agree, len(overlap))
                rows.append(
                    {
                        "variant": variant.upper(),
                        "field": f"admission_reason_tags::{tag}",
                        "n_overlap": len(overlap),
                        "agreement_pct": float(agree / len(overlap) * 100.0) if overlap else 0.0,
                        "ci95_low_pct": ci_lo * 100.0,
                        "ci95_high_pct": ci_hi * 100.0,
                        "kappa": _safe_kappa(left, right),
                    }
                )
            continue

        if field in TRISTATE_FIELDS:
            left = [
                _encode_tristate(str(gold_parsed[h].get(field, "not_documented"))) for h in overlap
            ]
            right = [
                _encode_tristate(str(nano_parsed[h].get(field, "not_documented"))) for h in overlap
            ]
        else:
            left_vals = [str(gold_parsed[h].get(field, "not_documented")) for h in overlap]
            right_vals = [str(nano_parsed[h].get(field, "not_documented")) for h in overlap]
            cats = sorted(set(left_vals) | set(right_vals))
            mapping = {c: i for i, c in enumerate(cats)}
            left = [mapping[v] for v in left_vals]
            right = [mapping[v] for v in right_vals]

        agree = int(
            sum(
                1
                for left_vote, right_vote in zip(left, right, strict=False)
                if left_vote == right_vote
            )
        )
        ci_lo, ci_hi = _wilson_ci(agree, len(overlap))
        rows.append(
            {
                "variant": variant.upper(),
                "field": field,
                "n_overlap": len(overlap),
                "agreement_pct": float(agree / len(overlap) * 100.0) if overlap else 0.0,
                "ci95_low_pct": ci_lo * 100.0,
                "ci95_high_pct": ci_hi * 100.0,
                "kappa": _safe_kappa(left, right),
            }
        )
    return rows


def _cross_variant_kappas(
    *,
    parsed_a: dict[int, dict[str, Any]],
    parsed_b: dict[int, dict[str, Any]],
    parsed_c: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    overlap = sorted(set(parsed_a) & set(parsed_b) & set(parsed_c))
    if not overlap:
        return []
    rows: list[dict[str, Any]] = []
    for field in _field_names():
        if field == "admission_reason_tags":
            for tag in ADMISSION_REASON_TAGS:
                va = [
                    1 if tag in list(parsed_a[h].get("admission_reason_tags", [])) else 0
                    for h in overlap
                ]
                vb = [
                    1 if tag in list(parsed_b[h].get("admission_reason_tags", [])) else 0
                    for h in overlap
                ]
                vc = [
                    1 if tag in list(parsed_c[h].get("admission_reason_tags", [])) else 0
                    for h in overlap
                ]
                k_ab = _safe_kappa(va, vb)
                k_ac = _safe_kappa(va, vc)
                k_bc = _safe_kappa(vb, vc)
                n_positive_total = int(sum(va) + sum(vb) + sum(vc))
                rows.append(
                    {
                        "field": f"admission_reason_tags::{tag}",
                        "kappa_A_B": k_ab,
                        "kappa_A_C": k_ac,
                        "kappa_B_C": k_bc,
                        "kappa_mean": float(np.mean([k_ab, k_ac, k_bc])),
                        "n_positive_total": n_positive_total,
                        "low_base_rate_flag": bool(low_base_rate_flag(n_positive_total)),
                    }
                )
            continue

        if field in TRISTATE_FIELDS:
            va = [_encode_tristate(str(parsed_a[h].get(field, "not_documented"))) for h in overlap]
            vb = [_encode_tristate(str(parsed_b[h].get(field, "not_documented"))) for h in overlap]
            vc = [_encode_tristate(str(parsed_c[h].get(field, "not_documented"))) for h in overlap]
            n_positive_total = int(
                sum(1 for v in va if v == 1)
                + sum(1 for v in vb if v == 1)
                + sum(1 for v in vc if v == 1)
            )
        else:
            values = (
                {str(parsed_a[h].get(field, "not_documented")) for h in overlap}
                | {str(parsed_b[h].get(field, "not_documented")) for h in overlap}
                | {str(parsed_c[h].get(field, "not_documented")) for h in overlap}
            )
            mapping = {c: i for i, c in enumerate(sorted(values))}
            va = [mapping[str(parsed_a[h].get(field, "not_documented"))] for h in overlap]
            vb = [mapping[str(parsed_b[h].get(field, "not_documented"))] for h in overlap]
            vc = [mapping[str(parsed_c[h].get(field, "not_documented"))] for h in overlap]
            if field in ENUM_FIELDS:
                n_positive_total = int(
                    sum(
                        1
                        for h in overlap
                        if str(parsed_a[h].get(field, "not_documented")) != "not_documented"
                    )
                    + sum(
                        1
                        for h in overlap
                        if str(parsed_b[h].get(field, "not_documented")) != "not_documented"
                    )
                    + sum(
                        1
                        for h in overlap
                        if str(parsed_c[h].get(field, "not_documented")) != "not_documented"
                    )
                )
            else:
                n_positive_total = 0

        k_ab = _safe_kappa(va, vb)
        k_ac = _safe_kappa(va, vc)
        k_bc = _safe_kappa(vb, vc)
        rows.append(
            {
                "field": field,
                "kappa_A_B": k_ab,
                "kappa_A_C": k_ac,
                "kappa_B_C": k_bc,
                "kappa_mean": float(np.mean([k_ab, k_ac, k_bc])),
                "n_positive_total": n_positive_total,
                "low_base_rate_flag": bool(low_base_rate_flag(n_positive_total)),
            }
        )
    return rows


def _merge_variant_runs(
    first: dict[int, dict[str, Any]], second: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    overlap = set(first) & set(second)
    if overlap:
        raise RuntimeError(f"Unexpected hadm_id overlap while merging split runs: n={len(overlap)}")
    merged = dict(first)
    merged.update(second)
    return merged


def _median_filtered(rows: list[dict[str, Any]]) -> float:
    values = [float(r["kappa_mean"]) for r in rows if not bool(r["low_base_rate_flag"])]
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _write_accuracy_report(rows: list[dict[str, Any]], output: Path) -> None:
    ordered = sorted(rows, key=lambda r: (str(r["variant"]), float(r["kappa"])))
    out_rows = []
    for r in ordered:
        out_rows.append(
            {
                "variant": r["variant"],
                "field": r["field"],
                "n_overlap": r["n_overlap"],
                "agreement_pct": f"{float(r['agreement_pct']):.2f}",
                "ci95": f"[{float(r['ci95_low_pct']):.2f}, {float(r['ci95_high_pct']):.2f}]",
                "kappa_gold_vs_nano": f"{float(r['kappa']):.4f}",
            }
        )
    lines = [
        "# Prompt 27 Paired Accuracy (Same-note Gold vs Nano)",
        "",
        "Rows are sorted by per-variant gold-vs-nano kappa ascending.",
        "",
        _markdown_table(
            out_rows,
            ["variant", "field", "n_overlap", "agreement_pct", "ci95", "kappa_gold_vs_nano"],
        ),
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _write_framing_report(
    *,
    nano_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    nano_by_field = {str(r["field"]): r for r in nano_rows}
    gold_by_field = {str(r["field"]): r for r in gold_rows}
    shared = sorted(set(nano_by_field) & set(gold_by_field))

    rows: list[dict[str, Any]] = []
    for field in shared:
        n = float(nano_by_field[field]["kappa_mean"])
        g = float(gold_by_field[field]["kappa_mean"])
        rows.append(
            {
                "field": field,
                "nano_kappa_mean": f"{n:.4f}",
                "gold_kappa_mean": f"{g:.4f}",
                "delta_pp": f"{(g - n) * 100.0:.2f}",
                "low_base_rate_flag": bool(gold_by_field[field]["low_base_rate_flag"]),
            }
        )
    rows.sort(key=lambda r: float(r["delta_pp"]), reverse=True)

    topline: list[dict[str, Any]] = [
        {
            "metric": "median_kappa_filtered_nano",
            "value": f"{_median_filtered(nano_rows):.4f}",
        },
        {
            "metric": "median_kappa_filtered_gold",
            "value": f"{_median_filtered(gold_rows):.4f}",
        },
        {
            "metric": "delta_filtered_median_pp",
            "value": f"{(_median_filtered(gold_rows) - _median_filtered(nano_rows)) * 100.0:.2f}",
        },
        {"metric": "n_shared_fields", "value": str(len(shared))},
    ]

    lines = [
        "# Prompt 27 Paired Framing-vs-Scale",
        "",
        "## Top-line",
        _markdown_table(topline, ["metric", "value"]),
        "",
        "## Field-level nano vs gold kappa mean",
        _markdown_table(
            rows,
            ["field", "nano_kappa_mean", "gold_kappa_mean", "delta_pp", "low_base_rate_flag"],
        ),
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(
    *, nano_rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]], output: Path
) -> None:
    nano_by = {str(r["field"]): float(r["kappa_mean"]) for r in nano_rows}
    gold_by = {str(r["field"]): float(r["kappa_mean"]) for r in gold_rows}
    shared = sorted(set(nano_by) & set(gold_by))
    if not shared:
        raise RuntimeError("No shared fields to plot.")
    deltas = [(f, (gold_by[f] - nano_by[f]) * 100.0) for f in shared]
    deltas.sort(key=lambda x: x[1], reverse=True)
    labels = [d[0] for d in deltas]
    values = [d[1] for d in deltas]

    fig_h = max(8.0, len(labels) * 0.22)
    plt.figure(figsize=(11.5, fig_h))
    colors = ["#2f7ed8" if v >= 0 else "#c43d3d" for v in values]
    y = np.arange(len(labels))
    plt.barh(y, values, color=colors)
    plt.yticks(y, labels, fontsize=8)
    plt.gca().invert_yaxis()
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Kappa mean delta (gold - nano), percentage points")
    plt.title("Field-level Model-Size Effect (Same-note, Cross-variant Kappa)")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Prompt 27 paired full-model runs.")
    parser.add_argument("--output-accuracy", default="codex_outputs/55_paired_accuracy.md")
    parser.add_argument("--output-framing", default="codex_outputs/55_paired_framing_vs_scale.md")
    parser.add_argument(
        "--output-figure",
        default="docs/figures/55_field_level_model_size_effect.png",
    )
    parser.add_argument("--output-verification", default="codex_outputs/55_verification.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    split_1k_ids = _ids_from_split(config.SPLITS_DIR / "methodology_1k.csv")
    split_500_ids = _ids_from_split(config.SPLITS_DIR / "methodology_5k_audit_500.csv")

    gold_1k = {
        "a": _load_results("paired_gold_methodology_1k_a")[0],
        "b": _load_results("paired_gold_methodology_1k_b")[0],
        "c": _load_results("paired_gold_methodology_1k_c")[0],
    }
    gold_500 = {
        "a": _load_results("paired_gold_methodology_5k_audit_a")[0],
        "b": _load_results("paired_gold_methodology_5k_audit_b")[0],
        "c": _load_results("paired_gold_methodology_5k_audit_c")[0],
    }
    gold_meta = {
        "paired_gold_methodology_1k_a": _load_results("paired_gold_methodology_1k_a")[1],
        "paired_gold_methodology_1k_b": _load_results("paired_gold_methodology_1k_b")[1],
        "paired_gold_methodology_1k_c": _load_results("paired_gold_methodology_1k_c")[1],
        "paired_gold_methodology_5k_audit_a": _load_results("paired_gold_methodology_5k_audit_a")[
            1
        ],
        "paired_gold_methodology_5k_audit_b": _load_results("paired_gold_methodology_5k_audit_b")[
            1
        ],
        "paired_gold_methodology_5k_audit_c": _load_results("paired_gold_methodology_5k_audit_c")[
            1
        ],
    }

    nano_1k = {
        "a": _load_results("methodology_1k_a")[0],
        "b": _load_results("methodology_1k_b")[0],
        "c": _load_results("methodology_1k_c")[0],
    }
    nano_500 = {
        "a": _load_results("production_v1")[0],
        "b": _load_results("methodology_5k_audit_b")[0],
        "c": _load_results("methodology_5k_audit_c")[0],
    }

    # Restrict each parsed map to relevant split ids.
    for variant in ["a", "b", "c"]:
        gold_1k[variant] = {k: v for k, v in gold_1k[variant].items() if k in split_1k_ids}
        gold_500[variant] = {k: v for k, v in gold_500[variant].items() if k in split_500_ids}
        nano_1k[variant] = {k: v for k, v in nano_1k[variant].items() if k in split_1k_ids}
        nano_500[variant] = {k: v for k, v in nano_500[variant].items() if k in split_500_ids}

    accuracy_rows: list[dict[str, Any]] = []
    for variant in ["a", "b", "c"]:
        gold_combined = _merge_variant_runs(gold_1k[variant], gold_500[variant])
        nano_combined = _merge_variant_runs(nano_1k[variant], nano_500[variant])
        accuracy_rows.extend(
            _paired_accuracy_rows(
                variant=variant,
                gold_parsed=gold_combined,
                nano_parsed=nano_combined,
            )
        )

    gold_cross_rows = _cross_variant_kappas(
        parsed_a=_merge_variant_runs(gold_1k["a"], gold_500["a"]),
        parsed_b=_merge_variant_runs(gold_1k["b"], gold_500["b"]),
        parsed_c=_merge_variant_runs(gold_1k["c"], gold_500["c"]),
    )
    nano_cross_rows = _cross_variant_kappas(
        parsed_a=_merge_variant_runs(nano_1k["a"], nano_500["a"]),
        parsed_b=_merge_variant_runs(nano_1k["b"], nano_500["b"]),
        parsed_c=_merge_variant_runs(nano_1k["c"], nano_500["c"]),
    )

    _write_accuracy_report(accuracy_rows, Path(args.output_accuracy))
    _write_framing_report(
        nano_rows=nano_cross_rows, gold_rows=gold_cross_rows, output=Path(args.output_framing)
    )
    _write_plot(
        nano_rows=nano_cross_rows, gold_rows=gold_cross_rows, output=Path(args.output_figure)
    )

    total_cost = sum(float(meta["total_cost_usd"]) for meta in gold_meta.values())
    complete = all(int(meta["attempted"]) == int(meta["parsed"]) for meta in gold_meta.values())
    verification_rows: list[dict[str, Any]] = [
        {"metric": "timestamp_utc", "value": datetime.now(tz=UTC).isoformat()},
        {"metric": "total_gold_runs", "value": len(gold_meta)},
        {"metric": "all_gold_runs_parse_complete", "value": complete},
        {"metric": "gold_total_cost_usd", "value": f"{total_cost:.6f}"},
        {"metric": "paired_accuracy_report", "value": args.output_accuracy},
        {"metric": "paired_framing_report", "value": args.output_framing},
        {"metric": "field_level_plot", "value": args.output_figure},
        {"metric": "status", "value": "PASS" if complete else "FAIL"},
    ]
    Path(args.output_verification).write_text(
        "\n".join(
            [
                "# Prompt 27 Agent 1 Verification",
                "",
                _markdown_table(verification_rows, ["metric", "value"]),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
