from __future__ import annotations

# Release documentation:
# Runs staged pipeline step `42_production_distributions.py`.
#
# Reads: data/production/parquet/production_v1_features.parquet, data/raw_responses/methodology_5k_a/results.jsonl, codex_outputs/42_production_distributions.md.
# Writes: data/production/parquet/production_v1_features.parquet, data/raw_responses/methodology_5k_a/results.jsonl, codex_outputs/42_production_distributions.md, docs/figures/40_production_tristate_distributions.png, docs/figures/40_production_enum_distributions.png, docs/figures/40_production_tag_prevalence.png.
# Backs production prevalence and distribution claims.
# Usage: `python scripts/42_production_distributions.py` unless the script's argparse help says otherwise.

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Iterable
from typing import Any, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema.fields import (
    DischargeCondition,
    FunctionalStatus,
    MentalStatus,
)
from src.schema.vocabulary import ADMISSION_REASON_TAGS

TRISTATE_FIELDS = [
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
]
TRISTATE_VALUES = ["yes", "no", "not_documented"]

ENUM_FIELD_VALUES: dict[str, list[str]] = {
    "functional_status": list(get_args(FunctionalStatus)),
    "mental_status": list(get_args(MentalStatus)),
    "discharge_condition_category": list(get_args(DischargeCondition)),
}

COUNT_FIELDS = ["new_meds_started_count", "meds_stopped_count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute production distributions and baseline drift."
    )
    parser.add_argument(
        "--production-features",
        default="data/production/parquet/production_v1_features.parquet",
    )
    parser.add_argument(
        "--baseline-results",
        default="data/raw_responses/methodology_5k_a/results.jsonl",
    )
    parser.add_argument("--output", default="codex_outputs/42_production_distributions.md")
    parser.add_argument(
        "--fig-tristate",
        default="docs/figures/40_production_tristate_distributions.png",
    )
    parser.add_argument(
        "--fig-enum",
        default="docs/figures/40_production_enum_distributions.png",
    )
    parser.add_argument(
        "--fig-tags",
        default="docs/figures/40_production_tag_prevalence.png",
    )
    parser.add_argument(
        "--fig-drift",
        default="docs/figures/40_production_baseline_drift.png",
    )
    return parser.parse_args()


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body: list[str] = []
    for row in rows:
        cells = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body])


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + (z**2) / n
    centre = p + (z**2) / (2.0 * n)
    margin = z * np.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))
    low = (centre - margin) / denom
    high = (centre + margin) / denom
    return max(0.0, low), min(1.0, high)


def _load_baseline_frame(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not bool(row.get("parse_ok", False)):
                continue
            features = row.get("features_json")
            if not isinstance(features, dict):
                continue
            rows.append({"hadm_id": int(row.get("hadm_id", 0) or 0), **features})
    return pd.DataFrame(rows)


def _rate_and_ci(series: pd.Series, value: str) -> tuple[int, int, float, float, float]:
    valid = series.dropna().astype(str)
    n = len(valid)
    if n == 0:
        return 0, 0, 0.0, 0.0, 0.0
    k = int((valid == value).sum())
    rate = k / n
    low, high = _wilson_interval(k, n)
    return k, n, rate, low, high


def _pp(value: float) -> str:
    return f"{value * 100:.2f}"


def _load_production_features(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "hadm_id" not in frame.columns:
        raise ValueError("production features parquet missing hadm_id")
    frame = frame.drop_duplicates(subset=["hadm_id"], keep="first").sort_values("hadm_id")
    frame = frame.reset_index(drop=True)
    return frame


def _safe_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def _ensure_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _plot_tristate(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    rows = []
    for field in TRISTATE_FIELDS:
        series = df[field].fillna("not_documented").astype(str)
        total = len(series)
        rows.append(
            {
                "field": field,
                "yes": float((series == "yes").sum() / total),
                "no": float((series == "no").sum() / total),
                "not_documented": float((series == "not_documented").sum() / total),
            }
        )
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(plot_df))
    yes_vals = plot_df["yes"].to_numpy(dtype=float)
    no_vals = plot_df["no"].to_numpy(dtype=float)
    nd_vals = plot_df["not_documented"].to_numpy(dtype=float)

    ax.bar(x, yes_vals, label="yes", color="#1b9e77")
    ax.bar(x, no_vals, bottom=yes_vals, label="no", color="#d95f02")
    ax.bar(x, nd_vals, bottom=yes_vals + no_vals, label="not_documented", color="#7570b3")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["field"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Production TriState distributions")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_enum(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (field, values) in enumerate(ENUM_FIELD_VALUES.items()):
        ax = axes[idx]
        series = df[field].fillna("not_documented").astype(str)
        total = len(series)
        rates = [float((series == value).sum() / total) for value in values]
        ax.bar(values, rates, color="#4e79a7")
        ax.set_title(field)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_ylim(0, max(0.35, max(rates) * 1.2 if rates else 0.35))
        if idx == 0:
            ax.set_ylabel("Rate")
    fig.suptitle("Production enum distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_tags(df: pd.DataFrame, output_path: Path) -> None:
    total = len(df)
    tag_rates: list[tuple[str, float]] = []
    for tag in ADMISSION_REASON_TAGS:
        k = int(
            df["admission_reason_tags"].apply(
                lambda values, current_tag=tag: current_tag in _safe_list(values)
            ).sum()
        )
        tag_rates.append((tag, k / total if total else 0.0))

    dominant_rates: list[tuple[str, float]] = []
    dom_series = df["dominant_admission_reason"].fillna("not_documented").astype(str)
    for tag in ADMISSION_REASON_TAGS:
        dominant_rates.append((tag, float((dom_series == tag).mean())))

    tag_rates = sorted(tag_rates, key=lambda x: x[1], reverse=True)[:20]
    dominant_rates = sorted(dominant_rates, key=lambda x: x[1], reverse=True)[:20]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].barh(
        [name for name, _ in reversed(tag_rates)],
        [v for _, v in reversed(tag_rates)],
        color="#2a9d8f",
    )
    axes[0].set_title("Top 20 admission_reason_tags prevalence")
    axes[0].set_xlabel("Rate")

    axes[1].barh(
        [name for name, _ in reversed(dominant_rates)],
        [v for _, v in reversed(dominant_rates)],
        color="#e76f51",
    )
    axes[1].set_title("Top 20 dominant_admission_reason prevalence")
    axes[1].set_xlabel("Rate")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_drift(drift_rows: list[dict[str, Any]], output_path: Path) -> None:
    filtered = [row for row in drift_rows if row["field"] != "admission_reason_tags"]
    filtered = sorted(filtered, key=lambda row: abs(float(row["delta_pp"])), reverse=True)[:40]
    if not filtered:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No drift rows", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    labels = [f"{row['field']}::{row['value']}" for row in filtered]
    values = [float(row["delta_pp"]) for row in filtered]
    colors = ["#1b9e77" if v >= 0 else "#d95f02" for v in values]

    fig, ax = plt.subplots(figsize=(14, 10))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Delta vs methodology_5k baseline (percentage points)")
    ax.set_title("Production baseline drift (top |delta| rows)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    prod_path = Path(args.production_features)
    baseline_path = Path(args.baseline_results)

    if not prod_path.exists():
        raise FileNotFoundError(f"Missing production features parquet: {prod_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline results jsonl: {baseline_path}")

    prod_df = _load_production_features(prod_path)
    prod_df = prod_df[prod_df["parse_ok"] == True].copy()  # noqa: E712
    base_df = _load_baseline_frame(baseline_path)

    tri_rows: list[dict[str, Any]] = []
    enum_rows: list[dict[str, Any]] = []
    tag_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []

    for field in TRISTATE_FIELDS:
        prod_series = prod_df[field].fillna("not_documented")
        base_series = base_df[field].fillna("not_documented")
        for value in TRISTATE_VALUES:
            k, n, rate, low, high = _rate_and_ci(prod_series, value)
            _, bn, brate, _, _ = _rate_and_ci(base_series, value)
            delta_pp = (rate - brate) * 100.0
            row = {
                "field": field,
                "value": value,
                "count": k,
                "n": n,
                "rate_pct": _pp(rate),
                "ci95_low_pct": _pp(low),
                "ci95_high_pct": _pp(high),
                "baseline_n": bn,
                "baseline_rate_pct": _pp(brate),
                "delta_pp": f"{delta_pp:+.2f}",
            }
            tri_rows.append(row)
            drift_rows.append({"field": field, "value": value, "delta_pp": f"{delta_pp:+.2f}"})

    for field, allowed_values in ENUM_FIELD_VALUES.items():
        prod_series = prod_df[field].fillna("not_documented")
        base_series = base_df[field].fillna("not_documented")
        for value in allowed_values:
            k, n, rate, low, high = _rate_and_ci(prod_series, value)
            _, bn, brate, _, _ = _rate_and_ci(base_series, value)
            delta_pp = (rate - brate) * 100.0
            row = {
                "field": field,
                "value": value,
                "count": k,
                "n": n,
                "rate_pct": _pp(rate),
                "ci95_low_pct": _pp(low),
                "ci95_high_pct": _pp(high),
                "baseline_n": bn,
                "baseline_rate_pct": _pp(brate),
                "delta_pp": f"{delta_pp:+.2f}",
            }
            enum_rows.append(row)
            drift_rows.append({"field": field, "value": value, "delta_pp": f"{delta_pp:+.2f}"})

    prod_tags = prod_df["admission_reason_tags"].apply(_safe_list)
    base_tags = base_df["admission_reason_tags"].apply(_safe_list)
    for tag in ADMISSION_REASON_TAGS:
        k = int(prod_tags.apply(lambda vals, current_tag=tag: current_tag in vals).sum())
        n = len(prod_tags)
        rate = k / n if n else 0.0
        low, high = _wilson_interval(k, n)

        bk = int(base_tags.apply(lambda vals, current_tag=tag: current_tag in vals).sum())
        bn = len(base_tags)
        brate = bk / bn if bn else 0.0

        delta_pp = (rate - brate) * 100.0
        tag_rows.append(
            {
                "field": "admission_reason_tags",
                "value": tag,
                "count": k,
                "n": n,
                "rate_pct": _pp(rate),
                "ci95_low_pct": _pp(low),
                "ci95_high_pct": _pp(high),
                "baseline_n": bn,
                "baseline_rate_pct": _pp(brate),
                "delta_pp": f"{delta_pp:+.2f}",
            }
        )

    prod_dom = prod_df["dominant_admission_reason"].fillna("other").astype(str)
    base_dom = base_df["dominant_admission_reason"].fillna("other").astype(str)
    dominant_rows: list[dict[str, Any]] = []
    for tag in ADMISSION_REASON_TAGS:
        k = int((prod_dom == tag).sum())
        n = len(prod_dom)
        rate = k / n if n else 0.0
        low, high = _wilson_interval(k, n)
        bk = int((base_dom == tag).sum())
        bn = len(base_dom)
        brate = bk / bn if bn else 0.0
        delta_pp = (rate - brate) * 100.0
        dominant_rows.append(
            {
                "field": "dominant_admission_reason",
                "value": tag,
                "count": k,
                "n": n,
                "rate_pct": _pp(rate),
                "ci95_low_pct": _pp(low),
                "ci95_high_pct": _pp(high),
                "baseline_n": bn,
                "baseline_rate_pct": _pp(brate),
                "delta_pp": f"{delta_pp:+.2f}",
            }
        )

    count_rows: list[dict[str, Any]] = []
    for field in COUNT_FIELDS:
        prod_vals = pd.to_numeric(prod_df[field], errors="coerce")
        base_vals = pd.to_numeric(base_df[field], errors="coerce")
        p_non_null = prod_vals.dropna()
        b_non_null = base_vals.dropna()
        count_rows.append(
            {
                "field": field,
                "prod_null_rate_pct": f"{prod_vals.isna().mean() * 100:.2f}",
                "prod_mean": f"{p_non_null.mean():.3f}" if len(p_non_null) else "",
                "prod_median": f"{p_non_null.median():.3f}" if len(p_non_null) else "",
                "prod_p95": (
                    f"{np.percentile(p_non_null.to_numpy(dtype=float), 95):.3f}"
                    if len(p_non_null)
                    else ""
                ),
                "base_null_rate_pct": f"{base_vals.isna().mean() * 100:.2f}",
                "base_mean": f"{b_non_null.mean():.3f}" if len(b_non_null) else "",
                "base_median": f"{b_non_null.median():.3f}" if len(b_non_null) else "",
                "base_p95": (
                    f"{np.percentile(b_non_null.to_numpy(dtype=float), 95):.3f}"
                    if len(b_non_null)
                    else ""
                ),
            }
        )

    drift_top = sorted(drift_rows, key=lambda row: abs(float(row["delta_pp"])), reverse=True)[:25]

    fig_tristate = Path(args.fig_tristate)
    fig_enum = Path(args.fig_enum)
    fig_tags = Path(args.fig_tags)
    fig_drift = Path(args.fig_drift)
    _ensure_paths([fig_tristate, fig_enum, fig_tags, fig_drift, Path(args.output)])

    _plot_tristate(prod_df, fig_tristate)
    _plot_enum(prod_df, fig_enum)
    _plot_tags(prod_df, fig_tags)
    _plot_drift(drift_rows, fig_drift)

    lines = [
        "# Production Distributions",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        "## Dataset sizes",
        _markdown_table(
            [
                {"dataset": "production_v1 (parsed)", "n": len(prod_df)},
                {"dataset": "methodology_5k_a (parsed)", "n": len(base_df)},
            ],
            ["dataset", "n"],
        ),
        "",
        "## TriState distributions (Wilson 95% CI)",
        _markdown_table(
            tri_rows,
            [
                "field",
                "value",
                "count",
                "n",
                "rate_pct",
                "ci95_low_pct",
                "ci95_high_pct",
                "baseline_rate_pct",
                "delta_pp",
            ],
        ),
        "",
        "## Enum distributions (Wilson 95% CI)",
        _markdown_table(
            enum_rows,
            [
                "field",
                "value",
                "count",
                "n",
                "rate_pct",
                "ci95_low_pct",
                "ci95_high_pct",
                "baseline_rate_pct",
                "delta_pp",
            ],
        ),
        "",
        "## admission_reason_tags prevalence (multi-label, Wilson 95% CI)",
        _markdown_table(
            tag_rows,
            [
                "value",
                "count",
                "n",
                "rate_pct",
                "ci95_low_pct",
                "ci95_high_pct",
                "baseline_rate_pct",
                "delta_pp",
            ],
        ),
        "",
        "## dominant_admission_reason distribution",
        _markdown_table(
            dominant_rows,
            [
                "value",
                "count",
                "n",
                "rate_pct",
                "ci95_low_pct",
                "ci95_high_pct",
                "baseline_rate_pct",
                "delta_pp",
            ],
        ),
        "",
        "## Count field summary",
        _markdown_table(
            count_rows,
            [
                "field",
                "prod_null_rate_pct",
                "prod_mean",
                "prod_median",
                "prod_p95",
                "base_null_rate_pct",
                "base_mean",
                "base_median",
                "base_p95",
            ],
        ),
        "",
        "## Top baseline drift rows (absolute delta_pp)",
        _markdown_table(drift_top, ["field", "value", "delta_pp"]),
        "",
        "## Figures",
        f"- `{fig_tristate}`",
        f"- `{fig_enum}`",
        f"- `{fig_tags}`",
        f"- `{fig_drift}`",
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Wrote {fig_tristate}")
    print(f"Wrote {fig_enum}")
    print(f"Wrote {fig_tags}")
    print(f"Wrote {fig_drift}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
