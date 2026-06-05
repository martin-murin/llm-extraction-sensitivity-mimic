from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]

TRISTATE_ORDER = ["yes", "no", "not_documented"]
TRISTATE_COLORS = {
    "yes": "#f28e2b",
    "no": "#e15759",
    "not_documented": "#9d9d9d",
}

HEATMAP_COLUMNS = ["kappa_A_B", "kappa_A_C", "kappa_B_C", "kappa_mean"]

OUTLIER_CATEGORIES = [
    ("A is outlier", "#d62728"),
    ("B is outlier", "#ff7f0e"),
    ("C is outlier", "#f2c94c"),
    ("all-different", "#7f7f7f"),
    ("no consistent outlier", "#4e79a7"),
]

TRISTATE_FIELDS = {
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


def _field_class(field_key: str) -> str:
    if field_key.startswith("admission_reason_tags::"):
        return "admission_tags"
    if field_key == "dominant_admission_reason":
        return "dominant_admission_reason"
    enum_fields = {"functional_status", "mental_status", "discharge_condition_category"}
    if field_key in TRISTATE_FIELDS:
        return "tristates"
    if field_key in enum_fields:
        return "enums"
    return "other"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_tristate_baserates(
    extractions: dict[str, list[dict[str, Any]]],
    fields: list[str],
    output_path: Path,
) -> None:
    if not fields:
        raise ValueError("fields must not be empty")
    variants = [variant for variant in ["A", "B", "C"] if variant in extractions]
    if not variants:
        raise ValueError("No variants found in extractions.")

    sns.set_theme(style="whitegrid")
    n_cols = 4
    n_rows = ceil(len(fields) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.0 * n_rows))
    axes_arr = np.atleast_1d(np.asarray(axes, dtype=object)).reshape(n_rows, n_cols)

    for idx, field in enumerate(fields):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes_arr[row, col]
        x = np.arange(len(variants))
        bottoms = np.zeros(len(variants), dtype=np.float64)

        for value in TRISTATE_ORDER:
            heights = []
            for variant in variants:
                payloads = extractions[variant]
                if not payloads:
                    heights.append(0.0)
                    continue
                count = sum(1 for payload in payloads if str(payload.get(field)) == value)
                heights.append((count / len(payloads)) * 100.0)
            heights_arr = np.asarray(heights, dtype=np.float64)
            ax.bar(
                x,
                heights_arr,
                bottom=bottoms,
                color=TRISTATE_COLORS[value],
                edgecolor="white",
                linewidth=0.5,
                label=value,
            )
            bottoms += heights_arr

        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_ylim(0, 100)
        ax.set_title(field, fontsize=10)
        ax.set_ylabel("Percent")

    for idx in range(len(fields), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes_arr[row, col].axis("off")

    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        ncol=1,
        frameon=False,
    )
    fig.suptitle("TriState base rates by variant (methodology_1k)", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 0.88, 0.965))

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pairwise_kappa_heatmap(kappa_data: dict[str, Any], output_path: Path) -> None:
    results = kappa_data.get("kappa_results", {})
    if not isinstance(results, dict) or not results:
        raise ValueError("kappa_data['kappa_results'] must be a non-empty mapping.")

    rows: list[dict[str, Any]] = []
    for field, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        if int(metrics.get("n_positive_total", 0) or 0) < 10:
            continue
        rows.append(
            {
                "field": str(field),
                "kappa_A_B": float(metrics.get("kappa_A_B", 0.0) or 0.0),
                "kappa_A_C": float(metrics.get("kappa_A_C", 0.0) or 0.0),
                "kappa_B_C": float(metrics.get("kappa_B_C", 0.0) or 0.0),
                "kappa_mean": float(metrics.get("kappa_mean", 0.0) or 0.0),
            }
        )

    if not rows:
        raise ValueError("No well-supported fields (n_positive_total >= 10) found.")

    frame = pd.DataFrame(rows).sort_values(by="kappa_mean", ascending=True, kind="mergesort")
    matrix = frame.set_index("field")[HEATMAP_COLUMNS]

    sns.set_theme(style="white")
    fig_height = max(8.0, 0.28 * len(matrix))
    fig, ax = plt.subplots(figsize=(12.8, fig_height))
    sns.heatmap(
        matrix,
        cmap=sns.color_palette("RdYlGn", as_cmap=True),
        vmin=-0.1,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Cohen's kappa", "pad": 0.03},
        ax=ax,
    )
    ax.set_title("Pairwise kappa heatmap (methodology_1k; n_positive_total >= 10)")
    ax.set_xlabel("")
    ax.set_ylabel("Field")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.985, 1))

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_disagreement_outlier_breakdown(
    corpus_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for record in corpus_records:
        field = str(record.get("field", ""))
        target_value = record.get("target_value")
        field_label = field if target_value in (None, "", "null") else f"{field}::{target_value}"
        count = int(record.get("disagreement_count", 0) or 0)
        summary = str(record.get("disagreement_pattern_summary", "")).strip()

        bucket = "no consistent outlier"
        if summary.startswith("A votes "):
            bucket = "A is outlier"
        elif summary.startswith("B votes "):
            bucket = "B is outlier"
        elif summary.startswith("C votes "):
            bucket = "C is outlier"
        elif "all three variants" in summary.lower():
            bucket = "all-different"

        rows.append({"field": field_label, "bucket": bucket, "count": count})

    if not rows:
        raise ValueError("corpus_records did not contain plottable disagreement data.")

    frame = pd.DataFrame(rows)
    top_fields = (
        frame.groupby("field", as_index=False)["count"]
        .sum()
        .sort_values(by="count", ascending=False, kind="mergesort")
        .head(20)["field"]
        .tolist()
    )
    filtered = frame[frame["field"].isin(top_fields)].copy()
    pivot = (
        filtered.pivot_table(
            index="field",
            columns="bucket",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(top_fields)
        .fillna(0)
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15.0, 8.0))
    bottoms = np.zeros(len(pivot), dtype=np.float64)
    x = np.arange(len(pivot))
    for label, color in OUTLIER_CATEGORIES:
        vals = pivot[label].to_numpy(dtype=np.float64) if label in pivot else np.zeros_like(bottoms)
        ax.bar(x, vals, bottom=bottoms, color=color, edgecolor="white", linewidth=0.3, label=label)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=70, ha="right")
    ax.set_ylabel("Disagreement count")
    ax.set_title("Outlier-pattern disagreement breakdown (top 20 fields; methodology_1k)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sample_size_kappa_comparison(
    refinement_kappa: dict[str, Any],
    holdout_kappa: dict[str, Any],
    methodology_1k_kappa: dict[str, Any],
    output_path: Path,
) -> None:
    ref_results = refinement_kappa.get("kappa_results", {})
    hold_results = holdout_kappa.get("kappa_results", {})
    meth_results = methodology_1k_kappa.get("kappa_results", {})
    if (
        not isinstance(ref_results, dict)
        or not isinstance(hold_results, dict)
        or not isinstance(meth_results, dict)
    ):
        raise ValueError("All kappa inputs must include a dict 'kappa_results'.")

    shared = sorted(set(ref_results) & set(hold_results) & set(meth_results))
    if not shared:
        raise ValueError("No shared field keys across refinement/holdout/methodology_1k.")

    filtered = []
    for key in shared:
        if (
            int(ref_results[key].get("n_positive_total", 0) or 0) < 10
            or int(hold_results[key].get("n_positive_total", 0) or 0) < 10
            or int(meth_results[key].get("n_positive_total", 0) or 0) < 10
        ):
            continue
        filtered.append(key)
    if not filtered:
        raise ValueError("No shared well-supported fields across all three datasets.")

    x = np.array([0, 1, 2], dtype=np.int64)
    labels = ["refinement_150", "holdout_150", "methodology_1k"]
    class_colors = {
        "admission_tags": "#4e79a7",
        "tristates": "#f28e2b",
        "enums": "#59a14f",
        "dominant_admission_reason": "#af7aa1",
        "other": "#9c755f",
    }

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    for key in filtered:
        y = np.array(
            [
                float(ref_results[key].get("kappa_mean", 0.0) or 0.0),
                float(hold_results[key].get("kappa_mean", 0.0) or 0.0),
                float(meth_results[key].get("kappa_mean", 0.0) or 0.0),
            ],
            dtype=np.float64,
        )
        cls = _field_class(key)
        ax.plot(
            x,
            y,
            color=class_colors.get(cls, class_colors["other"]),
            alpha=0.30,
            linewidth=1.0,
        )

    median_line = np.array(
        [
            float(np.median([float(ref_results[key]["kappa_mean"]) for key in filtered])),
            float(np.median([float(hold_results[key]["kappa_mean"]) for key in filtered])),
            float(np.median([float(meth_results[key]["kappa_mean"]) for key in filtered])),
        ],
        dtype=np.float64,
    )
    ax.plot(x, median_line, color="black", linewidth=3.0, marker="o", label="Median across fields")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Kappa mean")
    ax.set_title("Kappa trajectories across sample sizes (shared, well-supported fields)")

    legend_handles = [
        plt.Line2D([0], [0], color=color, linewidth=3, label=label)
        for label, color in [
            ("admission_tags", class_colors["admission_tags"]),
            ("tristates", class_colors["tristates"]),
            ("enums", class_colors["enums"]),
            ("dominant_admission_reason", class_colors["dominant_admission_reason"]),
            ("median line", "black"),
        ]
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    fig.tight_layout()

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_kappa_with_bootstrap_ci(
    *,
    field_series: dict[str, dict[str, dict[str, float]]],
    sample_order: list[str],
    sample_labels: list[str],
    output_path: Path,
) -> None:
    if not field_series:
        raise ValueError("field_series must not be empty")
    if len(sample_order) != len(sample_labels):
        raise ValueError("sample_order and sample_labels must have the same length")

    class_colors = {
        "admission_tags": "#4e79a7",
        "tristates": "#f28e2b",
        "enums": "#59a14f",
        "dominant_admission_reason": "#e15759",
        "other": "#9c755f",
    }
    x = np.arange(len(sample_order), dtype=np.int64)

    # Sort for stable rendering order (worst overall agreement first).
    keys = sorted(
        field_series,
        key=lambda key: float(
            np.median(
                [
                    field_series[key].get(sample, {}).get("mean", 0.0)
                    for sample in sample_order
                ]
            )
        ),
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15.5, 9.0))
    for key in keys:
        series = field_series[key]
        cls = _field_class(key)
        color = class_colors.get(cls, class_colors["other"])
        means = np.asarray(
            [float(series[sample]["mean"]) for sample in sample_order],
            dtype=np.float64,
        )
        lowers = np.asarray(
            [float(series[sample]["ci_low"]) for sample in sample_order],
            dtype=np.float64,
        )
        uppers = np.asarray(
            [float(series[sample]["ci_high"]) for sample in sample_order],
            dtype=np.float64,
        )
        ax.plot(x, means, color=color, alpha=0.25, linewidth=1.0)
        ax.fill_between(x, lowers, uppers, color=color, alpha=0.06)

    medians = []
    for sample in sample_order:
        vals = [float(field_series[key][sample]["mean"]) for key in keys]
        medians.append(float(np.median(np.asarray(vals, dtype=np.float64))))
    ax.plot(
        x,
        np.asarray(medians, dtype=np.float64),
        color="black",
        linewidth=3.0,
        marker="o",
        label="Median across fields",
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sample_labels)
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("Kappa (mean with 95% bootstrap CI)")
    ax.set_title("Kappa stability across sample sizes with bootstrap confidence intervals")
    legend_handles = [
        plt.Line2D([0], [0], color=color, linewidth=3, label=label)
        for label, color in [
            ("admission_tags", class_colors["admission_tags"]),
            ("tristates", class_colors["tristates"]),
            ("enums", class_colors["enums"]),
            ("dominant_admission_reason", class_colors["dominant_admission_reason"]),
            ("median", "black"),
        ]
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    fig.tight_layout()

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_snorkel_probability_distributions(
    *,
    predictions: pd.DataFrame,
    icd_target_values: set[str],
    regex_fields: set[str],
    output_path: Path,
) -> None:
    required_cols = {
        "target_field",
        "target_value",
        "snorkel_prob_positive",
        "fit_status",
    }
    if not required_cols.issubset(set(predictions.columns)):
        raise ValueError(
            "predictions must contain target_field, target_value, snorkel_prob_positive, fit_status"
        )

    frame = predictions.copy()
    frame["group"] = "other"
    icd_mask = (frame["target_field"] == "admission_reason_tags") & (
        frame["target_value"].astype(str).isin(icd_target_values)
    )
    regex_mask = frame["target_field"].astype(str).isin(regex_fields)
    single_lf_mask = (
        frame["fit_status"].astype(str).eq("single_lf_only")
        & frame["target_field"].astype(str).isin(TRISTATE_FIELDS)
        & ~regex_mask
    )
    frame.loc[icd_mask, "group"] = "ICD-anchored"
    frame.loc[regex_mask, "group"] = "Regex-anchored"
    frame.loc[single_lf_mask, "group"] = "Single-LF-only fallback"

    groups = [
        ("ICD-anchored", "#4e79a7"),
        ("Regex-anchored", "#f28e2b"),
        ("Single-LF-only fallback", "#59a14f"),
    ]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5), sharey=False)
    for idx, (label, color) in enumerate(groups):
        ax = axes[idx]
        vals = pd.to_numeric(
            frame.loc[frame["group"] == label, "snorkel_prob_positive"],
            errors="coerce",
        ).dropna()
        if vals.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_title(label)
            continue
        ax.hist(
            vals.to_numpy(dtype=np.float64),
            bins=30,
            color=color,
            alpha=0.88,
            edgecolor="white",
        )
        med = float(vals.median())
        mean = float(vals.mean())
        ax.axvline(med, color="black", linestyle="--", linewidth=1.4)
        ax.set_title(label)
        ax.set_xlabel("Snorkel POSITIVE probability")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, 1)
        ax.text(
            0.03,
            0.95,
            f"median={med:.3f}\nmean={mean:.3f}\nn={len(vals):,}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    fig.suptitle("Snorkel probability distributions by target type")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_baserate_stability_grid(
    *,
    rates_by_sample: dict[str, dict[str, dict[str, dict[str, float]]]],
    sample_order: list[str],
    variant_order: list[str],
    fields: list[str],
    output_path: Path,
) -> None:
    if not sample_order or not variant_order or not fields:
        raise ValueError("sample_order, variant_order, and fields must be non-empty")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=len(sample_order),
        ncols=len(variant_order),
        figsize=(20.0, max(10.0, 2.7 * len(sample_order))),
        sharey=True,
    )
    axes_arr = np.atleast_1d(np.asarray(axes, dtype=object)).reshape(
        len(sample_order),
        len(variant_order),
    )
    x = np.arange(len(fields), dtype=np.int64)

    for row_idx, sample in enumerate(sample_order):
        for col_idx, variant in enumerate(variant_order):
            ax = axes_arr[row_idx, col_idx]
            field_rates = rates_by_sample.get(sample, {}).get(variant, {})
            bottoms = np.zeros(len(fields), dtype=np.float64)
            for value in TRISTATE_ORDER:
                vals = np.asarray(
                    [float(field_rates.get(field, {}).get(value, 0.0)) for field in fields],
                    dtype=np.float64,
                )
                ax.bar(
                    x,
                    vals,
                    bottom=bottoms,
                    color=TRISTATE_COLORS[value],
                    edgecolor="white",
                    linewidth=0.25,
                    width=0.8,
                    label=value,
                )
                bottoms += vals

            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            if col_idx == 0:
                ax.set_ylabel(f"{sample}\nproportion")
            if row_idx == len(sample_order) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(fields, rotation=70, ha="right", fontsize=8)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            ax.set_title(f"Variant {variant}")

    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("TriState base-rate stability across four samples")
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
