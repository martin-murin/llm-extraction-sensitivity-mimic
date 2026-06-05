from __future__ import annotations

# Release documentation:
# Legacy builder for paper tables/figures from local aggregate artifacts.
#
# Reads: data/splits/, data/splits/refinement_150.csv, data/splits/holdout_150.csv, data/splits/smoke_200.csv, data/splits/methodology_1k.csv, data/splits/methodology_5k_audit_500.csv.
# Writes: data/splits/, data/splits/refinement_150.csv, data/splits/holdout_150.csv, data/splits/smoke_200.csv, data/splits/methodology_1k.csv, data/splits/methodology_5k_audit_500.csv.
# Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
# Usage: `python scripts/61_prepare_paper_artifacts.py` unless the script's argparse help says otherwise.

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "docs" / "figures" / "paper"
OUT_TABLES = REPO / "codex_outputs" / "61_paper_tables.md"

KAPPA_PATHS = {
    "refinement_150": REPO / "codex_outputs" / "16c_iter2_kappa.md.json",
    "holdout_150": REPO / "codex_outputs" / "21_holdout_kappa_report.md.json",
    "methodology_1k": REPO / "codex_outputs" / "22_methodology_1k_kappa_report.md.json",
    "methodology_5k_audit": REPO / "codex_outputs" / "26_methodology_5k_audit_kappa_report.md.json",
    "extended_5k": REPO / "codex_outputs" / "46_extended_kappa_report.md.json",
}

PAIR_KEYS: list[tuple[str, str]] = [("A", "B"), ("A", "C"), ("B", "C")]
TRISTATE_VALUES = ["yes", "no", "not_documented"]
TAGS_PATH = REPO / "src" / "schema" / "vocabulary.py"


@dataclass(frozen=True)
class SampleSpec:
    name: str
    n: int
    seed: str
    stratified: str
    used_for: str
    csv_path: str | None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _parse_markdown_table_by_columns(md_text: str, required_columns: list[str]) -> pd.DataFrame:
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("| ") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if all(col in header for col in required_columns):
                rows: list[list[str]] = []
                j = i + 2
                while j < len(lines) and lines[j].startswith("| "):
                    rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                    j += 1
                frame = pd.DataFrame(rows, columns=header)
                return frame
        i += 1
    raise ValueError(f"Could not find markdown table with columns: {required_columns}")


def _load_admission_tags() -> list[str]:
    import sys

    sys.path.insert(0, str(REPO))
    from src.schema.vocabulary import ADMISSION_REASON_TAGS

    return list(ADMISSION_REASON_TAGS)


def _class_for_field(field_key: str, field_group: str | None = None) -> str:
    if field_key.startswith("admission_reason_tags::"):
        return "admission_tags"
    if field_key == "dominant_admission_reason":
        return "dominant"
    if field_key in {"functional_status", "mental_status", "discharge_condition_category"}:
        return "enums"
    if field_group == "tristate":
        return "tristates"
    tristates = {
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
    if field_key in tristates:
        return "tristates"
    return "other"


def build_fig_01_methodology_overview() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    boxes = [
        (0.04, 0.72, 0.20, 0.16, "Prompt Variants\nA / B / C"),
        (0.30, 0.72, 0.20, 0.16, "Three-way Kappa\nBaseline"),
        (0.56, 0.72, 0.20, 0.16, "Audit Corpus\nDisagreement Clusters"),
        (0.82, 0.72, 0.14, 0.16, "Phase 3f\nOptimizer Loop"),
        (0.30, 0.44, 0.24, 0.16, "Locked Variant A\nProduction Prompt"),
        (0.60, 0.44, 0.24, 0.16, "332k Production\nExtraction"),
        (0.25, 0.16, 0.22, 0.16, "HEFRA\nFeature Store"),
        (0.55, 0.16, 0.22, 0.16, "Delta DRG\nGap Detection"),
    ]

    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fc="#f8f9fb", ec="#1f2d3d", lw=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, weight="bold")

    arrows = [
        ((0.24, 0.80), (0.30, 0.80)),
        ((0.50, 0.80), (0.56, 0.80)),
        ((0.76, 0.80), (0.82, 0.80)),
        ((0.89, 0.72), (0.46, 0.60)),
        ((0.54, 0.52), (0.60, 0.52)),
        ((0.72, 0.44), (0.66, 0.32)),
        ((0.72, 0.44), (0.36, 0.32)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#1f2d3d"),
        )

    ax.text(
        0.02,
        0.97,
        "Methodology Pipeline: Prompt Diversity, Optimization, and Population-Scale Extraction",
        fontsize=16,
        weight="bold",
        va="top",
    )
    out = FIG_DIR / "paper_fig_01_methodology_overview.png"
    _ensure_parent(out)
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_fig_02_kappa_stability_across_samples() -> None:
    kappa_by_sample = {name: _load_json(path) for name, path in KAPPA_PATHS.items()}
    shared = None
    for payload in kappa_by_sample.values():
        keys = set(payload["kappa_results"].keys())
        shared = keys if shared is None else shared & keys
    assert shared is not None
    shared_keys = sorted(shared)

    filtered_keys = []
    for key in shared_keys:
        ok = True
        for payload in kappa_by_sample.values():
            row = payload["kappa_results"][key]
            if bool(row.get("low_base_rate_flag", True)):
                ok = False
                break
        if ok:
            filtered_keys.append(key)

    x_labels = list(KAPPA_PATHS.keys())
    x = np.arange(len(x_labels), dtype=np.int64)
    class_colors = {
        "admission_tags": "#4e79a7",
        "tristates": "#f28e2b",
        "enums": "#59a14f",
        "dominant": "#e15759",
        "other": "#9c755f",
    }

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 10))

    medians = []
    ci_low = []
    ci_high = []

    rng = np.random.default_rng(42)
    for _idx, sample in enumerate(x_labels):
        vals = np.array(
            [
                float(kappa_by_sample[sample]["kappa_results"][k]["kappa_mean"])
                for k in filtered_keys
            ],
            dtype=np.float64,
        )
        medians.append(float(np.median(vals)))
        boots = []
        n = len(vals)
        for _ in range(500):
            sample_idx = rng.integers(0, n, n)
            boots.append(float(np.median(vals[sample_idx])))
        ci_low.append(float(np.quantile(boots, 0.025)))
        ci_high.append(float(np.quantile(boots, 0.975)))

    for key in filtered_keys:
        cls = _class_for_field(
            key, str(kappa_by_sample[x_labels[0]]["kappa_results"][key].get("field_group", ""))
        )
        y = [float(kappa_by_sample[s]["kappa_results"][key]["kappa_mean"]) for s in x_labels]
        ax.plot(x, y, color=class_colors.get(cls, "#9c755f"), alpha=0.20, linewidth=1.1)

    med_arr = np.array(medians, dtype=np.float64)
    low_arr = np.array(ci_low, dtype=np.float64)
    high_arr = np.array(ci_high, dtype=np.float64)
    ax.fill_between(
        x, low_arr, high_arr, color="black", alpha=0.12, label="Median 95% bootstrap CI"
    )
    ax.plot(x, med_arr, color="black", linewidth=3.0, marker="o", label="Median across fields")

    ax.set_xticks(x)
    ax.set_xticklabels(["refinement", "holdout", "1k", "5k-audit", "extended-5k"], fontsize=11)
    ax.set_ylabel("Kappa", fontsize=12)
    ax.set_title(
        "Kappa Stability Across Samples (well-supported fields)", fontsize=15, weight="bold"
    )
    ax.set_ylim(0.0, 1.0)

    legend_handles = [
        plt.Line2D([0], [0], color=class_colors["admission_tags"], lw=3, label="Admission tags"),
        plt.Line2D([0], [0], color=class_colors["tristates"], lw=3, label="TriStates"),
        plt.Line2D([0], [0], color=class_colors["enums"], lw=3, label="Enums"),
        plt.Line2D([0], [0], color=class_colors["dominant"], lw=3, label="Dominant tag"),
        plt.Line2D([0], [0], color="black", lw=3, label="Median across fields"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    out = FIG_DIR / "paper_fig_02_kappa_stability_across_samples.png"
    _ensure_parent(out)
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_fig_03_field_level_model_size_effect() -> pd.DataFrame:
    text = (REPO / "codex_outputs" / "55_paired_framing_vs_scale.md").read_text(encoding="utf-8")
    table = _parse_markdown_table_by_columns(
        text,
        ["field", "nano_kappa_mean", "gold_kappa_mean", "delta_pp", "low_base_rate_flag"],
    )
    for col in ["nano_kappa_mean", "gold_kappa_mean", "delta_pp"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table = table.sort_values(by="delta_pp", ascending=False, kind="mergesort").reset_index(
        drop=True
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 18))
    y = np.arange(len(table))
    colors = ["#2b6cb0" if d >= 0 else "#c53030" for d in table["delta_pp"]]
    ax.barh(y, table["delta_pp"], color=colors, edgecolor="white", linewidth=0.3)
    median_delta = float(table["delta_pp"].median())
    ax.axvline(
        median_delta,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Median delta ({median_delta:.2f} pp)",
    )
    ax.axvline(0, color="#666", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(table["field"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Nano → GPT-5.4-full kappa delta (pp)")
    ax.set_title(
        "Field-level Model Size Effect (Paired Same-Note Comparison)", fontsize=14, weight="bold"
    )
    ax.legend(loc="lower right", frameon=True)

    out = FIG_DIR / "paper_fig_03_field_level_model_size_effect.png"
    _ensure_parent(out)
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return table


def build_fig_04_disagreement_decomposition() -> None:
    text = (REPO / "codex_outputs" / "30_tristate_soft_vs_hard.md").read_text(encoding="utf-8")
    table = _parse_markdown_table_by_columns(
        text,
        [
            "field",
            "pair",
            "full_agreement",
            "soft_no_not_documented",
            "soft_yes_not_documented",
            "hard_yes_no",
        ],
    )
    for col in [
        "full_agreement",
        "soft_no_not_documented",
        "soft_yes_not_documented",
        "hard_yes_no",
    ]:
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0)
    agg = table.groupby("field", as_index=False)[
        ["full_agreement", "soft_no_not_documented", "soft_yes_not_documented", "hard_yes_no"]
    ].sum()
    totals = agg[
        ["full_agreement", "soft_no_not_documented", "soft_yes_not_documented", "hard_yes_no"]
    ].sum(axis=1)
    for col in [
        "full_agreement",
        "soft_no_not_documented",
        "soft_yes_not_documented",
        "hard_yes_no",
    ]:
        agg[col] = np.where(totals > 0, agg[col] / totals, 0.0)

    agg = agg.sort_values(by="hard_yes_no", ascending=False, kind="mergesort")
    colors = {
        "full_agreement": "#4e79a7",
        "soft_no_not_documented": "#f28e2b",
        "soft_yes_not_documented": "#edc948",
        "hard_yes_no": "#e15759",
    }

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))
    y = np.arange(len(agg))
    left = np.zeros(len(agg), dtype=np.float64)
    for col in [
        "full_agreement",
        "soft_no_not_documented",
        "soft_yes_not_documented",
        "hard_yes_no",
    ]:
        vals = agg[col].to_numpy(dtype=np.float64)
        ax.barh(
            y,
            vals,
            left=left,
            color=colors[col],
            label=col.replace("_", " "),
            edgecolor="white",
            linewidth=0.4,
        )
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(agg["field"], fontsize=9)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Share of pairwise comparisons")
    ax.set_title("TriState disagreement decomposition (soft vs hard)", fontsize=14, weight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)

    out = FIG_DIR / "paper_fig_04_disagreement_decomposition.png"
    _ensure_parent(out)
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _load_variant_features(run_id: str) -> dict[int, dict[str, Any]]:
    rows = _read_jsonl(REPO / "data" / "raw_responses" / run_id / "results.jsonl")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if bool(row.get("parse_ok")) and isinstance(row.get("features_json"), dict):
            out[int(row["hadm_id"])] = dict(row["features_json"])
    return out


def build_fig_05_admission_tag_confusion(tags: list[str]) -> None:
    by_variant = {
        "A": _load_variant_features("methodology_5k_a_subset500"),
        "B": _load_variant_features("methodology_5k_audit_b"),
        "C": _load_variant_features("methodology_5k_audit_c"),
    }
    common = sorted(set(by_variant["A"]) & set(by_variant["B"]) & set(by_variant["C"]))
    tag_idx = {t: i for i, t in enumerate(tags)}

    freq: Counter[str] = Counter()
    for v in ["A", "B", "C"]:
        for hadm in common:
            for t in set(by_variant[v][hadm].get("admission_reason_tags", [])):
                if t in tag_idx:
                    freq[t] += 1
    top_tags = [t for t, _ in freq.most_common(30)]
    if len(top_tags) < 30:
        for t in tags:
            if t not in top_tags:
                top_tags.append(t)
            if len(top_tags) == 30:
                break

    pair_mats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for left, right in PAIR_KEYS:
        confusion = np.zeros((len(tags), len(tags)), dtype=np.int64)
        diag = np.zeros(len(tags), dtype=np.int64)
        for hadm in common:
            lt = set(by_variant[left][hadm].get("admission_reason_tags", []))
            rt = set(by_variant[right][hadm].get("admission_reason_tags", []))
            for t in lt & rt:
                if t in tag_idx:
                    diag[tag_idx[t]] += 1
            left_only = [t for t in (lt - rt) if t in tag_idx]
            right_only = [t for t in (rt - lt) if t in tag_idx]
            for x in left_only:
                ix = tag_idx[x]
                for y in right_only:
                    confusion[ix, tag_idx[y]] += 1
        pair_mats[f"{left}-{right}"] = (confusion, diag)

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
    indices = [tag_idx[t] for t in top_tags]

    for ax, (left, right) in zip(axes, PAIR_KEYS, strict=True):
        confusion, diag = pair_mats[f"{left}-{right}"]
        sub_counts = confusion[np.ix_(indices, indices)].astype(np.float64)
        row_offdiag = sub_counts.sum(axis=1)
        diag_counts = np.array([diag[i] for i in indices], dtype=np.float64)
        denom = row_offdiag + diag_counts

        sub_rates = np.divide(
            sub_counts,
            row_offdiag[:, None],
            out=np.zeros_like(sub_counts),
            where=row_offdiag[:, None] > 0,
        )
        diag_rates = np.divide(diag_counts, denom, out=np.zeros_like(diag_counts), where=denom > 0)
        for i in range(len(indices)):
            sub_rates[i, i] = diag_rates[i]

        nz = sub_rates[sub_rates > 0]
        vmax = float(np.quantile(nz, 0.98)) if nz.size else 1.0
        vmax = max(0.15, min(1.0, vmax))

        hm = sns.heatmap(
            sub_rates,
            ax=ax,
            cmap="YlGnBu",
            vmin=0.0,
            vmax=vmax,
            xticklabels=top_tags,
            yticklabels=top_tags,
            cbar=True,
            cbar_kws={"label": "Confusion rate / diagonal agreement rate"},
            square=True,
            linewidths=0.1,
            linecolor="white",
        )
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(f"{left} vs {right}")
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_xlabel(f"Tag selected by {right} (not {left})")
        ax.set_ylabel(f"Tag selected by {left} (not {right})")

    fig.suptitle("Admission-tag confusion (top 30 tags)", fontsize=15, weight="bold")
    out = FIG_DIR / "paper_fig_05_admission_tag_confusion.png"
    _ensure_parent(out)
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_fig_06_lf_llm_complementarity() -> None:
    text = (REPO / "codex_outputs" / "30_lf_vs_llm_agreement.md").read_text(encoding="utf-8")
    table = _parse_markdown_table_by_columns(
        text,
        [
            "target",
            "group",
            "variant",
            "llm_pos_lf_pos",
            "llm_pos_lf_abstain",
            "llm_nonpos_lf_pos",
            "both_abstain",
        ],
    )
    for col in ["llm_pos_lf_pos", "llm_pos_lf_abstain", "llm_nonpos_lf_pos", "both_abstain"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0)

    table["total"] = table[
        ["llm_pos_lf_pos", "llm_pos_lf_abstain", "llm_nonpos_lf_pos", "both_abstain"]
    ].sum(axis=1)
    for col in ["llm_pos_lf_pos", "llm_pos_lf_abstain", "llm_nonpos_lf_pos", "both_abstain"]:
        table[f"{col}_pct"] = np.where(table["total"] > 0, table[col] / table["total"], 0.0)

    colors = {
        "llm_pos_lf_pos": "#59a14f",
        "llm_pos_lf_abstain": "#f28e2b",
        "llm_nonpos_lf_pos": "#e15759",
        "both_abstain": "#9d9d9d",
    }

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=False)
    variants = ["A", "B", "C"]
    offsets = {"A": -0.23, "B": 0.0, "C": 0.23}
    width = 0.22

    for ax, group in zip(axes, ["icd_anchored", "regex_anchored"], strict=True):
        sub = table[table["group"] == group].copy()
        targets = sorted(sub["target"].unique().tolist())
        x = np.arange(len(targets), dtype=np.float64)

        for variant in variants:
            vv = sub[sub["variant"] == variant]
            bottoms = np.zeros(len(targets), dtype=np.float64)
            for col in [
                "llm_pos_lf_pos",
                "llm_pos_lf_abstain",
                "llm_nonpos_lf_pos",
                "both_abstain",
            ]:
                vals = []
                for t in targets:
                    r = vv[vv["target"] == t]
                    vals.append(float(r.iloc[0][f"{col}_pct"]) if not r.empty else 0.0)
                vals_arr = np.array(vals, dtype=np.float64)
                ax.bar(
                    x + offsets[variant],
                    vals_arr,
                    width=width,
                    bottom=bottoms,
                    color=colors[col],
                    edgecolor="white",
                    linewidth=0.3,
                    alpha={"A": 1.0, "B": 0.8, "C": 0.6}[variant],
                )
                bottoms += vals_arr

        ax.set_title(
            "ICD-anchored targets" if group == "icd_anchored" else "Regex-anchored targets"
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of notes")
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=70, ha="right", fontsize=8)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors]
    legend_labels = [
        "LLM POS + LF POS",
        "LLM POS + LF ABSTAIN",
        "LLM non-POS + LF POS",
        "Both ABSTAIN",
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        "LF-LLM Complementarity by Target Type and Variant", fontsize=15, weight="bold", y=1.05
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = FIG_DIR / "paper_fig_06_lf_llm_complementarity.png"
    _ensure_parent(out)
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_fig_07_optimization_loop_iterations() -> None:
    iterations = []
    for path in sorted((REPO / "logs" / "optimization").glob("iteration_*.json")):
        payload = _load_json(path)
        iterations.append(
            {
                "iter": int(payload.get("iteration", 0)),
                "cluster": payload.get("cluster_targeted", {}).get("cluster_id", ""),
                "before": float(payload.get("cluster_kappa_before", 0.0)),
                "after": float(payload.get("cluster_kappa_after", 0.0)),
                "delta_pp": float(payload.get("cluster_delta_pp", 0.0)),
                "applied": bool(payload.get("applied", False)),
                "volume": int(
                    payload.get("cluster_targeted", {}).get("total_disagreement_count", 0)
                ),
            }
        )

    if not iterations:
        return

    df = pd.DataFrame(iterations).sort_values("iter")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x = np.arange(len(df))
    axes[0].plot(x, df["before"], marker="o", lw=2, color="#e15759", label="Before")
    axes[0].plot(x, df["after"], marker="o", lw=2, color="#59a14f", label="After")
    for i, row in df.iterrows():
        idx = int(i)
        axes[0].annotate(
            f"{row['delta_pp']:+.1f}pp",
            (idx, row["after"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )
    axes[0].set_ylabel("Cluster mean kappa")
    axes[0].set_title("Optimization loop per-iteration cluster kappa change")
    axes[0].legend(loc="lower right")

    axes[1].bar(x, df["volume"], color="#4e79a7", edgecolor="white")
    axes[1].set_ylabel("Targeted cluster disagreement volume")
    axes[1].set_xlabel("Iteration")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Iter {i}" for i in df["iter"]])

    out = FIG_DIR / "paper_fig_07_optimization_loop_iterations.png"
    _ensure_parent(out)
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_fig_08_production_qa_summary() -> None:
    prod = pd.read_parquet(
        REPO / "data" / "production" / "parquet" / "production_v1_features.parquet"
    )
    total = len(prod)
    success = int(prod["parse_ok"].fillna(False).sum())
    fail = total - success

    log_path = REPO / "logs" / "runs" / "production_v1_cost.json"
    times: list[datetime] = []
    costs: list[float] = []
    calls: list[int] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "add_call":
                continue
            ts = datetime.fromisoformat(str(row["ts"]))
            times.append(ts)
            costs.append(float(row.get("total_cost_usd", 0.0)))
            calls.append(int(row.get("n_calls", 0)))

    if times:
        t0 = times[0]
        hrs = np.array([(t - t0).total_seconds() / 3600 for t in times], dtype=np.float64)
        cost_arr = np.array(costs, dtype=np.float64)
        calls_arr = np.array(calls, dtype=np.float64)
        if len(hrs) > 100:
            step = max(1, len(hrs) // 4000)
            hrs = hrs[::step]
            cost_arr = cost_arr[::step]
            calls_arr = calls_arr[::step]
        throughput = np.gradient(calls_arr, hrs, edge_order=1)
    else:
        hrs = np.array([0.0])
        cost_arr = np.array([0.0])
        throughput = np.array([0.0])

    audit_rows = []
    for p in sorted((REPO / "codex_outputs").glob("audit_hook_*.md")):
        txt = p.read_text(encoding="utf-8")
        m_n = re.search(r"Current run: `production_v1` \(n=(\d+) parsed notes\)", txt)
        if not m_n:
            continue
        n = int(m_n.group(1))
        deltas = [
            float(x)
            for x in re.findall(r"\| [^|]+ \| [^|]+ \| [^|]+ \| [^|]+ \| ([+\-]\d+\.\d+) \|", txt)
        ]
        max_abs = max((abs(d) for d in deltas), default=0.0)
        audit_rows.append((n, max_abs))
    audit_rows.sort()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].bar(["parse_ok", "parse_fail"], [success, fail], color=["#59a14f", "#e15759"])
    axes[0, 0].set_title("Parse outcome")
    axes[0, 0].set_ylabel("Notes")
    axes[0, 0].text(
        0.5,
        max(success, 1) * 0.95,
        f"{(success / total) * 100:.2f}% parse success",
        ha="center",
        va="top",
    )

    axes[0, 1].plot(hrs, throughput, color="#4e79a7", lw=1.5)
    axes[0, 1].set_title("Throughput over wall-clock time")
    axes[0, 1].set_xlabel("Hours since start")
    axes[0, 1].set_ylabel("Calls/hour")

    axes[1, 0].plot(hrs, cost_arr, color="#f28e2b", lw=2)
    axes[1, 0].set_title("Cumulative cost")
    axes[1, 0].set_xlabel("Hours since start")
    axes[1, 0].set_ylabel("USD")

    if audit_rows:
        xs = [r[0] for r in audit_rows]
        ys = [r[1] for r in audit_rows]
        axes[1, 1].plot(xs, ys, marker="o", lw=2, color="#af7aa1")
        axes[1, 1].axhline(5.0, color="#666", linestyle="--", linewidth=1)
        axes[1, 1].set_title("Audit-hook max absolute drift (pp)")
        axes[1, 1].set_xlabel("Parsed notes at audit checkpoint")
        axes[1, 1].set_ylabel("Max |delta| pp")
    else:
        axes[1, 1].text(0.5, 0.5, "No audit hook data", ha="center", va="center")
        axes[1, 1].axis("off")

    fig.suptitle("Production QA summary (production_v1)", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = FIG_DIR / "paper_fig_08_production_qa_summary.png"
    _ensure_parent(out)
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _sha_for_file(path: str | None, manifest: dict[str, Any]) -> str:
    checksums = manifest.get("checksums_sha256", {})
    if path is None:
        return "n/a"
    key = Path(path).name
    return str(checksums.get(key, checksums.get(path.replace("data/splits/", ""), "n/a")))


def build_tables(field_delta_table: pd.DataFrame) -> None:
    manifest = _load_json(REPO / "data" / "splits" / "SPLITS_MANIFEST.json")

    sample_specs = [
        SampleSpec(
            "refinement_150",
            150,
            "42",
            "Y",
            "prompt iteration + audit",
            "data/splits/refinement_150.csv",
        ),
        SampleSpec(
            "holdout_150",
            150,
            "42",
            "Y",
            "firewalled single-touch eval",
            "data/splits/holdout_150.csv",
        ),
        SampleSpec(
            "smoke_200", 200, "42", "Y", "early coverage smoke", "data/splits/smoke_200.csv"
        ),
        SampleSpec(
            "methodology_1k",
            1000,
            "43",
            "Y",
            "variant selection + robustness",
            "data/splits/methodology_1k.csv",
        ),
        SampleSpec(
            "methodology_5k_audit_500",
            500,
            "44",
            "Y",
            "pre-production validation",
            "data/splits/methodology_5k_audit_500.csv",
        ),
        SampleSpec(
            "extended_5k",
            5000,
            "46",
            "Y",
            "post-production 3-way validation",
            "data/splits/extended_5k.csv",
        ),
        SampleSpec(
            "gold_1k", 1000, "45", "Y", "full-model proxy reference", "data/splits/gold_1k.csv"
        ),
        SampleSpec("production_v1", 331793, "n/a", "N", "full population extraction", None),
    ]
    table1 = [
        {
            "sample": s.name,
            "n": s.n,
            "seed": s.seed,
            "icd_stratified": s.stratified,
            "used_for": s.used_for,
            "manifest_sha256": _sha_for_file(s.csv_path, manifest),
        }
        for s in sample_specs
    ]

    table2 = []
    for sample, path in KAPPA_PATHS.items():
        payload = _load_json(path)
        summary = payload.get("kappa_summary_filtered", {})
        table2.append(
            {
                "sample": sample,
                "filtered_median_kappa": float(summary.get("overall_median_kappa", 0.0)),
                "n_fields_included": int(summary.get("n_fields_included", 0)),
                "firewalled_usage_note": (
                    "single-touch" if sample == "holdout_150" else "methodology run"
                ),
            }
        )

    table3 = field_delta_table[
        ["field", "nano_kappa_mean", "gold_kappa_mean", "delta_pp", "low_base_rate_flag"]
    ].copy()
    table3 = table3.sort_values(by="delta_pp", ascending=False, kind="mergesort")

    qa_text = (REPO / "codex_outputs" / "40_postrun_qa.md").read_text(encoding="utf-8")
    m_attempted = re.search(r"\| attempted \| (\d+) \|", qa_text)
    m_success = re.search(r"\| parse_ok \| (\d+) \|", qa_text)
    m_parse_fail = re.search(r"\| parse_fail \| (\d+) \|", qa_text)
    m_api = re.search(r"\| api_error \| (\d+) \|", qa_text)
    m_post_retry = re.search(
        r"Post-retry success rate over original production denominator: `([0-9.]+%)`", qa_text
    )
    m_retry_cost = re.search(r"Retry cost: `\$([0-9.]+)`", qa_text)

    log_text = (REPO / "codex_outputs" / "43_phase8_verification.md").read_text(encoding="utf-8")
    m_total_cost = re.search(
        (
            r"\| production_v1 \| [^|]+ \| [^|]+ \| [^|]+ \| [^|]+ \| [^|]+ \| [^|]+ \| "
            r"\$?([0-9.]+) \|"
        ),
        log_text,
    )

    table4 = [
        {
            "metric": "total_notes_attempted",
            "value": int(m_attempted.group(1)) if m_attempted else "n/a",
        },
        {
            "metric": "parse_success_rate",
            "value": f"{(int(m_success.group(1)) / int(m_attempted.group(1)) * 100):.2f}%"
            if (m_success and m_attempted)
            else "n/a",
        },
        {
            "metric": "parse_failures",
            "value": int(m_parse_fail.group(1)) if m_parse_fail else "n/a",
        },
        {
            "metric": "api_errors",
            "value": int(m_api.group(1)) if m_api else "n/a",
        },
        {
            "metric": "total_cost_usd",
            "value": f"{float(m_total_cost.group(1)):.2f}" if m_total_cost else "n/a",
        },
        {
            "metric": "retry_recovery_cost_usd",
            "value": f"{float(m_retry_cost.group(1)):.6f}" if m_retry_cost else "n/a",
        },
        {
            "metric": "post_retry_success_rate",
            "value": m_post_retry.group(1) if m_post_retry else "n/a",
        },
    ]

    icd_text = (REPO / "docs" / "reports" / "icd_lf_audit.md").read_text(encoding="utf-8")
    table5 = _parse_markdown_table_by_columns(
        icd_text,
        [
            "name",
            "target_field",
            "target_value",
            "firing_count",
            "firing_rate_pct",
            "known_issue",
        ],
    )[["name", "target_field", "target_value", "firing_count", "firing_rate_pct", "known_issue"]]

    def md_table_from_df(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        divider = "|" + "|".join(["---"] * len(cols)) + "|"
        body = []
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    if abs(v) >= 100 or c.endswith("_pp"):
                        vals.append(f"{v:.2f}")
                    else:
                        vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v).replace("|", "\\|"))
            body.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, divider, *body])

    sections = [
        "# Paper-ready Summary Tables",
        f"\nGenerated at UTC: {datetime.now(UTC).isoformat()}\n",
        "## Table 1 — Sample design",
        md_table_from_df(pd.DataFrame(table1)),
        "\n## Table 2 — Three-way kappa across samples (filtered)",
        md_table_from_df(pd.DataFrame(table2)),
        "\n## Table 3 — Nano vs full-model kappa per field (paired)",
        md_table_from_df(table3),
        "\n## Table 4 — Production extraction QA",
        md_table_from_df(pd.DataFrame(table4)),
        "\n## Table 5 — ICD LF firing rates and known issues",
        md_table_from_df(table5),
    ]
    _ensure_parent(OUT_TABLES)
    OUT_TABLES.write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> int:
    sns.set_theme(style="whitegrid")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tags = _load_admission_tags()

    build_fig_01_methodology_overview()
    build_fig_02_kappa_stability_across_samples()
    field_delta_table = build_fig_03_field_level_model_size_effect()
    build_fig_04_disagreement_decomposition()
    build_fig_05_admission_tag_confusion(tags)
    build_fig_06_lf_llm_complementarity()
    build_fig_07_optimization_loop_iterations()
    build_fig_08_production_qa_summary()
    build_tables(field_delta_table)

    print("Wrote paper figures to", FIG_DIR)
    print("Wrote tables to", OUT_TABLES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
