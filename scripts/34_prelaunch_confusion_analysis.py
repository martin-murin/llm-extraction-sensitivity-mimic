from __future__ import annotations

# Release documentation:
# Runs staged pipeline step `34_prelaunch_confusion_analysis.py`.
#
# Reads: codex_outputs/26_methodology_5k_audit_kappa_report.md.json, data/optimization/audit_corpus_methodology_5k_audit.jsonl, data/splits/methodology_5k_audit_500.csv, codex_outputs/30_tristate_soft_vs_hard.md, codex_outputs/30_admission_tag_confusion.md, codex_outputs/30_enum_confusion.md.
# Writes: codex_outputs/26_methodology_5k_audit_kappa_report.md.json, data/optimization/audit_corpus_methodology_5k_audit.jsonl, data/splits/methodology_5k_audit_500.csv, docs/figures/30_tristate_disagreement_decomposition.png, codex_outputs/30_tristate_soft_vs_hard.md, docs/figures/30_admission_tag_confusion_grid.png.
# Backs dominant-admission and enum/tag confusion figures.
# Usage: `python scripts/34_prelaunch_confusion_analysis.py` unless the script's argparse help says otherwise.

import argparse
import json
import sys
from collections import defaultdict
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.db.connection import get_engine
from src.db.queries import (
    fetch_icd_codes_by_hadm_ids,
    fetch_notes_by_hadm_ids,
    fetch_primary_icd_by_hadm_ids,
)
from src.labeling_functions.base import LFInput, Vote
from src.labeling_functions.icd_lf import build_all_icd_lfs
from src.labeling_functions.regex_lf import build_all_regex_lfs
from src.labeling_functions.section_parser import parse_sections
from src.schema.vocabulary import ADMISSION_REASON_TAGS

PAIR_KEYS: list[tuple[str, str]] = [("A", "B"), ("A", "C"), ("B", "C")]
TRISTATE_VALUES = ["yes", "no", "not_documented"]
TRISTATE_INDEX = {value: idx for idx, value in enumerate(TRISTATE_VALUES)}
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
ENUM_FIELDS = ["discharge_condition_category", "mental_status", "functional_status"]
DOMINANT_FIELD = "dominant_admission_reason"

CATEGORY_ORDER = [
    "full_agreement",
    "soft_no_not_documented",
    "soft_yes_not_documented",
    "hard_yes_no",
]
CATEGORY_COLORS = {
    "full_agreement": "#4e79a7",
    "soft_no_not_documented": "#f28e2b",
    "soft_yes_not_documented": "#edc948",
    "hard_yes_no": "#e15759",
}

LF_AGREEMENT_CATEGORIES = [
    "llm_pos_lf_pos",
    "llm_pos_lf_abstain",
    "llm_nonpos_lf_pos",
    "both_abstain",
]
LF_AGREEMENT_COLORS = {
    "llm_pos_lf_pos": "#59a14f",
    "llm_pos_lf_abstain": "#f28e2b",
    "llm_nonpos_lf_pos": "#e15759",
    "both_abstain": "#9d9d9d",
}


@dataclass(frozen=True)
class TargetKey:
    field: str
    value: str

    @property
    def as_text(self) -> str:
        return f"{self.field}::{self.value}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-launch confusion matrix analysis on existing 5k data"
    )
    parser.add_argument(
        "--run-ids",
        nargs=3,
        default=["methodology_5k_a_subset500", "methodology_5k_audit_b", "methodology_5k_audit_c"],
    )
    parser.add_argument(
        "--kappa-json",
        default="codex_outputs/26_methodology_5k_audit_kappa_report.md.json",
    )
    parser.add_argument(
        "--audit-corpus",
        default="data/optimization/audit_corpus_methodology_5k_audit.jsonl",
    )
    parser.add_argument(
        "--split-csv",
        default="data/splits/methodology_5k_audit_500.csv",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_variant_results(
    run_id: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    records = _read_jsonl(path)
    parsed: dict[int, dict[str, Any]] = {}
    full: dict[int, dict[str, Any]] = {}
    for record in records:
        hadm_id = int(record["hadm_id"])
        full[hadm_id] = record
        if bool(record.get("parse_ok")) and isinstance(record.get("features_json"), dict):
            parsed[hadm_id] = dict(record["features_json"])
    return parsed, full


def _intersect_hadm_ids(by_variant: dict[str, dict[int, dict[str, Any]]]) -> list[int]:
    key_sets = [set(items.keys()) for items in by_variant.values()]
    if not key_sets:
        return []
    return sorted(set.intersection(*key_sets))


def _is_low_base_rate(kappa_results: dict[str, Any], field: str) -> bool:
    row = kappa_results.get(field)
    if not isinstance(row, dict):
        return True
    return bool(row.get("low_base_rate_flag", True))


def _tristate_matrix(values_left: list[str], values_right: list[str]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for left, right in zip(values_left, values_right, strict=True):
        if left not in TRISTATE_INDEX or right not in TRISTATE_INDEX:
            continue
        matrix[TRISTATE_INDEX[left], TRISTATE_INDEX[right]] += 1
    return matrix


def _disagreement_category(left: str, right: str) -> str:
    if left == right:
        return "full_agreement"
    pair = {left, right}
    if pair == {"no", "not_documented"}:
        return "soft_no_not_documented"
    if pair == {"yes", "not_documented"}:
        return "soft_yes_not_documented"
    if pair == {"yes", "no"}:
        return "hard_yes_no"
    return "full_agreement"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body: list[str] = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|").replace("\n", " "))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body])


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _plot_tristate_decomposition(
    field_totals: dict[str, dict[str, int]],
    out_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for field, counts in field_totals.items():
        total = sum(counts.get(category, 0) for category in CATEGORY_ORDER)
        if total == 0:
            continue
        rows.append(
            {
                "field": field,
                **{category: counts.get(category, 0) / total for category in CATEGORY_ORDER},
            }
        )

    if not rows:
        return

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(by="hard_yes_no", ascending=False, kind="mergesort")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(frame))))
    y = np.arange(len(frame))
    left = np.zeros(len(frame), dtype=np.float64)

    for category in CATEGORY_ORDER:
        vals = frame[category].to_numpy(dtype=np.float64)
        ax.barh(
            y,
            vals,
            left=left,
            color=CATEGORY_COLORS[category],
            edgecolor="white",
            linewidth=0.4,
            label=category,
        )
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(frame["field"].tolist())
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of pairwise comparisons")
    ax.set_title("TriState disagreement decomposition (aggregated across A-B, A-C, B-C)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    fig.tight_layout(rect=(0, 0, 0.85, 1))

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_tag_confusions(
    by_variant: dict[str, dict[int, dict[str, Any]]],
    hadm_ids: list[int],
) -> tuple[dict[str, Any], list[str]]:
    tag_index = {tag: idx for idx, tag in enumerate(ADMISSION_REASON_TAGS)}
    n_tags = len(ADMISSION_REASON_TAGS)
    frequency: defaultdict[str, int] = defaultdict(int)

    for variant in ["A", "B", "C"]:
        for hadm_id in hadm_ids:
            tags = set(by_variant[variant][hadm_id].get("admission_reason_tags", []))
            for tag in tags:
                if tag in tag_index:
                    frequency[tag] += 1

    ranked_tags = sorted(ADMISSION_REASON_TAGS, key=lambda t: (-frequency[t], t))
    top_tags = ranked_tags[:30]

    pair_data: dict[str, Any] = {}
    for left, right in PAIR_KEYS:
        confusion = np.zeros((n_tags, n_tags), dtype=np.int64)
        diagonal_agreement = np.zeros(n_tags, dtype=np.int64)
        for hadm_id in hadm_ids:
            left_tags = set(by_variant[left][hadm_id].get("admission_reason_tags", []))
            right_tags = set(by_variant[right][hadm_id].get("admission_reason_tags", []))

            for tag in left_tags.intersection(right_tags):
                if tag in tag_index:
                    diagonal_agreement[tag_index[tag]] += 1

            left_only = [tag for tag in left_tags.difference(right_tags) if tag in tag_index]
            right_only = [tag for tag in right_tags.difference(left_tags) if tag in tag_index]
            for tag_x in left_only:
                idx_x = tag_index[tag_x]
                for tag_y in right_only:
                    confusion[idx_x, tag_index[tag_y]] += 1

        row_sums = confusion.sum(axis=1, keepdims=True)
        rates = np.divide(
            confusion,
            row_sums,
            out=np.zeros_like(confusion, dtype=np.float64),
            where=row_sums > 0,
        )

        offdiag_pairs: list[tuple[str, str, int, float]] = []
        for i, tag_x in enumerate(ADMISSION_REASON_TAGS):
            denom = int(row_sums[i, 0])
            if denom == 0:
                continue
            for j, tag_y in enumerate(ADMISSION_REASON_TAGS):
                count = int(confusion[i, j])
                if count <= 0:
                    continue
                offdiag_pairs.append((tag_x, tag_y, count, count / denom))
        offdiag_pairs.sort(key=lambda row: (-row[2], row[0], row[1]))

        pair_key = f"{left}-{right}"
        pair_data[pair_key] = {
            "confusion_counts": confusion,
            "confusion_rates": rates,
            "diagonal_agreement": diagonal_agreement,
            "top_pairs": offdiag_pairs[:10],
            "total_diag": int(diagonal_agreement.sum()),
            "total_offdiag": int(confusion.sum()),
        }

    return pair_data, top_tags


def _semantic_confusion_label(tag_x: str, tag_y: str) -> str:
    if tag_x == tag_y:
        return "agreement"
    domain_tokens = {
        "cardiac",
        "respiratory",
        "infection",
        "sepsis",
        "neuro",
        "gi",
        "renal",
        "gu",
        "metabolic",
        "oncology",
        "hepatic",
        "trauma",
        "substance",
        "psych",
        "symptom",
    }
    left_tokens = set(tag_x.split("_"))
    right_tokens = set(tag_y.split("_"))
    if left_tokens.intersection(right_tokens):
        return "semantic_likely"
    if left_tokens.intersection(domain_tokens) and right_tokens.intersection(domain_tokens):
        left_domain = next(iter(left_tokens.intersection(domain_tokens)))
        right_domain = next(iter(right_tokens.intersection(domain_tokens)))
        if left_domain == right_domain:
            return "semantic_likely"
    return "noise_or_cross-domain"


def _plot_admission_tag_confusion(
    pair_data: dict[str, Any],
    top_tags: list[str],
    out_path: Path,
) -> None:
    tag_index = {tag: idx for idx, tag in enumerate(ADMISSION_REASON_TAGS)}
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(24, 9), constrained_layout=True)

    for ax, (left, right) in zip(axes, PAIR_KEYS, strict=True):
        key = f"{left}-{right}"
        rates = pair_data[key]["confusion_rates"]
        diag = pair_data[key]["diagonal_agreement"]
        counts = pair_data[key]["confusion_counts"]

        indices = [tag_index[tag] for tag in top_tags]
        sub = rates[np.ix_(indices, indices)].copy()
        sub_counts = counts[np.ix_(indices, indices)]
        sub_diag_counts = np.array([int(diag[idx]) for idx in indices], dtype=np.float64)
        row_offdiag = sub_counts.sum(axis=1).astype(np.float64)
        denom = row_offdiag + sub_diag_counts
        diag_rates = np.divide(
            sub_diag_counts,
            denom,
            out=np.zeros_like(sub_diag_counts),
            where=denom > 0,
        )
        for i in range(len(indices)):
            sub[i, i] = diag_rates[i]

        finite_vals = sub[np.isfinite(sub)]
        nonzero_vals = finite_vals[finite_vals > 0]
        vmax = 0.35
        if nonzero_vals.size > 0:
            vmax = float(np.quantile(nonzero_vals, 0.95))
            vmax = max(0.15, min(1.0, vmax))

        sns.heatmap(
            sub,
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
        ax.set_title(f"Admission-tag confusion rates ({left} vs {right})")
        ax.set_xlabel(f"Tag selected by {right} (and not by {left})")
        ax.set_ylabel(f"Tag selected by {left} (and not by {right})")
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)

        for i, tag in enumerate(top_tags):
            idx = tag_index[tag]
            ax.text(
                i + 0.5,
                i + 0.5,
                str(int(diag[idx])),
                ha="center",
                va="center",
                color="black",
                fontsize=5,
                fontweight="bold",
            )

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _build_enum_confusion(
    by_variant: dict[str, dict[int, dict[str, Any]]],
    hadm_ids: list[int],
    field: str,
) -> dict[str, Any]:
    values = sorted(
        {
            str(by_variant[variant][hadm_id].get(field))
            for variant in ["A", "B", "C"]
            for hadm_id in hadm_ids
        }
    )
    value_index = {value: idx for idx, value in enumerate(values)}

    pair_mats: dict[str, np.ndarray] = {}
    pair_summaries: dict[str, dict[str, Any]] = {}
    for left, right in PAIR_KEYS:
        matrix = np.zeros((len(values), len(values)), dtype=np.int64)
        for hadm_id in hadm_ids:
            left_value = str(by_variant[left][hadm_id].get(field))
            right_value = str(by_variant[right][hadm_id].get(field))
            matrix[value_index[left_value], value_index[right_value]] += 1

        total = int(matrix.sum())
        diag_mass = (float(np.trace(matrix)) / total) if total else 0.0
        offdiag = matrix.copy()
        np.fill_diagonal(offdiag, 0)
        max_idx = np.unravel_index(np.argmax(offdiag), offdiag.shape) if offdiag.size else (0, 0)
        max_count = int(offdiag[max_idx]) if offdiag.size else 0
        pair_mats[f"{left}-{right}"] = matrix
        pair_summaries[f"{left}-{right}"] = {
            "diagonal_mass": diag_mass,
            "most_common_offdiag_from": values[max_idx[0]] if max_count > 0 else "",
            "most_common_offdiag_to": values[max_idx[1]] if max_count > 0 else "",
            "most_common_offdiag_count": max_count,
            "total": total,
        }

    return {"values": values, "pair_mats": pair_mats, "pair_summaries": pair_summaries}


def _plot_enum_confusion_grid(enum_data: dict[str, Any], out_path: Path) -> None:
    sns.set_theme(style="white")
    fig, axes = plt.subplots(len(ENUM_FIELDS), 3, figsize=(24, 18), constrained_layout=True)

    for row_idx, field in enumerate(ENUM_FIELDS):
        values = enum_data[field]["values"]
        for col_idx, (left, right) in enumerate(PAIR_KEYS):
            key = f"{left}-{right}"
            matrix = enum_data[field]["pair_mats"][key]
            ax = axes[row_idx, col_idx]
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="YlGnBu",
                annot=True,
                fmt="d",
                xticklabels=values,
                yticklabels=values,
                cbar=False,
                linewidths=0.2,
                linecolor="white",
            )
            ax.set_title(f"{field}: {left} vs {right}")
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_xlabel(right)
            ax.set_ylabel(left)

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_dominant_confusion(dominant_data: dict[str, Any], out_path: Path) -> None:
    values = dominant_data["values"]
    pair_mats = dominant_data["pair_mats"]

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), constrained_layout=True)

    for ax, (left, right) in zip(axes, PAIR_KEYS, strict=True):
        key = f"{left}-{right}"
        matrix = pair_mats[key].astype(np.float64)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_norm = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=row_sums > 0,
        )
        sns.heatmap(
            matrix_norm,
            ax=ax,
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            xticklabels=values,
            yticklabels=values,
            cbar=True,
            cbar_kws={"label": "Row-normalized rate"},
            linewidths=0.05,
            linecolor="white",
        )
        diag_mass = (np.trace(matrix) / matrix.sum()) if matrix.sum() > 0 else 0.0
        ax.set_title(
            f"dominant_admission_reason: {left} vs {right} "
            f"(diag {diag_mass * 100:.1f}%)"
        )
        ax.set_xlabel(right)
        ax.set_ylabel(left)
        ax.tick_params(axis="x", labelrotation=90, labelsize=5)
        ax.tick_params(axis="y", labelsize=5)

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _llm_vote_for_target(features: dict[str, Any], target: TargetKey) -> Vote:
    if target.field == "admission_reason_tags":
        tags = set(features.get("admission_reason_tags", []))
        return Vote.POSITIVE if target.value in tags else Vote.ABSTAIN

    value = str(features.get(target.field))
    if target.value == "yes":
        if value == "yes":
            return Vote.POSITIVE
        if value == "no":
            return Vote.NEGATIVE
        return Vote.ABSTAIN

    if target.value == "no":
        if value == "no":
            return Vote.POSITIVE
        if value == "yes":
            return Vote.NEGATIVE
        return Vote.ABSTAIN

    return Vote.ABSTAIN


def _aggregate_non_llm_vote(outputs: list[Vote]) -> Vote:
    if any(vote == Vote.POSITIVE for vote in outputs):
        return Vote.POSITIVE
    if any(vote == Vote.NEGATIVE for vote in outputs):
        return Vote.NEGATIVE
    return Vote.ABSTAIN


def _plot_lf_llm_agreement(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(24, 14))

    for ax, group_name in zip(axes, ["icd_anchored", "regex_anchored"], strict=True):
        sub = frame[frame["group"] == group_name].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{group_name} (no data)")
            continue

        targets = sorted(sub["target"].unique())
        x_base = np.arange(len(targets))
        width = 0.23
        offsets = {"A": -width, "B": 0.0, "C": width}

        for variant, alpha in [("A", 1.0), ("B", 0.85), ("C", 0.7)]:
            var_sub = sub[sub["variant"] == variant]
            bottoms = np.zeros(len(targets), dtype=np.float64)
            for category in LF_AGREEMENT_CATEGORIES:
                vals = []
                for target in targets:
                    row = var_sub[var_sub["target"] == target]
                    if row.empty:
                        vals.append(0.0)
                    else:
                        vals.append(float(row.iloc[0][f"{category}_pct"]))
                vals_arr = np.asarray(vals, dtype=np.float64)
                ax.bar(
                    x_base + offsets[variant],
                    vals_arr,
                    width=width,
                    bottom=bottoms,
                    color=LF_AGREEMENT_COLORS[category],
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=0.3,
                )
                bottoms += vals_arr

        ax.set_xticks(x_base)
        ax.set_xticklabels(targets, rotation=75, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Share of notes")
        ax.set_title(
            "ICD-anchored targets"
            if group_name == "icd_anchored"
            else "Regex-anchored TriState targets"
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=LF_AGREEMENT_COLORS[c]) for c in LF_AGREEMENT_CATEGORIES
    ]
    labels = [
        "LLM POS + LF POS",
        "LLM POS + LF ABSTAIN",
        "LLM non-POS + LF POS",
        "Both ABSTAIN",
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "LF vs LLM agreement composition by target and variant",
        fontsize=16,
        y=0.995,
    )
    fig.subplots_adjust(top=0.92, bottom=0.12)

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_tag_distribution_histograms(
    by_variant: dict[str, dict[int, dict[str, Any]]],
    hadm_ids: list[int],
    out_path: Path,
) -> None:
    variants = ["A", "B", "C"]
    n_notes = len(hadm_ids)
    tags = list(ADMISSION_REASON_TAGS)
    tag_idx = {tag: i for i, tag in enumerate(tags)}

    multi_rates: dict[str, list[float]] = {variant: [0.0] * len(tags) for variant in variants}
    dominant_rates: dict[str, list[float]] = {variant: [0.0] * len(tags) for variant in variants}

    for variant in variants:
        for hadm_id in hadm_ids:
            features = by_variant[variant][hadm_id]
            note_tags = set(features.get("admission_reason_tags", []))
            for tag in note_tags:
                if tag in tag_idx:
                    multi_rates[variant][tag_idx[tag]] += 1.0
            dominant = str(features.get("dominant_admission_reason"))
            if dominant in tag_idx:
                dominant_rates[variant][tag_idx[dominant]] += 1.0

    if n_notes > 0:
        for variant in variants:
            multi_rates[variant] = [value / n_notes for value in multi_rates[variant]]
            dominant_rates[variant] = [value / n_notes for value in dominant_rates[variant]]

    order = sorted(
        tags,
        key=lambda tag: -sum(multi_rates[variant][tag_idx[tag]] for variant in variants),
    )
    order_idx = [tag_idx[tag] for tag in order]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(28, 16), constrained_layout=True)
    width = 0.26
    x = np.arange(len(order))
    offsets = {"A": -width, "B": 0.0, "C": width}
    colors = {"A": "#4e79a7", "B": "#f28e2b", "C": "#59a14f"}

    panel_specs = [
        (
            axes[0],
            multi_rates,
            "Admission reason tag prevalence by variant (multi-label membership)",
        ),
        (
            axes[1],
            dominant_rates,
            "Dominant admission reason prevalence by variant",
        ),
    ]

    for ax, source, title in panel_specs:
        for variant in variants:
            vals = np.array([source[variant][idx] for idx in order_idx], dtype=np.float64)
            ax.bar(
                x + offsets[variant],
                vals * 100.0,
                width=width,
                color=colors[variant],
                edgecolor="white",
                linewidth=0.3,
                label=variant,
            )
        ax.set_title(title)
        ax.set_ylabel("Percent of notes")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=75, ha="right", fontsize=7)
        ax.legend(loc="upper right", frameon=True)

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    config.load_env()

    run_ids = args.run_ids
    variant_map = {"A": run_ids[0], "B": run_ids[1], "C": run_ids[2]}

    parsed_by_variant: dict[str, dict[int, dict[str, Any]]] = {}
    full_records_by_variant: dict[str, dict[int, dict[str, Any]]] = {}
    for variant, run_id in variant_map.items():
        parsed, full = _load_variant_results(run_id)
        parsed_by_variant[variant] = parsed
        full_records_by_variant[variant] = full

    hadm_ids = _intersect_hadm_ids(parsed_by_variant)
    if not hadm_ids:
        raise RuntimeError("No shared hadm_id intersection across all three variants.")

    kappa_json = json.loads(Path(args.kappa_json).read_text(encoding="utf-8"))
    kappa_results = kappa_json.get("kappa_results", {})

    # -----------------------------
    # Task 1: TriState decomposition
    # -----------------------------
    tristate_fields = [
        field
        for field in TRISTATE_FIELDS
        if field in kappa_results and not _is_low_base_rate(kappa_results, field)
    ]

    tri_pair_rows: list[dict[str, Any]] = []
    overall_counts = {category: 0 for category in CATEGORY_ORDER}
    field_totals: dict[str, dict[str, int]] = {
        field: {category: 0 for category in CATEGORY_ORDER} for field in tristate_fields
    }

    for field in tristate_fields:
        for left, right in PAIR_KEYS:
            left_values = [
                str(parsed_by_variant[left][hadm_id].get(field))
                for hadm_id in hadm_ids
            ]
            right_values = [
                str(parsed_by_variant[right][hadm_id].get(field))
                for hadm_id in hadm_ids
            ]
            matrix = _tristate_matrix(left_values, right_values)

            counts = {category: 0 for category in CATEGORY_ORDER}
            for lv, rv in zip(left_values, right_values, strict=True):
                category = _disagreement_category(lv, rv)
                counts[category] += 1

            for category in CATEGORY_ORDER:
                overall_counts[category] += counts[category]
                field_totals[field][category] += counts[category]

            total = len(hadm_ids)
            tri_pair_rows.append(
                {
                    "field": field,
                    "pair": f"{left}-{right}",
                    "n": total,
                    "full_agreement": counts["full_agreement"],
                    "soft_no_not_documented": counts["soft_no_not_documented"],
                    "soft_yes_not_documented": counts["soft_yes_not_documented"],
                    "hard_yes_no": counts["hard_yes_no"],
                    "full_agreement_pct": counts["full_agreement"] / total * 100,
                    "soft_no_not_documented_pct": counts["soft_no_not_documented"] / total * 100,
                    "soft_yes_not_documented_pct": counts["soft_yes_not_documented"] / total * 100,
                    "hard_yes_no_pct": counts["hard_yes_no"] / total * 100,
                    "matrix": matrix.tolist(),
                }
            )

    total_comparisons = sum(overall_counts.values())
    overall_disagreements = total_comparisons - overall_counts["full_agreement"]
    soft_no_pct = (
        overall_counts["soft_no_not_documented"] / overall_disagreements * 100
        if overall_disagreements
        else 0.0
    )
    soft_yes_pct = (
        overall_counts["soft_yes_not_documented"] / overall_disagreements * 100
        if overall_disagreements
        else 0.0
    )
    hard_pct = (
        overall_counts["hard_yes_no"] / overall_disagreements * 100
        if overall_disagreements
        else 0.0
    )

    tristate_plot_path = Path("docs/figures/30_tristate_disagreement_decomposition.png")
    _plot_tristate_decomposition(field_totals, tristate_plot_path)

    tri_table_rows: list[dict[str, Any]] = []
    for row in tri_pair_rows:
        tri_table_rows.append(
            {
                "field": row["field"],
                "pair": row["pair"],
                "full_agreement": row["full_agreement"],
                "soft_no_not_documented": row["soft_no_not_documented"],
                "soft_yes_not_documented": row["soft_yes_not_documented"],
                "hard_yes_no": row["hard_yes_no"],
                "hard_yes_no_pct": f"{row['hard_yes_no_pct']:.2f}%",
            }
        )

    tri_summary_rows: list[dict[str, Any]] = [
        {"metric": "total_pairwise_comparisons", "value": total_comparisons},
        {"metric": "total_disagreements", "value": overall_disagreements},
        {
            "metric": "soft_no_not_documented_pct_of_disagreements",
            "value": f"{soft_no_pct:.2f}%",
        },
        {
            "metric": "soft_yes_not_documented_pct_of_disagreements",
            "value": f"{soft_yes_pct:.2f}%",
        },
        {"metric": "hard_yes_no_pct_of_disagreements", "value": f"{hard_pct:.2f}%"},
    ]

    tri_report = [
        "# TriState Soft vs Hard Disagreement Decomposition (5k-audit, n=500 subset)\n",
        f"Generated at UTC: {datetime.now(UTC).isoformat()}\n",
        f"Run IDs: {', '.join(run_ids)}\n",
        f"Intersection of parse_ok notes across A/B/C: {len(hadm_ids)}\n",
        "## Headline\n",
        (
            f"{soft_no_pct:.2f}% of all TriState disagreements are on the "
            "`no` vs `not_documented` axis; "
            f"{hard_pct:.2f}% are hard `yes` vs `no` flips."
        ),
        "\n## Overall summary\n",
        _markdown_table(tri_summary_rows, ["metric", "value"]),
        "\n## Per-field per-pair decomposition\n",
        _markdown_table(
            tri_table_rows,
            [
                "field",
                "pair",
                "full_agreement",
                "soft_no_not_documented",
                "soft_yes_not_documented",
                "hard_yes_no",
                "hard_yes_no_pct",
            ],
        ),
        "\n## Figure\n",
        f"- `{tristate_plot_path}`",
    ]
    tri_out = Path("codex_outputs/30_tristate_soft_vs_hard.md")
    _ensure_parent(tri_out)
    tri_out.write_text("\n".join(tri_report) + "\n", encoding="utf-8")

    # -------------------------------------
    # Task 2: Admission tag confusion matrix
    # -------------------------------------
    pair_tag_data, top_tags = _build_tag_confusions(parsed_by_variant, hadm_ids)
    tag_plot_path = Path("docs/figures/30_admission_tag_confusion_grid.png")
    _plot_admission_tag_confusion(pair_tag_data, top_tags, tag_plot_path)

    tag_rows: list[dict[str, Any]] = []
    tag_summary_rows: list[dict[str, Any]] = []
    for pair_key, payload in pair_tag_data.items():
        total_diag = int(payload["total_diag"])
        total_offdiag = int(payload["total_offdiag"])
        diag_mass = (
            total_diag / (total_diag + total_offdiag)
            if (total_diag + total_offdiag)
            else 0.0
        )
        tag_summary_rows.append(
            {
                "pair": pair_key,
                "diag_agreement_count": total_diag,
                "offdiag_confusion_count": total_offdiag,
                "diagonal_mass": f"{diag_mass * 100:.2f}%",
            }
        )

        for tag_x, tag_y, count, rate in payload["top_pairs"]:
            tag_rows.append(
                {
                    "pair": pair_key,
                    "tag_from": tag_x,
                    "tag_to": tag_y,
                    "count": count,
                    "row_rate": f"{rate * 100:.2f}%",
                    "interpretation": _semantic_confusion_label(tag_x, tag_y),
                }
            )

    tag_report = [
        "# Admission Tag Confusion Analysis (5k-audit, n=500 subset)\n",
        f"Generated at UTC: {datetime.now(UTC).isoformat()}\n",
        f"Top tags rendered in heatmap: {len(top_tags)} of 47 (long tail moved to table).\n",
        "## Pair-level diagonal vs off-diagonal mass\n",
        _markdown_table(
            tag_summary_rows,
            ["pair", "diag_agreement_count", "offdiag_confusion_count", "diagonal_mass"],
        ),
        "\n## Top 10 confused tag pairs per variant pair\n",
        _markdown_table(
            tag_rows,
            ["pair", "tag_from", "tag_to", "count", "row_rate", "interpretation"],
        ),
        "\n## Figure\n",
        f"- `{tag_plot_path}`",
    ]
    tag_out = Path("codex_outputs/30_admission_tag_confusion.md")
    tag_out.write_text("\n".join(tag_report) + "\n", encoding="utf-8")

    # -------------------------------------------
    # Task 3: Enum confusion (plus dominant field)
    # -------------------------------------------
    enum_data: dict[str, Any] = {}
    enum_summary_rows: list[dict[str, Any]] = []

    for field in [*ENUM_FIELDS, DOMINANT_FIELD]:
        enum_data[field] = _build_enum_confusion(parsed_by_variant, hadm_ids, field)

    for field in [*ENUM_FIELDS, DOMINANT_FIELD]:
        for pair_key, summary in enum_data[field]["pair_summaries"].items():
            enum_summary_rows.append(
                {
                    "field": field,
                    "pair": pair_key,
                    "diagonal_mass": f"{summary['diagonal_mass'] * 100:.2f}%",
                    "most_common_offdiag": (
                        f"{summary['most_common_offdiag_from']} -> "
                        f"{summary['most_common_offdiag_to']}"
                        if summary["most_common_offdiag_count"] > 0
                        else "none"
                    ),
                    "offdiag_count": summary["most_common_offdiag_count"],
                }
            )

    enum_plot_path = Path("docs/figures/30_enum_confusion_grid.png")
    dominant_plot_path = Path("docs/figures/30_dominant_admission_confusion.png")
    _plot_enum_confusion_grid(enum_data, enum_plot_path)
    _plot_dominant_confusion(enum_data[DOMINANT_FIELD], dominant_plot_path)

    enum_report = [
        "# Enum Confusion Analysis (5k-audit, n=500 subset)\n",
        f"Generated at UTC: {datetime.now(UTC).isoformat()}\n",
        "## Per-field per-pair diagonal mass and top off-diagonal confusion\n",
        _markdown_table(
            enum_summary_rows,
            ["field", "pair", "diagonal_mass", "most_common_offdiag", "offdiag_count"],
        ),
        "\n## Figures\n",
        f"- `{enum_plot_path}`",
        f"- `{dominant_plot_path}`",
    ]
    enum_out = Path("codex_outputs/30_enum_confusion.md")
    enum_out.write_text("\n".join(enum_report) + "\n", encoding="utf-8")

    # ---------------------------------------------
    # Task 4: LF vs LLM agreement per variant/target
    # ---------------------------------------------
    engine = get_engine()
    notes_by_hadm = fetch_notes_by_hadm_ids(engine, hadm_ids)
    icd_by_hadm = fetch_icd_codes_by_hadm_ids(engine, hadm_ids)
    primary_icd_by_hadm = fetch_primary_icd_by_hadm_ids(engine, hadm_ids)

    sections_by_hadm = {
        hadm_id: parse_sections(notes_by_hadm.get(hadm_id, ""))
        for hadm_id in hadm_ids
    }

    icd_lfs = build_all_icd_lfs()
    regex_lfs = build_all_regex_lfs(Path("src/labeling_functions/patterns"))
    non_llm_lfs = [*icd_lfs, *regex_lfs]

    target_to_lfs: dict[TargetKey, list[Any]] = defaultdict(list)
    for lf in non_llm_lfs:
        target_value = lf.target_value
        if target_value is None:
            continue
        target = TargetKey(field=str(lf.target_field), value=str(target_value))
        target_to_lfs[target].append(lf)

    lf_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for target, lfs in sorted(target_to_lfs.items(), key=lambda item: item[0].as_text):
        group = "icd_anchored" if target.field == "admission_reason_tags" else "regex_anchored"
        for variant in ["A", "B", "C"]:
            counts = {category: 0 for category in LF_AGREEMENT_CATEGORIES}

            for hadm_id in hadm_ids:
                features = parsed_by_variant[variant][hadm_id]
                llm_vote = _llm_vote_for_target(features, target)

                icd_codes = icd_by_hadm.get(hadm_id, [])
                primary = primary_icd_by_hadm.get(hadm_id)
                lf_input = LFInput(
                    hadm_id=hadm_id,
                    note_text=notes_by_hadm.get(hadm_id, ""),
                    icd_codes=icd_codes,
                    primary_icd_code=primary[0] if primary else None,
                    primary_icd_version=primary[1] if primary else None,
                    sections=sections_by_hadm.get(hadm_id),
                )
                lf_votes = [lf(lf_input).vote for lf in lfs]
                agg_lf_vote = _aggregate_non_llm_vote(lf_votes)

                if llm_vote == Vote.POSITIVE and agg_lf_vote == Vote.POSITIVE:
                    counts["llm_pos_lf_pos"] += 1
                elif llm_vote == Vote.POSITIVE and agg_lf_vote == Vote.ABSTAIN:
                    counts["llm_pos_lf_abstain"] += 1
                elif llm_vote != Vote.POSITIVE and agg_lf_vote == Vote.POSITIVE:
                    counts["llm_nonpos_lf_pos"] += 1
                else:
                    counts["both_abstain"] += 1

            total = len(hadm_ids)
            agreement_rate = (
                (counts["llm_pos_lf_pos"] + counts["both_abstain"]) / total
                if total
                else 0.0
            )
            row = {
                "target": target.as_text,
                "field": target.field,
                "target_value": target.value,
                "group": group,
                "variant": variant,
                **counts,
                "total": total,
                **{
                    f"{key}_pct": counts[key] / total if total else 0.0
                    for key in LF_AGREEMENT_CATEGORIES
                },
                "agreement_rate": agreement_rate,
                "n_non_llm_lfs": len(lfs),
            }
            lf_rows.append(row)

    lf_frame = pd.DataFrame(lf_rows)
    for group in sorted(lf_frame["group"].unique()):
        for variant in ["A", "B", "C"]:
            sub = lf_frame[(lf_frame["group"] == group) & (lf_frame["variant"] == variant)]
            if sub.empty:
                continue
            weight = sub["total"].sum()
            weighted = (
                (sub["agreement_rate"] * sub["total"]).sum() / weight if weight else 0.0
            )
            aggregate_rows.append(
                {
                    "group": group,
                    "variant": variant,
                    "weighted_agreement_rate": f"{weighted * 100:.2f}%",
                    "n_targets": int(sub.shape[0]),
                }
            )

    lf_plot_path = Path("docs/figures/30_lf_vs_llm_agreement.png")
    _plot_lf_llm_agreement(lf_rows, lf_plot_path)
    tag_hist_path = Path("docs/figures/30_tag_distribution_histograms.png")
    _plot_tag_distribution_histograms(parsed_by_variant, hadm_ids, tag_hist_path)

    lf_report = [
        "# LF vs LLM Agreement Matrix (5k-audit, n=500 subset)\n",
        f"Generated at UTC: {datetime.now(UTC).isoformat()}\n",
        "## Aggregate agreement rate by LF-anchor group and variant\n",
        _markdown_table(
            aggregate_rows,
            ["group", "variant", "weighted_agreement_rate", "n_targets"],
        ),
        "\n## Per-target breakdown\n",
        _markdown_table(
            [
                {
                    "target": row["target"],
                    "group": row["group"],
                    "variant": row["variant"],
                    "llm_pos_lf_pos": row["llm_pos_lf_pos"],
                    "llm_pos_lf_abstain": row["llm_pos_lf_abstain"],
                    "llm_nonpos_lf_pos": row["llm_nonpos_lf_pos"],
                    "both_abstain": row["both_abstain"],
                    "agreement_rate": f"{row['agreement_rate'] * 100:.2f}%",
                }
                for row in lf_rows
            ],
            [
                "target",
                "group",
                "variant",
                "llm_pos_lf_pos",
                "llm_pos_lf_abstain",
                "llm_nonpos_lf_pos",
                "both_abstain",
                "agreement_rate",
            ],
        ),
        "\n## Figure\n",
        f"- `{lf_plot_path}`",
        f"- `{tag_hist_path}`",
    ]
    lf_out = Path("codex_outputs/30_lf_vs_llm_agreement.md")
    lf_out.write_text("\n".join(lf_report) + "\n", encoding="utf-8")

    # ---------------------------
    # Task 5: Combined synthesis
    # ---------------------------
    tag_diag_mass = []
    for pair_key, payload in pair_tag_data.items():
        denom = payload["total_diag"] + payload["total_offdiag"]
        mass = payload["total_diag"] / denom if denom else 0.0
        tag_diag_mass.append((pair_key, mass))

    enum_diag_rows = [
        row for row in enum_summary_rows if row["field"] in ENUM_FIELDS
    ]

    agreement_variant_summary: dict[str, float] = {}
    for variant in ["A", "B", "C"]:
        sub = [r for r in lf_rows if r["variant"] == variant]
        total_weight = sum(r["total"] for r in sub)
        weighted = (
            sum(r["agreement_rate"] * r["total"] for r in sub) / total_weight
            if total_weight
            else 0.0
        )
        agreement_variant_summary[variant] = weighted

    go_statement = "GO"
    rationale = []
    if hard_pct > 35.0:
        go_statement = "NO-GO"
        rationale.append("Hard yes/no disagreement proportion is high.")
    if not tag_diag_mass or min(mass for _, mass in tag_diag_mass) < 0.40:
        rationale.append("At least one variant-pair tag confusion matrix has low diagonal mass.")
    if not rationale:
        rationale.append("No new disagreement structure appears severe enough to block launch.")

    summary_lines = [
        "# Pre-Launch Confusion Analysis Summary\n",
        f"Generated at UTC: {datetime.now(UTC).isoformat()}\n",
        "## Soft vs hard TriState disagreement\n",
        (
            f"{soft_no_pct:.2f}% of TriState disagreements are `no` vs "
            f"`not_documented` (clinically soft), {soft_yes_pct:.2f}% are `yes` vs "
            f"`not_documented`, and {hard_pct:.2f}% are hard `yes` vs `no` flips."
        ),
        "\n## Admission tag confusion\n",
        "Diagonal mass by pair:",
    ]
    for pair_key, mass in tag_diag_mass:
        summary_lines.append(f"- {pair_key}: {mass * 100:.2f}%")

    summary_lines.extend(
        [
            "\n## Enum confusion\n",
            "Diagonal mass snapshots (A-B / A-C / B-C):",
        ]
    )
    for field in ENUM_FIELDS:
        parts = []
        for pair_key in ["A-B", "A-C", "B-C"]:
            item = next(
                row for row in enum_diag_rows if row["field"] == field and row["pair"] == pair_key
            )
            parts.append(f"{pair_key}: {item['diagonal_mass']}")
        summary_lines.append(f"- {field}: " + ", ".join(parts))

    summary_lines.extend(
        [
            "\n## LF vs LLM agreement\n",
            *[
                f"- Variant {variant}: "
                f"{agreement_variant_summary[variant] * 100:.2f}% weighted agreement"
                for variant in ["A", "B", "C"]
            ],
            "\n## Final readiness statement\n",
            f"**{go_statement}** for Phase 7 launch based on this pre-launch structural analysis.",
            "Rationale:",
            *[f"- {line}" for line in rationale],
        ]
    )

    summary_out = Path("codex_outputs/30_prelaunch_analysis_summary.md")
    summary_out.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # ---------------------------------
    # Task 6: Verification report output
    # ---------------------------------
    verification_out = Path("codex_outputs/30_prelaunch_analysis_verification.md")
    generated_paths = [
        Path("codex_outputs/30_tristate_soft_vs_hard.md"),
        Path("codex_outputs/30_admission_tag_confusion.md"),
        Path("codex_outputs/30_enum_confusion.md"),
        Path("codex_outputs/30_lf_vs_llm_agreement.md"),
        Path("codex_outputs/30_prelaunch_analysis_summary.md"),
        Path("docs/figures/30_tristate_disagreement_decomposition.png"),
        Path("docs/figures/30_admission_tag_confusion_grid.png"),
        Path("docs/figures/30_enum_confusion_grid.png"),
        Path("docs/figures/30_dominant_admission_confusion.png"),
        Path("docs/figures/30_lf_vs_llm_agreement.png"),
        Path("docs/figures/30_tag_distribution_histograms.png"),
    ]
    missing = [str(path) for path in generated_paths if not path.exists()]

    verification_lines = [
        "# Prelaunch Analysis Verification\n",
        f"Generated at UTC: {datetime.now(UTC).isoformat()}\n",
        f"Shared parse_ok intersection count (A/B/C): {len(hadm_ids)}",
        f"TriState fields analyzed (n_positive_total >= 10): {len(tristate_fields)}",
        f"Overall TriState disagreements: {overall_disagreements}",
        f"Soft no/not_documented proportion: {soft_no_pct:.2f}%",
        f"Hard yes/no proportion: {hard_pct:.2f}%",
        "\n## Artifact existence check",
    ]
    for path in generated_paths:
        status = "PASS" if path.exists() else "MISSING"
        verification_lines.append(f"- [{status}] {path}")

    if missing:
        verification_lines.append("\n## Result")
        verification_lines.append("- FAIL: Missing generated artifacts.")
    else:
        verification_lines.append("\n## Result")
        verification_lines.append("- PASS: All Prompt 25 analysis artifacts generated.")
        verification_lines.append("- OpenAI API cost: $0 (no extraction calls).")

    verification_out.write_text("\n".join(verification_lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
