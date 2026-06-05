#!/usr/bin/env python
from __future__ import annotations

# Release documentation:
# Runs staged pipeline step `92_disagreement_collapse_analysis.py`.
#
# Reads: data/raw_responses, codex_outputs/92_disagreement_collapse_decomposition.md, codex_outputs/92_cross_variant_kappa_collapse.md.
# Writes: data/raw_responses, codex_outputs/92_disagreement_collapse_decomposition.md, codex_outputs/92_cross_variant_kappa_collapse.md, docs/figures/92_tristate_collapse_decomposition.png.
# Backs Figure 3 and disagreement-decomposition claims.
# Usage: `python scripts/92_disagreement_collapse_analysis.py` unless the script's argparse help says otherwise.

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema.fields import LLMNoteFeatures
from src.utils.threeway_kappa import cohen_kappa_safe

REPO = Path(__file__).resolve().parent.parent

PAIR_KEYS: list[tuple[str, str]] = [("A", "B"), ("A", "C"), ("B", "C")]
TRISTATE_SET = {"yes", "no", "not_documented"}

ORIGINAL_CATEGORIES = [
    "full_agreement",
    "soft_no_not_documented",
    "soft_yes_not_documented",
    "hard_yes_no",
]
COLLAPSED_CATEGORIES = [
    "full_agreement_collapsed",
    "residual_yes_vs_not_yes_disagreement",
    "null_disagreement",
]

ORIGINAL_COLORS = {
    "full_agreement": "#4e79a7",
    "soft_no_not_documented": "#f28e2b",
    "soft_yes_not_documented": "#edc948",
    "hard_yes_no": "#e15759",
}
COLLAPSED_COLORS = {
    "full_agreement_collapsed": "#4e79a7",
    "residual_yes_vs_not_yes_disagreement": "#e15759",
    "null_disagreement": "#9c755f",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute TriState disagreement decomposition before/after yes-vs-not_yes collapse."
        )
    )
    parser.add_argument(
        "--run-ids",
        nargs=3,
        default=[
            "methodology_5k_a_subset500",
            "methodology_5k_audit_b",
            "methodology_5k_audit_c",
        ],
        help="Run IDs in A, B, C order.",
    )
    parser.add_argument(
        "--raw-responses-dir",
        default="data/raw_responses",
    )
    parser.add_argument(
        "--decomposition-out",
        default="codex_outputs/92_disagreement_collapse_decomposition.md",
    )
    parser.add_argument(
        "--kappa-out",
        default="codex_outputs/92_cross_variant_kappa_collapse.md",
    )
    parser.add_argument(
        "--figure-out",
        default="docs/figures/92_tristate_collapse_decomposition.png",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_variant_results(
    run_id: str,
    raw_responses_dir: Path,
) -> dict[int, dict[str, Any]]:
    path = raw_responses_dir / run_id / "results.jsonl"
    records = _read_jsonl(path)
    parsed: dict[int, dict[str, Any]] = {}
    for record in records:
        hadm_id = int(record["hadm_id"])
        if bool(record.get("parse_ok")) and isinstance(record.get("features_json"), dict):
            parsed[hadm_id] = dict(record["features_json"])
    return parsed


def _intersect_hadm_ids(by_variant: dict[str, dict[int, dict[str, Any]]]) -> list[int]:
    key_sets = [set(values.keys()) for values in by_variant.values()]
    if not key_sets:
        return []
    return sorted(set.intersection(*key_sets))


def _is_tristate_literal(annotation: Any) -> bool:
    if get_origin(annotation) is Literal:
        args = set(get_args(annotation))
        return args == TRISTATE_SET
    return False


def _tristate_fields_from_schema() -> list[str]:
    out: list[str] = []
    for name, field_info in LLMNoteFeatures.model_fields.items():
        if _is_tristate_literal(field_info.annotation):
            out.append(name)
    return out


def _normalize_tristate(value: Any) -> str | None:
    if isinstance(value, str) and value in TRISTATE_SET:
        return value
    return None


def _original_category(left: str | None, right: str | None) -> str | None:
    if left is None or right is None:
        return None
    if left == right:
        return "full_agreement"
    pair = {left, right}
    if pair == {"no", "not_documented"}:
        return "soft_no_not_documented"
    if pair == {"yes", "not_documented"}:
        return "soft_yes_not_documented"
    if pair == {"yes", "no"}:
        return "hard_yes_no"
    return None


def _collapsed_category(left: str | None, right: str | None) -> str:
    if left is None or right is None:
        return "null_disagreement"

    left_collapsed = "yes" if left == "yes" else "not_yes"
    right_collapsed = "yes" if right == "yes" else "not_yes"

    if left_collapsed == right_collapsed:
        return "full_agreement_collapsed"
    return "residual_yes_vs_not_yes_disagreement"


def _safe_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator * 100.0 / denominator


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body: list[str] = []
    for row in rows:
        cells: list[str] = []
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


def _plot_dual_panel(
    sorted_fields: list[str],
    original_props: dict[str, dict[str, float]],
    collapsed_props: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    if not sorted_fields:
        return

    y = np.arange(len(sorted_fields))
    fig_height = max(7.0, 0.45 * len(sorted_fields))
    fig, (ax_left, ax_right) = plt.subplots(
        ncols=2,
        figsize=(16, fig_height),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.06},
    )

    left_offsets = np.zeros(len(sorted_fields), dtype=np.float64)
    for category in ORIGINAL_CATEGORIES:
        values = np.array(
            [original_props[field].get(category, 0.0) for field in sorted_fields],
            dtype=np.float64,
        )
        ax_left.barh(
            y,
            values,
            left=left_offsets,
            color=ORIGINAL_COLORS[category],
            edgecolor="white",
            linewidth=0.5,
            label=category,
        )
        left_offsets += values

    right_offsets = np.zeros(len(sorted_fields), dtype=np.float64)
    for category in COLLAPSED_CATEGORIES:
        values = np.array(
            [collapsed_props[field].get(category, 0.0) for field in sorted_fields],
            dtype=np.float64,
        )
        ax_right.barh(
            y,
            values,
            left=right_offsets,
            color=COLLAPSED_COLORS[category],
            edgecolor="white",
            linewidth=0.5,
            label=category,
        )
        right_offsets += values

    ax_left.set_title("Original TriState decomposition")
    ax_right.set_title("Collapsed decomposition (yes vs not_yes)")
    ax_left.set_xlim(0, 1)
    ax_right.set_xlim(0, 1)
    ax_left.set_xlabel("Share of pairwise comparisons")
    ax_right.set_xlabel("Share of pairwise comparisons")
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(sorted_fields)
    ax_right.tick_params(axis="y", left=False, labelleft=False)

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    fig.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
    )
    fig.suptitle(
        "TriState decomposition before/after collapse (sorted by collapsed disagreement)",
        y=0.995,
    )

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = _parse_args()

    raw_responses_dir = REPO / args.raw_responses_dir
    run_ids = args.run_ids
    variant_map = {"A": run_ids[0], "B": run_ids[1], "C": run_ids[2]}

    parsed_by_variant: dict[str, dict[int, dict[str, Any]]] = {}
    for variant, run_id in variant_map.items():
        parsed_by_variant[variant] = _load_variant_results(run_id, raw_responses_dir)

    hadm_ids = _intersect_hadm_ids(parsed_by_variant)
    if not hadm_ids:
        raise RuntimeError("No shared parse_ok hadm_id intersection across all three variants.")

    tristate_fields = _tristate_fields_from_schema()
    if not tristate_fields:
        raise RuntimeError("No TriState fields discovered from LLMNoteFeatures schema.")

    field_stats: dict[str, dict[str, Any]] = {}
    global_original = {category: 0 for category in ORIGINAL_CATEGORIES}
    global_collapsed = {category: 0 for category in COLLAPSED_CATEGORIES}
    global_crosstab = {
        category: {collapsed: 0 for collapsed in COLLAPSED_CATEGORIES}
        for category in ORIGINAL_CATEGORIES
    }

    for field in tristate_fields:
        original_counts = {category: 0 for category in ORIGINAL_CATEGORIES}
        collapsed_counts = {category: 0 for category in COLLAPSED_CATEGORIES}
        crosstab = {
            category: {collapsed: 0 for collapsed in COLLAPSED_CATEGORIES}
            for category in ORIGINAL_CATEGORIES
        }
        null_pairs_total = 0

        for left_variant, right_variant in PAIR_KEYS:
            for hadm_id in hadm_ids:
                left_raw = parsed_by_variant[left_variant][hadm_id].get(field)
                right_raw = parsed_by_variant[right_variant][hadm_id].get(field)
                left = _normalize_tristate(left_raw)
                right = _normalize_tristate(right_raw)

                original_category = _original_category(left, right)
                collapsed_category = _collapsed_category(left, right)

                collapsed_counts[collapsed_category] += 1
                if original_category is not None:
                    original_counts[original_category] += 1
                    crosstab[original_category][collapsed_category] += 1
                else:
                    null_pairs_total += 1

        total_pairs = len(hadm_ids) * len(PAIR_KEYS)
        original_disagreements = (
            original_counts["soft_no_not_documented"]
            + original_counts["soft_yes_not_documented"]
            + original_counts["hard_yes_no"]
        )
        collapsed_disagreements = (
            collapsed_counts["residual_yes_vs_not_yes_disagreement"]
            + collapsed_counts["null_disagreement"]
        )
        dissolved_disagreements = crosstab["soft_no_not_documented"][
            "full_agreement_collapsed"
        ]
        preserved_disagreements = max(0, original_disagreements - dissolved_disagreements)

        field_stats[field] = {
            "total_pairs": total_pairs,
            "original": original_counts,
            "collapsed": collapsed_counts,
            "crosstab": crosstab,
            "null_pairs_total": null_pairs_total,
            "original_disagreements": original_disagreements,
            "collapsed_disagreements": collapsed_disagreements,
            "dissolved_disagreements": dissolved_disagreements,
            "preserved_disagreements": preserved_disagreements,
            "dissolved_pct": _safe_pct(dissolved_disagreements, original_disagreements),
            "preserved_pct": _safe_pct(preserved_disagreements, original_disagreements),
        }

        for category in ORIGINAL_CATEGORIES:
            global_original[category] += original_counts[category]
            for collapsed in COLLAPSED_CATEGORIES:
                global_crosstab[category][collapsed] += crosstab[category][collapsed]

        for category in COLLAPSED_CATEGORIES:
            global_collapsed[category] += collapsed_counts[category]

    total_pairs_global = len(hadm_ids) * len(PAIR_KEYS) * len(tristate_fields)
    original_disagreements_global = (
        global_original["soft_no_not_documented"]
        + global_original["soft_yes_not_documented"]
        + global_original["hard_yes_no"]
    )
    collapsed_disagreements_global = (
        global_collapsed["residual_yes_vs_not_yes_disagreement"]
        + global_collapsed["null_disagreement"]
    )
    dissolved_global = global_crosstab["soft_no_not_documented"]["full_agreement_collapsed"]
    preserved_global = max(0, original_disagreements_global - dissolved_global)

    original_props: dict[str, dict[str, float]] = {}
    collapsed_props: dict[str, dict[str, float]] = {}
    for field in tristate_fields:
        total = max(field_stats[field]["total_pairs"], 1)
        original_props[field] = {
            category: field_stats[field]["original"][category] / total
            for category in ORIGINAL_CATEGORIES
        }
        collapsed_props[field] = {
            category: field_stats[field]["collapsed"][category] / total
            for category in COLLAPSED_CATEGORIES
        }

    sorted_fields = sorted(
        tristate_fields,
        key=lambda field: field_stats[field]["collapsed_disagreements"]
        / max(field_stats[field]["total_pairs"], 1),
        reverse=True,
    )

    # Pairwise kappa before/after collapse
    kappa_rows: list[dict[str, Any]] = []
    pair_to_full: dict[str, list[float]] = defaultdict(list)
    pair_to_collapsed: dict[str, list[float]] = defaultdict(list)

    for field in tristate_fields:
        for left_variant, right_variant in PAIR_KEYS:
            left_labels_full: list[int] = []
            right_labels_full: list[int] = []
            left_labels_collapsed: list[int] = []
            right_labels_collapsed: list[int] = []

            for hadm_id in hadm_ids:
                left = _normalize_tristate(parsed_by_variant[left_variant][hadm_id].get(field))
                right = _normalize_tristate(parsed_by_variant[right_variant][hadm_id].get(field))
                if left is None or right is None:
                    continue

                full_map = {"yes": 2, "no": 1, "not_documented": 0}
                collapsed_map = {"yes": 1, "no": 0, "not_documented": 0}

                left_labels_full.append(full_map[left])
                right_labels_full.append(full_map[right])
                left_labels_collapsed.append(collapsed_map[left])
                right_labels_collapsed.append(collapsed_map[right])

            kappa_full = cohen_kappa_safe(left_labels_full, right_labels_full)
            kappa_collapsed = cohen_kappa_safe(left_labels_collapsed, right_labels_collapsed)
            pair_key = f"{left_variant}-{right_variant}"
            pair_to_full[pair_key].append(kappa_full)
            pair_to_collapsed[pair_key].append(kappa_collapsed)

            kappa_rows.append(
                {
                    "field": field,
                    "pair": pair_key,
                    "kappa_full": kappa_full,
                    "kappa_collapsed": kappa_collapsed,
                    "delta_collapsed_minus_full": kappa_collapsed - kappa_full,
                    "n_valid": len(left_labels_full),
                }
            )

    kappa_summary_rows: list[dict[str, Any]] = []
    for left_variant, right_variant in PAIR_KEYS:
        pair_key = f"{left_variant}-{right_variant}"
        full_values = np.array(pair_to_full[pair_key], dtype=np.float64)
        collapsed_values = np.array(pair_to_collapsed[pair_key], dtype=np.float64)
        kappa_summary_rows.append(
            {
                "pair": pair_key,
                "median_kappa_full": float(np.median(full_values)) if full_values.size else 0.0,
                "median_kappa_collapsed": float(np.median(collapsed_values))
                if collapsed_values.size
                else 0.0,
                "median_delta": (
                    float(np.median(collapsed_values) - np.median(full_values))
                    if full_values.size and collapsed_values.size
                    else 0.0
                ),
            }
        )

    median_full_across_fields = float(np.median([row["kappa_full"] for row in kappa_rows]))
    median_collapsed_across_fields = float(
        np.median([row["kappa_collapsed"] for row in kappa_rows])
    )

    # Build decomposition report
    global_summary_rows: list[dict[str, Any]] = [
        {"metric": "tri_state_fields", "value": len(tristate_fields)},
        {"metric": "shared_parse_ok_hadm_ids", "value": len(hadm_ids)},
        {"metric": "total_pairwise_comparisons", "value": total_pairs_global},
        {"metric": "original_disagreements", "value": original_disagreements_global},
        {"metric": "collapsed_disagreements", "value": collapsed_disagreements_global},
        {
            "metric": "dissolved_disagreements",
            "value": dissolved_global,
        },
        {
            "metric": "pct_disagreements_dissolved",
            "value": f"{_safe_pct(dissolved_global, original_disagreements_global):.2f}%",
        },
        {
            "metric": "pct_disagreements_preserved",
            "value": f"{_safe_pct(preserved_global, original_disagreements_global):.2f}%",
        },
        {
            "metric": "original_null_pairs_not_classified",
            "value": total_pairs_global - sum(global_original.values()),
        },
        {
            "metric": "collapsed_null_disagreement",
            "value": global_collapsed["null_disagreement"],
        },
    ]

    field_summary_rows: list[dict[str, Any]] = []
    for field in sorted_fields:
        stats = field_stats[field]
        dissolved_pct = stats["dissolved_pct"]
        preserved_pct = stats["preserved_pct"]
        field_summary_rows.append(
            {
                "field": field,
                "total_pairs": stats["total_pairs"],
                "original_disagreements": stats["original_disagreements"],
                "collapsed_disagreements": stats["collapsed_disagreements"],
                "dissolved_disagreements": stats["dissolved_disagreements"],
                "dissolved_pct": f"{dissolved_pct:.2f}%",
                "preserved_pct": f"{preserved_pct:.2f}%",
                "flag_dissolved_gt_90pct": "yes" if dissolved_pct > 90.0 else "no",
                "flag_preserved_gt_50pct": "yes" if preserved_pct > 50.0 else "no",
                "orig_full": stats["original"]["full_agreement"],
                "orig_soft_no_nd": stats["original"]["soft_no_not_documented"],
                "orig_soft_yes_nd": stats["original"]["soft_yes_not_documented"],
                "orig_hard_yes_no": stats["original"]["hard_yes_no"],
                "coll_full": stats["collapsed"]["full_agreement_collapsed"],
                "coll_residual": stats["collapsed"]["residual_yes_vs_not_yes_disagreement"],
                "coll_null": stats["collapsed"]["null_disagreement"],
            }
        )

    cross_tab_rows: list[dict[str, Any]] = []
    for field in sorted_fields:
        stats = field_stats[field]
        for original in ORIGINAL_CATEGORIES:
            row = {
                "field": field,
                "original_category": original,
            }
            for collapsed in COLLAPSED_CATEGORIES:
                row[collapsed] = stats["crosstab"][original][collapsed]
            cross_tab_rows.append(row)

    out_decomposition = REPO / args.decomposition_out
    out_figure = REPO / args.figure_out

    _plot_dual_panel(sorted_fields, original_props, collapsed_props, out_figure)

    original_totals_rows: list[dict[str, Any]] = [
        {"category": key, "count": value} for key, value in global_original.items()
    ]
    collapsed_totals_rows: list[dict[str, Any]] = [
        {"category": key, "count": value} for key, value in global_collapsed.items()
    ]

    decomposition_lines = [
        "# TriState Disagreement Collapse Decomposition (Task 2)",
        "",
        f"Generated at UTC: {datetime.now(tz=UTC).isoformat()}",
        "",
        f"Run IDs: {', '.join(run_ids)}",
        f"TriState universe (from `LLMNoteFeatures`): {len(tristate_fields)} fields",
        f"Shared parse_ok intersection across A/B/C: {len(hadm_ids)}",
        "",
        "## Global summary",
        "",
        _markdown_table(global_summary_rows, ["metric", "value"]),
        "",
        "## Aggregate category totals",
        "",
        _markdown_table(original_totals_rows, ["category", "count"]),
        "",
        _markdown_table(collapsed_totals_rows, ["category", "count"]),
        "",
        "## Per-field summary and flags",
        "",
        _markdown_table(
            field_summary_rows,
            [
                "field",
                "total_pairs",
                "original_disagreements",
                "collapsed_disagreements",
                "dissolved_disagreements",
                "dissolved_pct",
                "preserved_pct",
                "flag_dissolved_gt_90pct",
                "flag_preserved_gt_50pct",
                "orig_full",
                "orig_soft_no_nd",
                "orig_soft_yes_nd",
                "orig_hard_yes_no",
                "coll_full",
                "coll_residual",
                "coll_null",
            ],
        ),
        "",
        "## Per-field cross-tab (original category x collapsed category)",
        "",
        _markdown_table(
            cross_tab_rows,
            [
                "field",
                "original_category",
                "full_agreement_collapsed",
                "residual_yes_vs_not_yes_disagreement",
                "null_disagreement",
            ],
        ),
        "",
        "## Figure",
        "",
        f"- `{args.figure_out}`",
        "",
    ]
    _ensure_parent(out_decomposition)
    out_decomposition.write_text("\n".join(decomposition_lines), encoding="utf-8")

    # Build kappa report
    out_kappa = REPO / args.kappa_out

    kappa_lines = [
        "# Cross-Variant Kappa Before vs After Collapse (Task 2)",
        "",
        f"Generated at UTC: {datetime.now(tz=UTC).isoformat()}",
        "",
        f"Run IDs: {', '.join(run_ids)}",
        f"TriState fields evaluated: {len(tristate_fields)}",
        f"Shared parse_ok intersection across A/B/C: {len(hadm_ids)}",
        "",
        "## Pairwise median kappa across fields",
        "",
        _markdown_table(
            kappa_summary_rows,
            ["pair", "median_kappa_full", "median_kappa_collapsed", "median_delta"],
        ),
        "",
        "## Global median across all field-pair kappas",
        "",
        _markdown_table(
            [
                {
                    "metric": "median_kappa_full_all_field_pairs",
                    "value": median_full_across_fields,
                },
                {
                    "metric": "median_kappa_collapsed_all_field_pairs",
                    "value": median_collapsed_across_fields,
                },
                {
                    "metric": "median_delta_collapsed_minus_full_all_field_pairs",
                    "value": median_collapsed_across_fields - median_full_across_fields,
                },
            ],
            ["metric", "value"],
        ),
        "",
        "## Field-level pairwise kappa detail",
        "",
        _markdown_table(
            sorted(
                kappa_rows,
                key=lambda row: (row["pair"], row["delta_collapsed_minus_full"]),
                reverse=False,
            ),
            [
                "field",
                "pair",
                "n_valid",
                "kappa_full",
                "kappa_collapsed",
                "delta_collapsed_minus_full",
            ],
        ),
        "",
    ]
    _ensure_parent(out_kappa)
    out_kappa.write_text("\n".join(kappa_lines), encoding="utf-8")

    print(f"Wrote {out_decomposition.relative_to(REPO)}")
    print(f"Wrote {out_kappa.relative_to(REPO)}")
    print(f"Wrote {out_figure.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
