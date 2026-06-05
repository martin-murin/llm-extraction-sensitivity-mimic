from __future__ import annotations

# Release documentation:
# Builds publication figure module `figS04_disagreement_decomposition_dual`.
#
# Reads: data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl, codex_outputs/30_tristate_soft_vs_hard.md, codex_outputs/92_disagreement_collapse_decomposition.md.
# Writes: data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl, codex_outputs/30_tristate_soft_vs_hard.md, codex_outputs/92_disagreement_collapse_decomposition.md.
# Backs Supplement Figure S4.
# Usage: `python -m src.paper_figures.figS04_disagreement_decomposition_dual` or `python scripts/build_paper_figures.py`.

# ruff: noqa: E501

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.patches import Patch

from src.paper_figures import config
from src.paper_figures.data_loaders import (
    get_tristate_field_list,
)
from src.paper_figures.plot_utils import apply_paper_style, humanize_field_id, save_figure

TRISTATE_SET = {"yes", "no", "not_documented"}


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_S04_disagreement_decomposition_dual"
    width_inches: float = config.FULL_PAGE_WIDTH
    height_inches: float = 7.0

    pair_order: tuple[str, ...] = ("A-B", "A-C", "B-C")
    variant_order: tuple[str, ...] = ("A", "B", "C")

    original_categories: tuple[str, ...] = (
        "hard_yes_no",
        "soft_yes_not_documented",
        "soft_no_not_documented",
        "full_agreement",
    )
    collapsed_categories: tuple[str, ...] = (
        "null_disagreement",
        "residual_yes_vs_not_yes_disagreement",
        "full_agreement_collapsed",
    )

    category_colors: dict[str, str] = field(
        default_factory=lambda: {
            "full_agreement": config.COLOR_DISAGREE_AGREEMENT,
            "soft_no_not_documented": config.COLOR_DISAGREE_SOFT_NO_VS_NOT,
            "soft_yes_not_documented": config.COLOR_DISAGREE_SOFT_YES_VS_NOT,
            "hard_yes_no": config.COLOR_DISAGREE_HARD_YES_VS_NO,
            "full_agreement_collapsed": config.COLOR_DISAGREE_AGREEMENT,
            "residual_yes_vs_not_yes_disagreement": config.COLOR_DISAGREE_RESIDUAL_COLLAPSED,
            "null_disagreement": "#999999",
        }
    )
    pair_hatches: dict[str, str] = field(
        default_factory=lambda: {
            "A-B": "||",
            "A-C": "//",
            "B-C": "\\\\",
        }
    )

    bar_height: float = 0.22
    bar_offset_step: float = 0.25
    sort_field: str = "residual_yes_vs_not_yes_disagreement_share_mean"
    show_pair_legend: bool = True
    show_category_legend: bool = True
    x_axis_min: float = 0.0
    x_axis_max: float = 0.5
    top_headroom_fraction: float = 0.28
    bottom_padding_fraction: float = 0.03


CFG = FigConfig()
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_RESPONSES_DIR = REPO_ROOT / "data" / "raw_responses"
SPLITS_DIR = REPO_ROOT / "data" / "splits"


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


def _read_split_ids(path: Path) -> set[int]:
    frame = pd.read_csv(path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").dropna().astype("int64").tolist())


def _read_parse_ok_rows(run_id: str, allowed_hadm_ids: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = RAW_RESPONSES_DIR / run_id / "results.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not bool(payload.get("parse_ok", False)):
                continue
            hadm_id = int(payload["hadm_id"])
            if hadm_id not in allowed_hadm_ids:
                continue
            feats = payload.get("features_json")
            if not isinstance(feats, dict):
                continue
            rows.append({"hadm_id": hadm_id, **feats})
    return rows


def _load_combined_threeway_nano_off() -> pd.DataFrame:
    """Load A/B/C nano reasoning-OFF rows on the disjoint 1k + 500 + extended_5k pools.

    Combined pools:
    - methodology_1k (1000)
    - methodology_5k_audit_500 (500)
    - extended_5k (5000)
    """
    ids_1k = _read_split_ids(SPLITS_DIR / "methodology_1k.csv")
    ids_500 = _read_split_ids(SPLITS_DIR / "methodology_5k_audit_500.csv")
    ids_ext = _read_split_ids(SPLITS_DIR / "extended_5k.csv")

    run_map: dict[str, list[tuple[str, set[int]]]] = {
        "A": [
            ("methodology_1k_a", ids_1k),
            ("methodology_5k_a_subset500", ids_500),
            ("production_v1", ids_ext),
        ],
        "B": [
            ("methodology_1k_b", ids_1k),
            ("methodology_5k_audit_b", ids_500),
            ("extended_5k_b", ids_ext),
        ],
        "C": [
            ("methodology_1k_c", ids_1k),
            ("methodology_5k_audit_c", ids_500),
            ("extended_5k_c", ids_ext),
        ],
    }

    all_rows: list[dict[str, Any]] = []
    for variant, parts in run_map.items():
        rows: list[dict[str, Any]] = []
        for run_id, allowed_ids in parts:
            rows.extend(_read_parse_ok_rows(run_id, allowed_ids))
        variant_df = pd.DataFrame(rows)
        if variant_df.empty:
            continue
        variant_df = variant_df.drop_duplicates(subset=["hadm_id"], keep="last").copy()
        variant_df["variant"] = variant
        all_rows.extend(variant_df.to_dict(orient="records"))

    out = pd.DataFrame(all_rows)
    if out.empty:
        raise RuntimeError("Combined three-way nano/OFF frame is empty.")
    return out


def prepare_data(cfg: FigConfig = CFG) -> dict[str, Any]:
    """Load decomposition data by TriState field and prompt pair."""
    frame = _load_combined_threeway_nano_off()
    frame = frame[frame["variant"].isin(cfg.variant_order)].copy()
    if frame.empty:
        raise RuntimeError("No methodology 5k audit extraction rows found for variants A/B/C.")

    fields = [f for f in get_tristate_field_list("base") if f in frame.columns]
    if not fields:
        raise RuntimeError("No TriState fields available in methodology 5k audit frame.")

    by_variant = {
        variant: frame[frame["variant"] == variant].drop_duplicates(subset=["hadm_id"], keep="last")
        for variant in cfg.variant_order
    }

    common_ids = set(by_variant[cfg.variant_order[0]]["hadm_id"])
    for variant in cfg.variant_order[1:]:
        common_ids &= set(by_variant[variant]["hadm_id"])
    common = sorted(common_ids)
    if not common:
        raise RuntimeError("No shared hadm_id intersection across A/B/C for fig04.")

    for variant in cfg.variant_order:
        by_variant[variant] = by_variant[variant][by_variant[variant]["hadm_id"].isin(common)].copy()

    records: list[dict[str, Any]] = []
    pair_map = {
        "A-B": ("A", "B"),
        "A-C": ("A", "C"),
        "B-C": ("B", "C"),
    }

    for pair_key in cfg.pair_order:
        left_variant, right_variant = pair_map[pair_key]
        left = by_variant[left_variant].set_index("hadm_id")[fields]
        right = by_variant[right_variant].set_index("hadm_id")[fields]
        joined = left.join(right, how="inner", lsuffix="_l", rsuffix="_r")

        for field_name in fields:
            original_counts = {c: 0 for c in cfg.original_categories}
            collapsed_counts = {c: 0 for c in cfg.collapsed_categories}

            left_values = joined[f"{field_name}_l"].to_numpy()
            right_values = joined[f"{field_name}_r"].to_numpy()
            total_pairs = len(left_values)

            for left_raw, right_raw in zip(left_values, right_values, strict=True):
                left_value = _normalize_tristate(left_raw)
                right_value = _normalize_tristate(right_raw)

                original = _original_category(left_value, right_value)
                collapsed = _collapsed_category(left_value, right_value)

                if original is not None:
                    original_counts[original] += 1
                collapsed_counts[collapsed] += 1

            row: dict[str, Any] = {
                "field": field_name,
                "pair": pair_key,
                "total_pairs": total_pairs,
            }
            for cat in cfg.original_categories:
                row[f"{cat}_share"] = (original_counts[cat] / total_pairs) if total_pairs else 0.0
            for cat in cfg.collapsed_categories:
                row[f"{cat}_share"] = (collapsed_counts[cat] / total_pairs) if total_pairs else 0.0
            records.append(row)

    plot_df = pd.DataFrame.from_records(records)
    sort_df = (
        plot_df.groupby("field", as_index=False)["residual_yes_vs_not_yes_disagreement_share"]
        .mean()
        .rename(
            columns={
                "residual_yes_vs_not_yes_disagreement_share": "residual_yes_vs_not_yes_disagreement_share_mean"
            }
        )
    )
    sort_df = sort_df.sort_values(by=cfg.sort_field, ascending=False, kind="mergesort")
    field_order = sort_df["field"].tolist()

    return {
        "plot_df": plot_df,
        "field_order": field_order,
        "n_common": len(common),
    }


def _draw_stacked_grouped_barh(
    ax: plt.Axes,
    *,
    panel_df: pd.DataFrame,
    field_order: list[str],
    pair_order: tuple[str, ...],
    categories: tuple[str, ...],
    cfg: FigConfig,
    title: str,
    show_y_labels: bool,
) -> None:
    y_base = np.arange(len(field_order), dtype=np.float64)
    offsets = np.array([-cfg.bar_offset_step, 0.0, cfg.bar_offset_step], dtype=np.float64)

    for pair_idx, pair_key in enumerate(pair_order):
        sub = panel_df[panel_df["pair"] == pair_key].set_index("field").loc[field_order]
        y = y_base + offsets[pair_idx]
        left = np.zeros(len(field_order), dtype=np.float64)

        for cat in categories:
            vals = sub[f"{cat}_share"].to_numpy(dtype=np.float64)
            ax.barh(
                y,
                vals,
                left=left,
                height=cfg.bar_height,
                color=cfg.category_colors[cat],
                edgecolor="white",
                linewidth=0.4,
                hatch=cfg.pair_hatches[pair_key],
            )
            left += vals

    ax.set_xlim(cfg.x_axis_min, cfg.x_axis_max)
    ax.set_title(title)
    ax.set_xlabel("Share of pairwise comparisons")
    ax.set_yticks(y_base)
    if show_y_labels:
        ax.set_yticklabels([humanize_field_id(x) for x in field_order], fontsize=7)
    else:
        ax.tick_params(axis="y", labelleft=False)
    # Apply asymmetric y-padding: extra space at top only for in-panel legend.
    y_min = float(np.min(y_base + offsets.min()) - cfg.bar_height / 2.0)
    y_max = float(np.max(y_base + offsets.max()) + cfg.bar_height / 2.0)
    y_span = max(1e-9, y_max - y_min)
    top_pad = cfg.top_headroom_fraction * y_span
    bottom_pad = cfg.bottom_padding_fraction * y_span
    ax.set_ylim(y_max + bottom_pad, y_min - top_pad)
    ax.grid(axis="x", alpha=config.GRID_ALPHA)
    ax.grid(axis="y", visible=False)


def render(data: dict[str, Any], cfg: FigConfig = CFG) -> plt.Figure:
    """Render dual-panel disagreement decomposition figure."""
    apply_paper_style()

    plot_df = data["plot_df"]
    field_order = data["field_order"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(cfg.width_inches, cfg.height_inches),
        sharex=False,
        sharey=True,
        constrained_layout=False,
    )

    _draw_stacked_grouped_barh(
        axes[0],
        panel_df=plot_df,
        field_order=field_order,
        pair_order=cfg.pair_order,
        categories=cfg.original_categories,
        cfg=cfg,
        title="TriState decomposition",
        show_y_labels=True,
    )
    _draw_stacked_grouped_barh(
        axes[1],
        panel_df=plot_df,
        field_order=field_order,
        pair_order=cfg.pair_order,
        categories=cfg.collapsed_categories,
        cfg=cfg,
        title="Collapsed decomposition",
        show_y_labels=False,
    )

    category_label_map = {
        "full_agreement": "Agreement",
        "soft_no_not_documented": "No vs Not documented",
        "soft_yes_not_documented": "Yes vs Not documented",
        "hard_yes_no": "Yes vs No",
    }
    category_handles = [
        Patch(facecolor=cfg.category_colors[cat], edgecolor="white", label=category_label_map[cat])
        for cat in cfg.original_categories
    ]
    pair_handles = [
        Patch(facecolor="white", edgecolor="#444444", hatch=cfg.pair_hatches[pair], label=pair)
        for pair in cfg.pair_order
    ]

    if cfg.show_category_legend:
        category_legend = axes[0].legend(
            handles=category_handles,
            title="Categories",
            loc="upper left",
            ncol=1,
            frameon=True,
            bbox_to_anchor=(0.01, 0.99),
        )
        category_legend._legend_box.align = "left"
    if cfg.show_pair_legend:
        pair_legend = axes[1].legend(
            handles=pair_handles,
            title="Variant pairs",
            loc="upper left",
            ncol=1,
            frameon=True,
            bbox_to_anchor=(0.01, 0.99),
        )
        pair_legend._legend_box.align = "left"
    fig.subplots_adjust(left=0.31, right=0.98, top=0.94, bottom=0.12, hspace=0.10, wspace=0.12)
    return fig


def build(cfg: FigConfig = CFG) -> tuple[plt.Figure, list[Any]]:
    """Build and save figure outputs."""
    data = prepare_data(cfg)
    fig = render(data, cfg)
    outputs = save_figure(fig, cfg.figure_name, supplement=True)
    plt.close(fig)
    return fig, outputs


if __name__ == "__main__":
    build()
