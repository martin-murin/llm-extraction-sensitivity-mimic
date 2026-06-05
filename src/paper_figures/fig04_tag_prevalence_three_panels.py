from __future__ import annotations

# Release documentation:
# Builds publication figure module `fig04_tag_prevalence_three_panels`.
#
# Reads: data/raw_responses/production_v1/results.jsonl, data/raw_responses/extended_5k_b/results.jsonl, data/raw_responses/extended_5k_c/results.jsonl, data/splits/extended_5k.csv.
# Writes: data/raw_responses/production_v1/results.jsonl, data/raw_responses/extended_5k_b/results.jsonl, data/raw_responses/extended_5k_c/results.jsonl, data/splits/extended_5k.csv.
# Backs Figure 4.
# Usage: `python -m src.paper_figures.fig04_tag_prevalence_three_panels` or `python scripts/build_paper_figures.py`.

# ruff: noqa: E501

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.paper_figures import config
from src.paper_figures.data_loaders import get_admission_tag_vocabulary, load_extended_5k_extractions
from src.paper_figures.plot_utils import apply_paper_style, save_figure


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_04_tag_prevalence_three_panels"
    width_inches: float = config.FULL_PAGE_WIDTH
    height_inches: float = 7.8

    variant_order: tuple[str, ...] = ("A", "B", "C")
    top_n_tags: int = 20
    panel_width_ratios: tuple[float, float] = (1.0, 1.0)

    variant_colors: dict[str, str] = field(
        default_factory=lambda: {
            "A": config.COLOR_VARIANT_A,
            "B": config.COLOR_VARIANT_B,
            "C": config.COLOR_VARIANT_C,
        }
    )
    bar_height: float = 0.30
    row_spacing: float = 1.35
    edge_padding_fraction: float = 0.02


CFG = FigConfig()


def _normalize_tag_list(raw: Any, vocab_set: set[str]) -> list[str]:
    if not isinstance(raw, list):
        return []
    tags = [str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag not in vocab_set or tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def prepare_data(cfg: FigConfig = CFG) -> dict[str, Any]:
    """Load and aggregate variant-wise admission-tag prevalence metrics."""
    frame = load_extended_5k_extractions().copy()
    frame = frame[frame["variant"].isin(cfg.variant_order)].copy()

    by_variant = {
        variant: frame[frame["variant"] == variant].drop_duplicates(subset=["hadm_id"], keep="last")
        for variant in cfg.variant_order
    }

    common_ids = set(by_variant[cfg.variant_order[0]]["hadm_id"])
    for variant in cfg.variant_order[1:]:
        common_ids &= set(by_variant[variant]["hadm_id"])
    common = sorted(common_ids)
    if not common:
        raise RuntimeError("No shared hadm_id intersection across extended_5k A/B/C for fig05.")

    vocab = get_admission_tag_vocabulary()
    vocab_set = set(vocab)

    prevalence_rows: list[dict[str, Any]] = []
    overall_note_hits: Counter[str] = Counter()

    for variant in cfg.variant_order:
        sub = by_variant[variant][by_variant[variant]["hadm_id"].isin(common)].copy()
        sub["_tags"] = sub["admission_reason_tags"].apply(lambda x: _normalize_tag_list(x, vocab_set))

        n_notes = len(sub)
        total_firings = 0
        tag_note_counts: Counter[str] = Counter()

        for tags in sub["_tags"].tolist():
            total_firings += len(tags)
            for tag in tags:
                tag_note_counts[tag] += 1
                overall_note_hits[tag] += 1

        for tag in vocab:
            count_notes = int(tag_note_counts.get(tag, 0))
            prevalence_pct = (100.0 * count_notes / n_notes) if n_notes else 0.0
            share_pct = (100.0 * count_notes / total_firings) if total_firings else 0.0
            prevalence_rows.append(
                {
                    "variant": variant,
                    "tag": tag,
                    "count_notes": count_notes,
                    "n_notes": n_notes,
                    "total_firings": total_firings,
                    "prevalence_pct": prevalence_pct,
                    "share_pct": share_pct,
                }
            )

    # Order all tags by absolute prevalence (note-level hits) descending.
    ordered_tags = sorted(vocab, key=lambda t: (-overall_note_hits.get(t, 0), vocab.index(t)))
    panel_bc_df = pd.DataFrame(prevalence_rows)
    ordered_tags = ordered_tags[: cfg.top_n_tags]
    panel_bc_df = panel_bc_df[panel_bc_df["tag"].isin(ordered_tags)].copy()

    return {
        "panel_bc": panel_bc_df,
        "ordered_tags": ordered_tags,
        "n_common": len(common),
    }


def _render_grouped_tag_bars_horizontal(
    ax: plt.Axes,
    panel_bc: pd.DataFrame,
    ordered_tags: list[str],
    cfg: FigConfig,
    value_col: str,
    title: str,
    xlabel: str,
    *,
    show_y_labels: bool,
) -> None:
    pivot = panel_bc.pivot(index="tag", columns="variant", values=value_col)
    pivot = pivot.reindex(index=ordered_tags, columns=list(cfg.variant_order)).fillna(0.0)

    y = np.arange(len(ordered_tags), dtype=np.float64) * cfg.row_spacing
    offsets = np.array([-cfg.bar_height, 0.0, cfg.bar_height], dtype=np.float64)

    for idx, variant in enumerate(cfg.variant_order):
        vals = pivot[variant].to_numpy(dtype=np.float64)
        ax.barh(
            y + offsets[idx],
            vals,
            height=cfg.bar_height,
            color=cfg.variant_colors[variant],
            label=variant,
            edgecolor="white",
            linewidth=0.3,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks(y)
    if show_y_labels:
        ax.set_yticklabels(ordered_tags, fontsize=7, fontfamily="monospace")
    else:
        # With shared y-axis, blanking ticklabels directly can clear labels on both panels.
        # Hide labels only on this axis via tick params.
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(axis="x", alpha=config.GRID_ALPHA)
    # Tighten edge margins while keeping larger inter-row spacing.
    y_min = float(np.min(y + offsets.min()) - cfg.bar_height / 2.0)
    y_max = float(np.max(y + offsets.max()) + cfg.bar_height / 2.0)
    y_span = max(1e-9, y_max - y_min)
    edge_pad = cfg.edge_padding_fraction * y_span
    ax.set_ylim(y_max + edge_pad, y_min - edge_pad)


def render(data: dict[str, Any], cfg: FigConfig = CFG) -> plt.Figure:
    """Render two-panel admission-tag prevalence figure."""
    apply_paper_style()

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(cfg.width_inches, cfg.height_inches),
        gridspec_kw={"width_ratios": cfg.panel_width_ratios},
        constrained_layout=False,
        sharey=True,
    )

    panel_bc = data["panel_bc"]
    ordered_tags = data["ordered_tags"]

    _render_grouped_tag_bars_horizontal(
        axes[0],
        panel_bc,
        ordered_tags,
        cfg,
        value_col="prevalence_pct",
        title="A) Absolute prevalence",
        xlabel="Notes with tag (%)",
        show_y_labels=True,
    )
    _render_grouped_tag_bars_horizontal(
        axes[1],
        panel_bc,
        ordered_tags,
        cfg,
        value_col="share_pct",
        title="B) Normalized share",
        xlabel="Share of tag firings (%)",
        show_y_labels=False,
    )

    axes[1].legend(title="Variant", loc="center right", frameon=True)
    fig.subplots_adjust(left=0.31, right=0.98, wspace=0.08, top=0.94, bottom=0.07)
    return fig


def build(cfg: FigConfig = CFG) -> tuple[plt.Figure, list[Any]]:
    """Build and save figure outputs."""
    data = prepare_data(cfg)
    fig = render(data, cfg)
    outputs = save_figure(fig, cfg.figure_name)
    plt.close(fig)
    return fig, outputs


if __name__ == "__main__":
    build()
