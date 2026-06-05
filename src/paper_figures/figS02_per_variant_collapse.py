from __future__ import annotations

# Release documentation:
# Builds publication figure module `figS02_per_variant_collapse`.
#
# Reads: data/raw_responses/{methodology_1k_*.
# Writes: data/raw_responses/{methodology_1k_*.
# Backs Supplement Figure S2.
# Usage: `python -m src.paper_figures.figS02_per_variant_collapse` or `python scripts/build_paper_figures.py`.

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from paper.claims.scripts._model_size_primary import (
    load_paired_model_size_data,
    per_variant_cross_model_kappa,
)

from src.paper_figures.config import (
    COLOR_SEM_COLLAPSED,
    COLOR_SEM_TRISTATE,
    SINGLE_COLUMN_WIDTH,
)
from src.paper_figures.plot_utils import apply_paper_style, save_figure


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_S02_per_variant_collapse"
    width_inches: float = SINGLE_COLUMN_WIDTH
    height_inches: float = 3.2
    title: str = ""

    color_full: str = COLOR_SEM_TRISTATE
    color_collapsed: str = COLOR_SEM_COLLAPSED
    bar_width: float = 0.34
    annotate: bool = True
    annotation_fontsize: int = 6
    tick_fontsize: int = 7
    axis_label_fontsize: int = 8
    legend_fontsize: int = 7


CFG = FigConfig()


def prepare_data() -> pd.DataFrame:
    data = load_paired_model_size_data(base_rate_threshold=10)
    per_variant = per_variant_cross_model_kappa(data)
    rows = []
    for variant in ("A", "B", "C"):
        tri, collapsed = per_variant[variant]
        rows.append(
            {
                "variant": variant,
                "kappa_full": float(tri),
                "kappa_collapsed": float(collapsed),
            }
        )
    return pd.DataFrame(rows)


def render(df: pd.DataFrame, cfg: FigConfig = CFG) -> plt.Figure:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(cfg.width_inches, cfg.height_inches))

    x = np.arange(len(df))
    ax.bar(
        x - cfg.bar_width / 2,
        df["kappa_full"],
        width=cfg.bar_width,
        color=cfg.color_full,
        edgecolor="#000000",
        linewidth=0.6,
        label="TriState",
        zorder=3,
    )
    ax.bar(
        x + cfg.bar_width / 2,
        df["kappa_collapsed"],
        width=cfg.bar_width,
        color=cfg.color_collapsed,
        edgecolor="#000000",
        linewidth=0.6,
        label="Collapsed",
        zorder=3,
    )

    if cfg.annotate:
        for i, row in df.iterrows():
            ax.text(
                i - cfg.bar_width / 2,
                float(row["kappa_full"]) + 0.02,
                f"{row['kappa_full']:.2f}",
                ha="center",
                va="bottom",
                fontsize=cfg.annotation_fontsize,
            )
            ax.text(
                i + cfg.bar_width / 2,
                float(row["kappa_collapsed"]) + 0.02,
                f"{row['kappa_collapsed']:.2f}",
                ha="center",
                va="bottom",
                fontsize=cfg.annotation_fontsize,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"].tolist(), fontsize=cfg.tick_fontsize)
    ax.set_xlabel("Variant", fontsize=cfg.axis_label_fontsize)
    ax.set_ylabel(
        r"$\overline{\kappa}$ (small vs full model; median over fields)",
        fontsize=cfg.axis_label_fontsize,
    )
    ax.set_ylim(0, 1.0)
    ax.set_title(cfg.title)
    ax.tick_params(axis="y", labelsize=cfg.tick_fontsize)
    ax.legend(loc="upper left", frameon=False, fontsize=cfg.legend_fontsize)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    return fig


def build() -> plt.Figure:
    df = prepare_data()
    fig = render(df)
    save_figure(fig, CFG.figure_name, supplement=True)
    return fig


if __name__ == "__main__":
    build()
