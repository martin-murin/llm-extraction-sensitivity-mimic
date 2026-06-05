from __future__ import annotations

# Release documentation:
# Builds publication figure module `fig02_cross_prompt_grouped_bars`.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs Figure 2.
# Usage: `python -m src.paper_figures.fig02_cross_prompt_grouped_bars` or `python scripts/build_paper_figures.py`.

# ruff: noqa: E501

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from paper.claims.scripts._model_size_primary import (
    load_paired_model_size_data,
    mean_pairwise_agreements,
    pooled_kappa_levels,
)
from src.paper_figures.config import (
    COLOR_MODEL_FULL,
    COLOR_MODEL_SMALL,
    DOUBLE_COLUMN_WIDTH,
)
from src.paper_figures.plot_utils import apply_paper_style, save_figure


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_02_cross_prompt_agreement_modelsize_reasoning"
    width_inches: float = DOUBLE_COLUMN_WIDTH
    height_inches: float = 4.2
    bar_width: float = 0.22
    condition_order: tuple[str, ...] = ("nano_off", "full_off")
    condition_labels: dict[str, str] = None  # type: ignore[assignment]
    condition_colors: dict[str, str] = None  # type: ignore[assignment]


CFG = FigConfig(
    condition_labels={"nano_off": "gpt-5.4-nano", "full_off": "gpt-5.4"},
    condition_colors={"nano_off": COLOR_MODEL_SMALL, "full_off": COLOR_MODEL_FULL},
)


def prepare_data() -> pd.DataFrame:
    data = load_paired_model_size_data(base_rate_threshold=10)
    k_small_tri, k_full_tri, k_small_col, k_full_col = pooled_kappa_levels(data)
    tag_small, tag_full, primary_small, primary_full = mean_pairwise_agreements(data)

    rows = [
        {
            "context": "TriState filtered median kappa",
            "condition": "nano_off",
            "value": k_small_tri,
            "n": len(data.included_fields),
        },
        {
            "context": "TriState filtered median kappa",
            "condition": "full_off",
            "value": k_full_tri,
            "n": len(data.included_fields),
        },
        {
            "context": "TriState collapsed median kappa",
            "condition": "nano_off",
            "value": k_small_col,
            "n": len(data.included_fields),
        },
        {
            "context": "TriState collapsed median kappa",
            "condition": "full_off",
            "value": k_full_col,
            "n": len(data.included_fields),
        },
        {
            "context": "Admission-tag set agreement (Jaccard)",
            "condition": "nano_off",
            "value": tag_small,
            "n": len(data.common_hadm_ids),
        },
        {
            "context": "Admission-tag set agreement (Jaccard)",
            "condition": "full_off",
            "value": tag_full,
            "n": len(data.common_hadm_ids),
        },
        {
            "context": "Primary-admission diagonal mass",
            "condition": "nano_off",
            "value": primary_small,
            "n": len(data.common_hadm_ids),
        },
        {
            "context": "Primary-admission diagonal mass",
            "condition": "full_off",
            "value": primary_full,
            "n": len(data.common_hadm_ids),
        },
    ]
    return pd.DataFrame(rows)


def render(df: pd.DataFrame, cfg: FigConfig = CFG) -> plt.Figure:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(cfg.width_inches, cfg.height_inches))

    contexts = [
        "TriState filtered median kappa",
        "TriState collapsed median kappa",
        "Admission-tag set agreement (Jaccard)",
        "Primary-admission diagonal mass",
    ]
    x = np.arange(len(contexts), dtype=float)
    offsets = {"nano_off": -cfg.bar_width / 2, "full_off": cfg.bar_width / 2}

    for cond in cfg.condition_order:
        sub = df[df["condition"] == cond].set_index("context")
        vals = np.array([float(sub.at[c, "value"]) for c in contexts], dtype=float)
        ax.bar(
            x + offsets[cond],
            vals,
            width=cfg.bar_width,
            color=cfg.condition_colors[cond],
            edgecolor="#333333",
            linewidth=0.5,
            alpha=0.9,
            label=cfg.condition_labels[cond],
            zorder=3,
        )
        for i, v in enumerate(vals):
            ax.text(x[i] + offsets[cond], v / 2, f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "TriState\nmedian kappa",
            "Collapsed\nmedian kappa",
            "Admission-tag set\nagreement (Jaccard)",
            "Primary admission\nreason diagonal mass",
        ]
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Agreement metric")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def build() -> plt.Figure:
    df = prepare_data()
    fig = render(df)
    save_figure(fig, CFG.figure_name)
    return fig


if __name__ == "__main__":
    build()
