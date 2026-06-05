from __future__ import annotations

# Release documentation:
# Builds publication figure module `figS05_refinement_holdout_generalization`.
#
# Reads: data/raw_responses/*/results.jsonl, data/splits/{refinement_150.
# Writes: data/raw_responses/*/results.jsonl, data/splits/{refinement_150.
# Backs Supplement Figure S5.
# Usage: `python -m src.paper_figures.figS05_refinement_holdout_generalization` or `python scripts/build_paper_figures.py`.

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from paper.claims.scripts._threeway_kappa_primary import (
    SAMPLE_SPECS,
    recompute_per_variant_medians,
)

from src.paper_figures.config import (
    COLOR_SAMPLE_EXTENDED,
    COLOR_SAMPLE_HOLDOUT,
    COLOR_SAMPLE_METH_1K,
    COLOR_SAMPLE_REFINEMENT,
    SINGLE_COLUMN_WIDTH,
)
from src.paper_figures.plot_utils import apply_paper_style, save_figure


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_S05_refinement_holdout_generalization"
    width_inches: float = SINGLE_COLUMN_WIDTH * 1.35
    height_inches: float = 3.6
    title: str = ""

    sample_order: tuple[str, ...] = (
        "refinement_150",
        "holdout_150",
        "methodology_1k",
        "extended_5k",
    )
    sample_labels: dict[str, str] = None  # type: ignore[assignment]
    sample_colors: dict[str, str] = None  # type: ignore[assignment]
    bar_width: float = 0.19
    y_min: float = 0.55
    y_max: float = 0.85
    annotate_bars: bool = True
    annotate_delta_per_variant: bool = False
    annotation_fontsize: int = 5
    legend_fontsize: int = 7
    legend_loc: str = "upper right"


CFG = FigConfig(
    sample_labels={
        "refinement_150": "Refinement (n=150)",
        "holdout_150": "Holdout (n=150)",
        "methodology_1k": "Methodology 1k",
        "extended_5k": "Extended 5k",
    },
    sample_colors={
        "refinement_150": COLOR_SAMPLE_REFINEMENT,
        "holdout_150": COLOR_SAMPLE_HOLDOUT,
        "methodology_1k": COLOR_SAMPLE_METH_1K,
        "extended_5k": COLOR_SAMPLE_EXTENDED,
    },
)

REPO = Path(__file__).resolve().parents[2]


def prepare_data(cfg: FigConfig = CFG) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for sample_key in cfg.sample_order:
        if sample_key not in SAMPLE_SPECS:
            raise ValueError(f"Unsupported sample key in S05 config: {sample_key}")
        run_ids, split_csv = SAMPLE_SPECS[sample_key]
        per_variant = recompute_per_variant_medians(sample_key, run_ids, split_csv)

        for variant in ("A", "B", "C"):
            rows.append(
                {
                    "sample_key": sample_key,
                    "variant": variant,
                    "kappa": float(per_variant[variant]),
                }
            )

    out = pd.DataFrame(rows)
    out["sample_key"] = pd.Categorical(
        out["sample_key"], categories=list(cfg.sample_order), ordered=True
    )
    return out.sort_values(["variant", "sample_key"]).reset_index(drop=True)


def render(
    per_variant: pd.DataFrame,
    cfg: FigConfig = CFG,
) -> plt.Figure:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(cfg.width_inches, cfg.height_inches))

    variants = ["A", "B", "C"]
    x = np.arange(len(variants), dtype=float)
    offsets = np.linspace(
        -1.5 * cfg.bar_width, 1.5 * cfg.bar_width, num=len(cfg.sample_order), dtype=float
    )

    for sample_idx, sample_key in enumerate(cfg.sample_order):
        vals = []
        for variant in variants:
            row = per_variant[
                (per_variant["variant"] == variant) & (per_variant["sample_key"] == sample_key)
            ]
            vals.append(float(row["kappa"].iloc[0]))
        ax.bar(
            x + offsets[sample_idx],
            vals,
            width=cfg.bar_width,
            color=cfg.sample_colors[sample_key],
            edgecolor="#2f2f2f",
            linewidth=0.6,
            label=cfg.sample_labels[sample_key],
            zorder=3,
        )

    if cfg.annotate_bars:
        for sample_idx, sample_key in enumerate(cfg.sample_order):
            for variant_idx, variant in enumerate(variants):
                row = per_variant[
                    (per_variant["variant"] == variant)
                    & (per_variant["sample_key"] == sample_key)
                ]
                val = float(row["kappa"].iloc[0])
                ax.text(
                    x[variant_idx] + offsets[sample_idx],
                    val + 0.006,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=cfg.annotation_fontsize,
                )

    if cfg.annotate_delta_per_variant:
        for i, row in per_variant.iterrows():
            delta_pp = float(row["delta_pp"])
            y = max(float(row["refinement_kappa"]), float(row["holdout_kappa"])) + 0.02
            ax.text(
                i,
                y,
                f"{delta_pp:+.2f}%",
                ha="center",
                va="bottom",
                fontsize=cfg.annotation_fontsize,
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_xlabel("Variant")
    ax.set_ylabel("Median κ")
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_title(cfg.title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.grid(axis="x", visible=False)
    ax.legend(loc=cfg.legend_loc, frameon=False, fontsize=cfg.legend_fontsize)

    fig.tight_layout()
    return fig


def build() -> plt.Figure:
    per_variant = prepare_data()
    fig = render(per_variant)
    save_figure(fig, CFG.figure_name, supplement=True)
    return fig


if __name__ == "__main__":
    build()
