from __future__ import annotations

# Release documentation:
# Builds publication figure module `figS09_methodology_pipeline_schematic`.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs supplement methodology schematic.
# Usage: `python -m src.paper_figures.figS09_methodology_pipeline_schematic` or `python scripts/build_paper_figures.py`.

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from src.paper_figures.config import (
    COLOR_SCHEMA_COLLAPSED,
    COLOR_SCHEMA_TRISTATE,
    DOUBLE_COLUMN_WIDTH,
)
from src.paper_figures.plot_utils import apply_paper_style, save_figure


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_S09_methodology_pipeline_schematic"
    width_inches: float = DOUBLE_COLUMN_WIDTH
    height_inches: float = 6.4
    title: str = ""
    box_face: str = COLOR_SCHEMA_TRISTATE
    emphasis_face: str = COLOR_SCHEMA_COLLAPSED
    edge: str = "#3f3f3f"
    text_size: int = 8


CFG = FigConfig()


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    cfg: FigConfig,
) -> None:
    rect = Rectangle((x, y), w, h, facecolor=face, edgecolor=cfg.edge, linewidth=0.9)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=cfg.text_size)


def _arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color="#4a4a4a",
        )
    )


def render(cfg: FigConfig = CFG) -> plt.Figure:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(cfg.width_inches, cfg.height_inches))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Row 1
    _box(ax, 0.34, 0.90, 0.32, 0.07, "MIMIC-IV discharge notes", cfg.box_face, cfg)

    # Row 2
    w, h = 0.17, 0.07
    _box(ax, 0.09, 0.76, w, h, "Variant A", cfg.box_face, cfg)
    _box(ax, 0.31, 0.76, w, h, "Variant B", cfg.box_face, cfg)
    _box(ax, 0.53, 0.76, w, h, "Variant C", cfg.box_face, cfg)
    _box(ax, 0.75, 0.76, w, h, "ICD + regex LFs", cfg.box_face, cfg)

    # Row 3
    _box(ax, 0.34, 0.61, 0.32, 0.07, "Snorkel ensemble", cfg.box_face, cfg)
    _box(ax, 0.34, 0.48, 0.32, 0.07, "Cross-prompt agreement metrics", cfg.box_face, cfg)

    # Row 4
    _box(ax, 0.10, 0.34, 0.23, 0.07, "Binary collapse", cfg.box_face, cfg)
    _box(ax, 0.39, 0.34, 0.23, 0.07, "Reasoning ON/OFF", cfg.box_face, cfg)
    _box(ax, 0.68, 0.34, 0.23, 0.07, "Same-note paired full-model", cfg.box_face, cfg)

    # Final emphasis
    _box(
        ax,
        0.28,
        0.16,
        0.44,
        0.09,
        "Three-component decomposition",
        cfg.emphasis_face,
        cfg,
    )

    # Arrows top-down
    _arrow(ax, 0.50, 0.90, 0.17, 0.83)
    _arrow(ax, 0.50, 0.90, 0.39, 0.83)
    _arrow(ax, 0.50, 0.90, 0.61, 0.83)
    _arrow(ax, 0.50, 0.90, 0.83, 0.83)

    _arrow(ax, 0.17, 0.76, 0.50, 0.68)
    _arrow(ax, 0.39, 0.76, 0.50, 0.68)
    _arrow(ax, 0.61, 0.76, 0.50, 0.68)
    _arrow(ax, 0.83, 0.76, 0.50, 0.68)

    _arrow(ax, 0.50, 0.61, 0.50, 0.55)
    _arrow(ax, 0.50, 0.48, 0.22, 0.41)
    _arrow(ax, 0.50, 0.48, 0.50, 0.41)
    _arrow(ax, 0.50, 0.48, 0.79, 0.41)

    _arrow(ax, 0.22, 0.34, 0.50, 0.25)
    _arrow(ax, 0.50, 0.34, 0.50, 0.25)
    _arrow(ax, 0.79, 0.34, 0.50, 0.25)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.02)
    return fig


def build() -> plt.Figure:
    fig = render()
    save_figure(fig, CFG.figure_name, supplement=True)
    return fig


if __name__ == "__main__":
    build()
