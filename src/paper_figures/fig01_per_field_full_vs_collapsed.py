from __future__ import annotations

# Release documentation:
# Builds publication figure module `fig01_per_field_full_vs_collapsed`.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs Figure 1.
# Usage: `python -m src.paper_figures.fig01_per_field_full_vs_collapsed` or `python scripts/build_paper_figures.py`.

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from paper.claims.scripts._model_size_primary import (
    load_paired_model_size_data,
    per_field_model_size_deltas,
)
from src.paper_figures.config import (
    FONT_SIZE_ANNOTATION,
    FULL_PAGE_WIDTH,
    COLOR_SEM_COLLAPSED,
    COLOR_SEM_TRISTATE,
)
from src.paper_figures.plot_utils import apply_paper_style, save_figure


# =============================================================================
# CONFIGURATION - edit values here for cosmetic changes
# =============================================================================


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_01_per_field_full_vs_collapsed"
    width_inches: float = FULL_PAGE_WIDTH
    min_height_inches: float = 4.8
    per_field_height_inches: float = 0.30
    title: str = ""
    subtitle: str = ""
    x_label: str = r"Per-field model-size difference,  $\Delta\kappa_f$  [pp]"

    # Sorting and filtering knobs.
    sort_by: str = "delta_collapsed_pp"
    sort_descending: bool = True
    top_n: int | None = None

    # Visual knobs.
    bar_alpha_full: float = 0.45
    bar_alpha_collapsed: float = 0.50
    bar_height: float = 0.38
    median_full_style: str = "--"
    median_collapsed_style: str = "-."
    median_line_width: float = 1.1
    median_label_y_axes: float = 0.01
    median_label_x_offset: float = 1.50
    top_margin: float = 0.84
    bottom_margin: float = 0.20

    # Balanced contrast pair (no "before/after" visual bias).
    color_full: str = COLOR_SEM_TRISTATE
    color_collapsed: str = COLOR_SEM_COLLAPSED
    ci_color_darken_factor: float = 0.60


CFG = FigConfig()
CLAIMS_PATH = Path(__file__).resolve().parents[2] / "paper" / "claims" / "claims.json"


# =============================================================================
# DATA PREP
# =============================================================================


def prepare_data(cfg: FigConfig = CFG) -> pd.DataFrame:
    """Load and shape per-field delta table for plotting.

    Returns:
        DataFrame with plotting columns:
        - field
        - label
        - delta_full_pp
        - delta_collapsed_pp
    """
    data = load_paired_model_size_data(base_rate_threshold=10)
    full_delta_pp, collapsed_delta_pp = per_field_model_size_deltas(data)
    ci_lookup = _load_ci_lookup()
    rows = []
    included = set(data.included_fields)
    for field in data.tri_fields:
        ci = ci_lookup.get(field, {})
        rows.append(
            {
                "field": field,
                "delta_full_pp": float(full_delta_pp[field]),
                "delta_collapsed_pp": float(collapsed_delta_pp[field]),
                "included_in_median": field in included,
                "delta_full_ci_low_pp": ci.get("full_low"),
                "delta_full_ci_high_pp": ci.get("full_high"),
                "delta_collapsed_ci_low_pp": ci.get("collapsed_low"),
                "delta_collapsed_ci_high_pp": ci.get("collapsed_high"),
            }
        )
    frame = pd.DataFrame(rows)

    frame = frame.sort_values(cfg.sort_by, ascending=not cfg.sort_descending, kind="mergesort")
    if cfg.top_n is not None:
        frame = frame.head(cfg.top_n).copy()

    frame["label"] = frame["field"]
    frame = frame.reset_index(drop=True)
    return frame


def _load_ci_lookup() -> dict[str, dict[str, float]]:
    """Load per-field CI bounds from the claims registry.

    Expected keys per field:
    - <field>_delta_full_ci_low_pp / _high_pp
    - <field>_delta_collapsed_ci_low_pp / _high_pp
    """
    if not CLAIMS_PATH.exists():
        return {}
    payload = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for key, obj in payload.items():
        if not key.endswith(("_delta_full_ci_low_pp", "_delta_full_ci_high_pp", "_delta_collapsed_ci_low_pp", "_delta_collapsed_ci_high_pp")):
            continue
        field = key
        suffix = ""
        for candidate in (
            "_delta_full_ci_low_pp",
            "_delta_full_ci_high_pp",
            "_delta_collapsed_ci_low_pp",
            "_delta_collapsed_ci_high_pp",
        ):
            if key.endswith(candidate):
                field = key[: -len(candidate)]
                suffix = candidate
                break
        value = obj.get("value") if isinstance(obj, dict) else None
        if not isinstance(value, (int, float)):
            continue
        slot = out.setdefault(field, {})
        if suffix == "_delta_full_ci_low_pp":
            slot["full_low"] = float(value)
        elif suffix == "_delta_full_ci_high_pp":
            slot["full_high"] = float(value)
        elif suffix == "_delta_collapsed_ci_low_pp":
            slot["collapsed_low"] = float(value)
        elif suffix == "_delta_collapsed_ci_high_pp":
            slot["collapsed_high"] = float(value)
    return out


def _darken_color(hex_color: str, factor: float) -> str:
    """Return a darker shade of a hex color (factor in [0,1], lower=darker)."""
    r, g, b = to_rgb(hex_color)
    return to_hex((r * factor, g * factor, b * factor))


# =============================================================================
# RENDERING
# =============================================================================


def render(df: pd.DataFrame, cfg: FigConfig = CFG) -> plt.Figure:
    """Render horizontal paired bars with median reference lines."""
    apply_paper_style()

    n_fields = len(df)
    height = max(cfg.min_height_inches, n_fields * cfg.per_field_height_inches + 1.2)
    fig, ax = plt.subplots(figsize=(cfg.width_inches, height))

    y = np.arange(n_fields, dtype=float)
    # Offset bars slightly so both metrics are visible for each field.
    y_full = y - cfg.bar_height / 2
    y_collapsed = y + cfg.bar_height / 2

    ax.barh(
        y_full,
        df["delta_full_pp"].to_numpy(dtype=float),
        height=cfg.bar_height,
        color=cfg.color_full,
        alpha=cfg.bar_alpha_full,
        label=r"TriState $\Delta\kappa_f$",
    )
    ax.barh(
        y_collapsed,
        df["delta_collapsed_pp"].to_numpy(dtype=float),
        height=cfg.bar_height,
        color=cfg.color_collapsed,
        alpha=cfg.bar_alpha_collapsed,
        label=r"Collapsed $\Delta\kappa_f$",
    )

    # Draw horizontal 95% CI whiskers around each bar value. Bounds come from
    # paired bootstrap claims in paper/claims/claims.json.
    full_vals = df["delta_full_pp"].to_numpy(dtype=float)
    full_low = df["delta_full_ci_low_pp"].to_numpy(dtype=float)
    full_high = df["delta_full_ci_high_pp"].to_numpy(dtype=float)
    full_mask = np.isfinite(full_vals) & np.isfinite(full_low) & np.isfinite(full_high)
    if np.any(full_mask):
        full_ci_color = _darken_color(cfg.color_full, cfg.ci_color_darken_factor)
        full_xerr = np.vstack(
            [
                np.maximum(0.0, full_vals[full_mask] - full_low[full_mask]),
                np.maximum(0.0, full_high[full_mask] - full_vals[full_mask]),
            ]
        )
        ax.errorbar(
            full_vals[full_mask],
            y_full[full_mask],
            xerr=full_xerr,
            fmt="none",
            ecolor=full_ci_color,
            elinewidth=1.1,
            capsize=2.5,
            capthick=1.1,
            zorder=4,
        )

    collapsed_vals = df["delta_collapsed_pp"].to_numpy(dtype=float)
    collapsed_low = df["delta_collapsed_ci_low_pp"].to_numpy(dtype=float)
    collapsed_high = df["delta_collapsed_ci_high_pp"].to_numpy(dtype=float)
    collapsed_mask = (
        np.isfinite(collapsed_vals)
        & np.isfinite(collapsed_low)
        & np.isfinite(collapsed_high)
    )
    if np.any(collapsed_mask):
        collapsed_ci_color = _darken_color(cfg.color_collapsed, cfg.ci_color_darken_factor)
        collapsed_xerr = np.vstack(
            [
                np.maximum(0.0, collapsed_vals[collapsed_mask] - collapsed_low[collapsed_mask]),
                np.maximum(0.0, collapsed_high[collapsed_mask] - collapsed_vals[collapsed_mask]),
            ]
        )
        ax.errorbar(
            collapsed_vals[collapsed_mask],
            y_collapsed[collapsed_mask],
            xerr=collapsed_xerr,
            fmt="none",
            ecolor=collapsed_ci_color,
            elinewidth=1.1,
            capsize=2.5,
            capthick=1.1,
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontfamily="monospace")
    ax.invert_yaxis()  # Highest collapsed delta stays at the top for scanability.

    included = df[df["included_in_median"]].copy()
    full_median = float(np.median(included["delta_full_pp"].to_numpy(dtype=float)))
    collapsed_median = float(np.median(included["delta_collapsed_pp"].to_numpy(dtype=float)))

    ax.axvline(
        full_median,
        color=cfg.color_full,
        linestyle=cfg.median_full_style,
        linewidth=cfg.median_line_width,
        alpha=0.9,
        label=r"TriState filtered median  $\overline{\Delta\kappa}$",
    )
    ax.axvline(
        collapsed_median,
        color=cfg.color_collapsed,
        linestyle=cfg.median_collapsed_style,
        linewidth=cfg.median_line_width,
        alpha=0.9,
        label=r"Collapsed filtered median  $\overline{\Delta\kappa}$",
    )

    # Place median labels near the bottom INSIDE plotting area; label anchors are
    # intentionally opposite so they flank their respective vertical lines.
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        full_median - cfg.median_label_x_offset,
        cfg.median_label_y_axes,
        rf"$\overline{{\Delta\kappa}}$: {full_median:+.2f} pp",
        transform=blend,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZE_ANNOTATION,
        color=cfg.color_full,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.0},
    )
    ax.text(
        collapsed_median + cfg.median_label_x_offset,
        cfg.median_label_y_axes,
        rf"$\overline{{\Delta\kappa}}$: {collapsed_median:+.2f} pp",
        transform=blend,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE_ANNOTATION,
        color=cfg.color_collapsed,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.0},
    )

    ax.set_xlabel(cfg.x_label)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cfg.color_full, alpha=cfg.bar_alpha_full, label=r"TriState $\Delta\kappa_f$"),
        plt.Rectangle((0, 0), 1, 1, color=cfg.color_collapsed, alpha=cfg.bar_alpha_collapsed, label=r"Collapsed $\Delta\kappa_f$"),
        Line2D([0], [0], color=cfg.color_full, linestyle=cfg.median_full_style, linewidth=cfg.median_line_width, label=r"TriState filtered median  $\overline{\Delta\kappa}$"),
        Line2D([0], [0], color=cfg.color_collapsed, linestyle=cfg.median_collapsed_style, linewidth=cfg.median_line_width, label=r"Collapsed filtered median  $\overline{\Delta\kappa}$"),
    ]
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(0.98, 0.50), frameon=True)

    fig.subplots_adjust(top=0.96, bottom=cfg.bottom_margin)
    return fig


# =============================================================================
# MAIN
# =============================================================================


def build() -> plt.Figure:
    """Build and save Figure 3."""
    df = prepare_data()
    fig = render(df)
    save_figure(fig, CFG.figure_name)
    return fig


if __name__ == "__main__":
    build()
