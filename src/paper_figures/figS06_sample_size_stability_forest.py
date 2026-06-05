from __future__ import annotations

# Release documentation:
# Builds publication figure module `figS06_sample_size_stability_forest`.
#
# Reads: data/raw_responses/*/results.jsonl, data/splits/*.csv.
# Writes: data/raw_responses/*/results.jsonl, data/splits/*.csv.
# Backs Supplement Figure S6.
# Usage: `python -m src.paper_figures.figS06_sample_size_stability_forest` or `python scripts/build_paper_figures.py`.

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from paper.claims.scripts._threeway_kappa_primary import SAMPLE_SPECS, recompute_sample_details

from src.paper_figures.config import DOUBLE_COLUMN_WIDTH
from src.paper_figures.config import (
    COLOR_SAMPLE_EXTENDED,
    COLOR_SAMPLE_HOLDOUT,
    COLOR_SAMPLE_METH_1K,
    COLOR_SAMPLE_METH_5K_AUDIT,
    COLOR_SAMPLE_REFINEMENT,
)
from src.paper_figures.plot_utils import apply_paper_style, save_figure


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_S06_sample_size_stability_forest"
    width_inches: float = DOUBLE_COLUMN_WIDTH
    height_inches: float = 4.4
    title: str = ""

    n_bootstrap: int = 5000
    seed: int = 1106

    # Progressive palette from early/small to later/larger samples.
    sample_colors: tuple[str, ...] = (
        COLOR_SAMPLE_REFINEMENT,
        COLOR_SAMPLE_HOLDOUT,
        COLOR_SAMPLE_METH_1K,
        COLOR_SAMPLE_METH_5K_AUDIT,
        COLOR_SAMPLE_EXTENDED,
    )
    point_size: float = 30.0
    line_width: float = 1.4
    x_min: float | None = 0.50
    x_max: float | None = 0.80
    x_margin: float = 0.008
    x_tick_step: float = 0.05
    label_y_offset: float = -0.24
    top_padding: float = 0.90


CFG = FigConfig()


def prepare_data(cfg: FigConfig = CFG) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    order = [
        "refinement_150",
        "holdout_150",
        "methodology_1k",
        "methodology_5k_audit_500",
        "extended_5k",
    ]
    label_map = {
        "refinement_150": "Refinement 150",
        "holdout_150": "Holdout 150",
        "methodology_1k": "Methodology 1k",
        "methodology_5k_audit_500": "Methodology 5k-audit 500",
        "extended_5k": "Extended 5k",
    }

    rows: list[dict[str, float | int | str]] = []
    for sample_key in order:
        run_ids, split_csv = SAMPLE_SPECS[sample_key]
        details = recompute_sample_details(sample_key, run_ids, split_csv)
        arr = np.asarray(details.filtered_kappa_values, dtype=np.float64)
        n = int(arr.size)
        if n <= 0:
            raise ValueError(f"No filtered kappa values for sample: {sample_key}")
        idx = rng.integers(0, n, size=(int(cfg.n_bootstrap), n))
        sampled = arr[idx]
        boot_meds = np.median(sampled, axis=1)
        ci_low = float(np.percentile(boot_meds, 2.5))
        ci_high = float(np.percentile(boot_meds, 97.5))
        rows.append(
            {
                "sample_key": sample_key,
                "sample_label": label_map[sample_key],
                "n_notes": int(details.n_intersection_notes),
                "n_fields_filtered": int(details.n_fields_filtered),
                "median_kappa_filtered": float(details.median_kappa_filtered),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    df = pd.DataFrame(rows)
    df["sample_key"] = pd.Categorical(df["sample_key"], categories=order, ordered=True)
    return df.sort_values("sample_key").reset_index(drop=True)


def render(df: pd.DataFrame, cfg: FigConfig = CFG) -> plt.Figure:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(cfg.width_inches, cfg.height_inches))

    y = np.arange(len(df))
    colors = list(cfg.sample_colors)
    x = df["median_kappa_filtered"].to_numpy(dtype=float)
    lo = df["ci_low"].to_numpy(dtype=float)
    hi = df["ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([x - lo, hi - x])

    for i in range(len(df)):
        ax.errorbar(
            x[i],
            y[i],
            xerr=np.asarray([[xerr[0, i]], [xerr[1, i]]], dtype=float),
            fmt="o",
            ms=np.sqrt(cfg.point_size),
            color=colors[i],
            ecolor=colors[i],
            elinewidth=cfg.line_width,
            capsize=2.2,
            capthick=cfg.line_width,
            markeredgecolor="#1f1f1f",
            markeredgewidth=0.45,
            zorder=3,
        )
        ax.text(
            x[i],
            y[i] + cfg.label_y_offset,
            f"{x[i]:.3f}",
            va="bottom",
            ha="center",
            fontsize=7,
            color="#2a2a2a",
        )

    labels = [
        f"{row.sample_label} (n={int(row.n_notes)})"
        for row in df.itertuples(index=False)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    lo_min = float(np.min(lo))
    hi_max = float(np.max(hi))
    x_min = cfg.x_min if cfg.x_min is not None else lo_min - cfg.x_margin
    x_max = cfg.x_max if cfg.x_max is not None else hi_max + cfg.x_margin
    ax.set_xlim(x_min, x_max)

    tick_start = np.floor(x_min / cfg.x_tick_step) * cfg.x_tick_step
    tick_end = np.ceil(x_max / cfg.x_tick_step) * cfg.x_tick_step
    ticks = np.arange(tick_start, tick_end + (cfg.x_tick_step / 2.0), cfg.x_tick_step)
    ax.set_xticks(ticks.tolist())
    ax.set_xlabel("Filtered median cross-variant κ")
    ax.set_title(cfg.title)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(len(df) - 0.5, -cfg.top_padding)
    legend_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="#444444",
        markerfacecolor="#ffffff",
        markersize=5,
        linewidth=1.2,
        label="Point estimate ± 95% bootstrap CI",
    )
    ax.legend(
        handles=[legend_handle],
        loc="upper right",
        frameon=False,
        fontsize=7,
    )

    fig.subplots_adjust(left=0.34, right=0.98, top=0.95, bottom=0.15)
    return fig


def build() -> plt.Figure:
    df = prepare_data()
    fig = render(df)
    save_figure(fig, CFG.figure_name, supplement=True)
    return fig


if __name__ == "__main__":
    build()
