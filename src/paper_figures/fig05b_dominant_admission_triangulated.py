from __future__ import annotations

# Release documentation:
# Builds publication figure module `fig05b_dominant_admission_triangulated`.
#
# Reads: data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Writes: data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Backs Figure 5b.
# Usage: `python -m src.paper_figures.fig05b_dominant_admission_triangulated` or `python scripts/build_paper_figures.py`.

# ruff: noqa: E501

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib import colors
from matplotlib.patches import Polygon, Rectangle

from src.paper_figures.config import (
    CMAP_AGREEMENT_HEATMAP,
    COLOR_PAIR_AB,
    COLOR_PAIR_AC,
    COLOR_PAIR_BC,
)
from src.paper_figures.plot_utils import apply_paper_style, save_figure


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_05b_dominant_admission_triangulated"
    width_inches: float = 12.8
    height_inches: float = 10.4
    dominant_field: str = "dominant_admission_reason"
    top_n_tags: int = 15
    min_tag_support: int = 10
    cmap: str = CMAP_AGREEMENT_HEATMAP
    vmin: float = 1e-3
    vmax: float = 1.0
    tick_fontsize: int = 6
    title_fontsize: int = 10
    tile_border_linewidth: float = 0.15
    tile_border_color: str = "#e0e0e0"
    wedge_edge_linewidth: float = 0.0
    wedge_edge_color: str = "none"
    legend_tile_linewidth: float = 0.8
    legend_fontsize: int = 8


CFG = FigConfig()
PAIR_KEYS: list[tuple[str, str]] = [("A", "B"), ("A", "C"), ("B", "C")]
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_RESPONSES_DIR = REPO_ROOT / "data" / "raw_responses"
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def _read_split_ids(path: Path) -> set[int]:
    frame = pd.read_csv(path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").dropna().astype("int64").tolist())


def _read_parse_ok_rows(
    run_id: str, allowed_hadm_ids: set[int], dominant_field: str
) -> list[dict[str, Any]]:
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
            rows.append(
                {
                    "hadm_id": hadm_id,
                    "variant": "",
                    dominant_field: feats.get(dominant_field),
                }
            )
    return rows


def _load_combined_threeway_nano_off(dominant_field: str) -> pd.DataFrame:
    """Load A/B/C nano reasoning-OFF rows on disjoint 1k + 500 + extended_5k pools."""
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
            rows.extend(_read_parse_ok_rows(run_id, allowed_ids, dominant_field))
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


def _compute_pair_rates(common: pd.DataFrame, tags: list[str]) -> dict[str, dict[str, np.ndarray]]:
    tag_index = {tag: idx for idx, tag in enumerate(tags)}
    out: dict[str, dict[str, np.ndarray]] = {}
    for left, right in PAIR_KEYS:
        left_values = common[left].astype(str)
        right_values = common[right].astype(str)
        valid = left_values.isin(tags) & right_values.isin(tags)
        left_valid = left_values[valid]
        right_valid = right_values[valid]

        counts = np.zeros((len(tags), len(tags)), dtype=np.int64)
        for lval, rval in zip(left_valid, right_valid, strict=True):
            counts[tag_index[lval], tag_index[rval]] += 1

        row_totals = counts.sum(axis=1, keepdims=True)
        rates = np.divide(
            counts,
            row_totals,
            out=np.zeros_like(counts, dtype=np.float64),
            where=row_totals > 0,
        )
        out[f"{left}-{right}"] = {
            "rates": rates,
            "row_totals": row_totals.astype(np.int64).reshape(-1),
        }
    return out


def prepare_data(cfg: FigConfig = CFG) -> dict[str, Any]:
    """Load and shape the triangulated tile inputs."""
    df = _load_combined_threeway_nano_off(cfg.dominant_field)
    if df.empty:
        raise RuntimeError("No combined three-way extraction rows found.")

    required = {"hadm_id", "variant", cfg.dominant_field}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    pivot = df.pivot_table(
        index="hadm_id",
        columns="variant",
        values=cfg.dominant_field,
        aggfunc="first",
    )
    pivot = pivot.reindex(columns=["A", "B", "C"])
    common = pivot.dropna(subset=["A", "B", "C"]).copy()
    if common.empty:
        raise RuntimeError("No shared hadm_id rows with complete A/B/C dominant labels.")

    all_values = pd.concat([common["A"], common["B"], common["C"]], axis=0).astype(str)
    # Select tags by confusion level: lowest diagonal agreement (highest 1-diag)
    # averaged across available variant-pair rows where the tag appears.
    all_tags = all_values.value_counts().index.tolist()
    full_pair = _compute_pair_rates(common, all_tags)
    tag_confusion: list[tuple[str, float, int]] = []
    for idx, tag in enumerate(all_tags):
        per_pair_disagree: list[float] = []
        for pair_key in ["A-B", "A-C", "B-C"]:
            rates = full_pair[pair_key]["rates"]
            row_totals = full_pair[pair_key]["row_totals"]
            if int(row_totals[idx]) <= 0:
                continue
            diag = float(rates[idx, idx])
            per_pair_disagree.append(1.0 - diag)
        support = int((all_values == tag).sum())
        if per_pair_disagree:
            tag_confusion.append((tag, float(np.mean(per_pair_disagree)), support))

    # Apply minimum support threshold first, then rank by confusion.
    tag_confusion = [row for row in tag_confusion if row[2] >= cfg.min_tag_support]

    # Most confused first; break ties by higher frequency, then stable lexical.
    tag_confusion.sort(key=lambda x: (-x[1], -x[2], x[0]))
    top_tags = [t for t, _c, _n in tag_confusion[: cfg.top_n_tags]]

    pair_payload = _compute_pair_rates(common, top_tags)
    pair_rates = {
        "A-B": pair_payload["A-B"]["rates"],
        "A-C": pair_payload["A-C"]["rates"],
        "B-C": pair_payload["B-C"]["rates"],
    }
    avg_rates = (pair_rates["A-B"] + pair_rates["A-C"] + pair_rates["B-C"]) / 3.0

    return {
        "tags": top_tags,
        "pair_rates": pair_rates,
        "avg_rates": avg_rates,
        "n_common_hadm": len(common),
    }


def _draw_tile(
    ax: plt.Axes,
    x0: float,
    y0: float,
    ab: float,
    ac: float,
    bc: float,
    avg: float,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    cfg: FigConfig,
) -> None:
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    cx = x0 + 0.5
    cy = y0 + 0.5

    wedges = [
        ([(x0, y0), (x1, y0), (cx, cy)], ab),  # top: A-B
        ([(x0, y0), (x0, y1), (cx, cy)], ac),  # left: A-C
        ([(x1, y0), (x1, y1), (cx, cy)], bc),  # right: B-C
        ([(x0, y1), (x1, y1), (cx, cy)], avg),  # bottom: avg
    ]

    for verts, value in wedges:
        color_value = max(float(value), cfg.vmin)
        patch = Polygon(
            verts,
            closed=True,
            facecolor=cmap(norm(color_value)),
            edgecolor=cfg.wedge_edge_color,
            linewidth=cfg.wedge_edge_linewidth,
        )
        ax.add_patch(patch)

    ax.add_patch(
        Rectangle(
            (x0, y0),
            1.0,
            1.0,
            facecolor="none",
            edgecolor=cfg.tile_border_color,
            linewidth=cfg.tile_border_linewidth,
        )
    )


def _draw_legend_tile(ax: plt.Axes, cmap: colors.Colormap, norm: colors.Normalize, cfg: FigConfig) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Compact square legend tile
    x0, y0 = 0.14, 0.06
    size = 0.72
    x1, y1 = x0 + size, y0 + size
    cx, cy = x0 + size / 2, y0 + size / 2

    values = {"ab": 0.80, "ac": 0.60, "bc": 0.40, "avg": 0.60}
    wedges = [
        ([(x0, y0), (x1, y0), (cx, cy)], values["ab"], "A→B"),
        ([(x0, y0), (x0, y1), (cx, cy)], values["ac"], "A→C"),
        ([(x1, y0), (x1, y1), (cx, cy)], values["bc"], "B→C"),
        ([(x0, y1), (x1, y1), (cx, cy)], values["avg"], "Avg"),
    ]
    for verts, value, _label in wedges:
        ax.add_patch(
            Polygon(
                verts,
                closed=True,
                facecolor=cmap(norm(value)),
                edgecolor="white",
                linewidth=0.6,
            )
        )

    ax.add_patch(
        Rectangle(
            (x0, y0),
            size,
            size,
            facecolor="none",
            edgecolor="#444444",
            linewidth=cfg.legend_tile_linewidth,
        )
    )

    ax.text(
        cx,
        y0 - 0.03,
        "top: A→B",
        ha="center",
        va="bottom",
        fontsize=cfg.legend_fontsize,
        color="#000000",
    )
    ax.text(
        x0 - 0.03,
        cy,
        "left:\nA→C",
        ha="right",
        va="center",
        fontsize=cfg.legend_fontsize,
        color="#000000",
    )
    ax.text(
        x1 + 0.03,
        cy,
        "right:\nB→C",
        ha="left",
        va="center",
        fontsize=cfg.legend_fontsize,
        color="#000000",
    )
    ax.text(
        cx,
        y1 + 0.10,
        "bottom: avg",
        ha="center",
        va="top",
        fontsize=cfg.legend_fontsize,
        color="#000000",
    )


def render(prepped: dict[str, Any], cfg: FigConfig = CFG) -> plt.Figure:
    """Render 30x30 triangulated confusion tiles."""
    apply_paper_style()

    tags: list[str] = prepped["tags"]
    display_tags = tags
    pair_rates: dict[str, np.ndarray] = prepped["pair_rates"]
    avg_rates: np.ndarray = prepped["avg_rates"]

    norm = colors.LogNorm(vmin=cfg.vmin, vmax=cfg.vmax)
    cmap = plt.get_cmap(cfg.cmap)

    fig = plt.figure(figsize=(cfg.width_inches, cfg.height_inches), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[6.6, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    # Keep both helper elements high in the right column; enlarge the colorbar
    # host panel to ~150% of its original height allocation.
    side = gs[0, 1].subgridspec(3, 1, height_ratios=[1.0, 1.8, 1.2], hspace=0.20)
    legend_ax = fig.add_subplot(side[0, 0])
    cax_host = fig.add_subplot(side[1, 0])
    fig.add_subplot(side[2, 0]).axis("off")
    cax_host.set_axis_off()
    # Thin colorbar centered in its host panel.
    cax = cax_host.inset_axes((0.42, 0.03, 0.16, 0.94))

    n = len(tags)
    for i in range(n):
        for j in range(n):
            _draw_tile(
                ax=ax,
                x0=float(j),
                y0=float(i),
                ab=float(pair_rates["A-B"][i, j]),
                ac=float(pair_rates["A-C"][i, j]),
                bc=float(pair_rates["B-C"][i, j]),
                avg=float(avg_rates[i, j]),
                cmap=cmap,
                norm=norm,
                cfg=cfg,
            )

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(
        display_tags,
        rotation=35,
        ha="right",
        fontsize=cfg.tick_fontsize,
        fontfamily="monospace",
    )
    ax.set_yticklabels(display_tags, fontsize=cfg.tick_fontsize, fontfamily="monospace")
    # Title intentionally omitted; figure caption carries context in the paper.

    _draw_legend_tile(legend_ax, cmap, norm, cfg)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Row-normalized agreement rate", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return fig


def build(cfg: FigConfig = CFG) -> plt.Figure:
    """Build and save Figure 6b."""
    prepped = prepare_data(cfg)
    fig = render(prepped, cfg)
    save_figure(fig, cfg.figure_name)
    return fig


if __name__ == "__main__":
    build()
