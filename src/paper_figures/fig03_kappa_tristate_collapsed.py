from __future__ import annotations

# Release documentation:
# Builds publication figure module `fig03_kappa_tristate_collapsed`.
#
# Reads: data/raw_responses/methodology_1k_{a, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_{b, data/raw_responses/extended_5k_{b, data/raw_responses/production_v1/results.jsonl, data/splits/{methodology_1k.
# Writes: data/raw_responses/methodology_1k_{a, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_{b, data/raw_responses/extended_5k_{b, data/raw_responses/production_v1/results.jsonl, data/splits/{methodology_1k.
# Backs Figure 3.
# Usage: `python -m src.paper_figures.fig03_kappa_tristate_collapsed` or `python scripts/build_paper_figures.py`.

# ruff: noqa: E501

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from src.paper_figures import config
from src.paper_figures.data_loaders import get_tristate_field_list
from src.paper_figures.plot_utils import (
    apply_paper_style,
    format_kappa_axis,
    save_figure,
)


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_03_tristate_kappa_full_vs_collapsed"
    width_inches: float = config.FULL_PAGE_WIDTH
    height_inches: float = 7.2

    pair_order: tuple[str, ...] = ("A-B", "A-C", "B-C")
    variant_order: tuple[str, ...] = ("A", "B", "C")

    pair_colors: dict[str, str] = field(
        default_factory=lambda: {
            "A-B": config.COLOR_PAIR_AB,
            "A-C": config.COLOR_PAIR_AC,
            "B-C": config.COLOR_PAIR_BC,
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
    legend_fontsize: float = 7.0
    legend_title_fontsize: float = 7.0

    # Order fields by mean difference (collapsed - TriState), ascending.
    # With shared y-axis and inverted y orientation, this puts smallest at top.
    sort_field: str = "kappa_delta_mean"

    # Delta panel axis settings.
    delta_xmin: float | None = None
    delta_xmax: float | None = None
    delta_xtick_step: float = 0.20


CFG = FigConfig()
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_RESPONSES_DIR = REPO_ROOT / "data" / "raw_responses"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

TRISTATE_SET = {"yes", "no", "not_documented"}


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
    """Load A/B/C nano/OFF rows on disjoint 1k + 500 + extended_5k pools."""
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


def _norm_tristate(v: Any) -> str | None:
    if isinstance(v, str) and v in TRISTATE_SET:
        return v
    return None


def _enc_full(v: str | None) -> int | None:
    if v == "yes":
        return 1
    if v == "no":
        return 0
    if v == "not_documented":
        return 2
    return None


def _enc_collapsed(v: str | None) -> int | None:
    if v == "yes":
        return 1
    if v in {"no", "not_documented"}:
        return 0
    return None


def _pair_kappa(left_raw: list[Any], right_raw: list[Any], collapsed: bool) -> float:
    left_enc: list[int] = []
    right_enc: list[int] = []
    for lv_raw, rv_raw in zip(left_raw, right_raw, strict=True):
        lv = _norm_tristate(lv_raw)
        rv = _norm_tristate(rv_raw)
        if collapsed:
            le = _enc_collapsed(lv)
            re = _enc_collapsed(rv)
        else:
            le = _enc_full(lv)
            re = _enc_full(rv)
        if le is None or re is None:
            continue
        left_enc.append(le)
        right_enc.append(re)

    if len(left_enc) < 2:
        return float("nan")
    return float(cohen_kappa_score(left_enc, right_enc))


def prepare_data(cfg: FigConfig = CFG) -> dict[str, Any]:
    frame = _load_combined_threeway_nano_off()
    frame = frame[frame["variant"].isin(cfg.variant_order)].copy()

    fields = [f for f in get_tristate_field_list("base") if f in frame.columns]
    if not fields:
        raise RuntimeError("No base TriState fields found in combined frame.")

    by_variant = {
        variant: frame[frame["variant"] == variant].drop_duplicates(subset=["hadm_id"], keep="last")
        for variant in cfg.variant_order
    }

    common_ids = set(by_variant[cfg.variant_order[0]]["hadm_id"])
    for variant in cfg.variant_order[1:]:
        common_ids &= set(by_variant[variant]["hadm_id"])
    common = sorted(common_ids)
    if not common:
        raise RuntimeError("No shared hadm_id intersection across A/B/C for fig04 kappa.")

    for variant in cfg.variant_order:
        by_variant[variant] = by_variant[variant][by_variant[variant]["hadm_id"].isin(common)].copy()

    pair_map = {"A-B": ("A", "B"), "A-C": ("A", "C"), "B-C": ("B", "C")}
    records: list[dict[str, Any]] = []
    for field_name in fields:
        row: dict[str, Any] = {"field": field_name}
        full_vals: list[float] = []
        col_vals: list[float] = []
        for pair in cfg.pair_order:
            lv, rv = pair_map[pair]
            left = by_variant[lv].set_index("hadm_id")[field_name].reindex(common).tolist()
            right = by_variant[rv].set_index("hadm_id")[field_name].reindex(common).tolist()
            k_full = _pair_kappa(left, right, collapsed=False)
            k_col = _pair_kappa(left, right, collapsed=True)
            k_delta = (
                (k_col - k_full)
                if np.isfinite(k_col) and np.isfinite(k_full)
                else float("nan")
            )
            row[f"kappa_full_{pair}"] = k_full
            row[f"kappa_collapsed_{pair}"] = k_col
            row[f"kappa_delta_{pair}"] = k_delta
            full_vals.append(k_full)
            col_vals.append(k_col)
        row["kappa_full_mean"] = float(np.nanmean(np.asarray(full_vals, dtype=float)))
        row["kappa_collapsed_mean"] = float(np.nanmean(np.asarray(col_vals, dtype=float)))
        row["kappa_delta_mean"] = row["kappa_collapsed_mean"] - row["kappa_full_mean"]
        records.append(row)

    plot_df = pd.DataFrame.from_records(records)
    plot_df = plot_df.sort_values(by=cfg.sort_field, ascending=True, kind="mergesort").reset_index(drop=True)
    plot_df["label"] = plot_df["field"].map(lambda x: str(x))

    return {
        "plot_df": plot_df,
        "n_common": len(common),
    }


def _draw_panel(
    ax: plt.Axes,
    *,
    panel_df: pd.DataFrame,
    field_labels: list[str],
    value_prefix: str,
    cfg: FigConfig,
    title: str,
    show_y_labels: bool,
    is_delta_panel: bool = False,
) -> None:
    y_base = np.arange(len(field_labels), dtype=np.float64)
    offsets = np.array([-cfg.bar_offset_step, 0.0, cfg.bar_offset_step], dtype=np.float64)

    for i, pair in enumerate(cfg.pair_order):
        vals = panel_df[f"{value_prefix}_{pair}"].to_numpy(dtype=np.float64)
        ax.barh(
            y_base + offsets[i],
            vals,
            height=cfg.bar_height,
            color=cfg.pair_colors[pair],
            edgecolor="#444444",
            linewidth=0.3,
            hatch=cfg.pair_hatches[pair],
            label=pair,
        )

    if is_delta_panel:
        delta_cols = [f"{value_prefix}_{pair}" for pair in cfg.pair_order]
        delta_vals = panel_df[delta_cols].to_numpy(dtype=np.float64).ravel()
        finite = delta_vals[np.isfinite(delta_vals)]
        if finite.size == 0:
            x_min_data = -0.2
            x_max_data = 0.2
        else:
            x_min_data = float(np.min(finite))
            x_max_data = float(np.max(finite))

        step = cfg.delta_xtick_step
        pad = 0.02
        x_min_auto = np.floor((x_min_data - pad) / step) * step
        x_max_auto = np.ceil((x_max_data + pad) / step) * step
        x_min = cfg.delta_xmin if cfg.delta_xmin is not None else float(x_min_auto)
        x_max = cfg.delta_xmax if cfg.delta_xmax is not None else float(x_max_auto)
        if x_min >= x_max:
            x_min, x_max = -0.2, 0.2

        ax.set_xlim(x_min, x_max)
        ax.set_xticks(np.arange(x_min, x_max + (step / 2.0), step))
        ax.axvline(0.0, color="#444444", linewidth=0.8, linestyle="-", alpha=0.8)
        ax.set_xlabel(r"$\kappa^{collapsed} - \kappa^{TriState}$")
    else:
        format_kappa_axis(ax, ymin=0.0, ymax=1.0)
        ax.set_xlim(0.0, 1.1)
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))
        # Remove dashed zero-reference line; it adds clutter here.
        for ln in list(ax.lines):
            if ln.get_linestyle() == "--":
                ln.remove()
        if value_prefix == "kappa_full":
            ax.set_xlabel(r"$\kappa^{TriState}$")
        else:
            ax.set_xlabel(r"$\kappa^{collapsed}$")
    ax.set_title(title)
    ax.set_yticks(y_base)
    if show_y_labels:
        ax.set_yticklabels(field_labels, fontsize=7, fontfamily="monospace")
    else:
        ax.tick_params(axis="y", labelleft=False)
    # Add top/bottom vertical padding so first/last rows are fully visible.
    y_min = float(np.min(y_base + offsets.min()) - cfg.bar_height / 2.0)
    y_max = float(np.max(y_base + offsets.max()) + cfg.bar_height / 2.0)
    y_span = max(1e-9, y_max - y_min)
    top_pad = 0.08 * y_span
    bottom_pad = 0.05 * y_span
    ax.set_ylim(y_max + bottom_pad, y_min - top_pad)
    ax.grid(axis="x", alpha=config.GRID_ALPHA)
    ax.grid(axis="y", visible=False)


def render(data: dict[str, Any], cfg: FigConfig = CFG) -> plt.Figure:
    apply_paper_style()

    plot_df: pd.DataFrame = data["plot_df"]
    labels = plot_df["label"].tolist()

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(cfg.width_inches, cfg.height_inches),
        sharey=True,
        constrained_layout=False,
    )

    _draw_panel(
        axes[0],
        panel_df=plot_df,
        field_labels=labels,
        value_prefix="kappa_full",
        cfg=cfg,
        title="TriState kappa",
        show_y_labels=True,
    )
    _draw_panel(
        axes[1],
        panel_df=plot_df,
        field_labels=labels,
        value_prefix="kappa_collapsed",
        cfg=cfg,
        title="Collapsed kappa",
        show_y_labels=False,
    )
    _draw_panel(
        axes[2],
        panel_df=plot_df,
        field_labels=labels,
        value_prefix="kappa_delta",
        cfg=cfg,
        title="Difference",
        show_y_labels=False,
        is_delta_panel=True,
    )

    handles, labels_legend = axes[2].get_legend_handles_labels()
    uniq = {}
    for handle, label in zip(handles, labels_legend, strict=True):
        if label not in uniq:
            uniq[label] = handle
    axes[2].legend(
        list(uniq.values()),
        list(uniq.keys()),
        title="Variant pairs",
        loc="upper right",
        frameon=True,
        fontsize=cfg.legend_fontsize,
        title_fontsize=cfg.legend_title_fontsize,
    )

    fig.subplots_adjust(left=0.32, right=0.98, top=0.92, bottom=0.10, wspace=0.12)
    return fig


def build(cfg: FigConfig = CFG) -> tuple[plt.Figure, list[Path]]:
    data = prepare_data(cfg)
    fig = render(data, cfg)
    outputs = save_figure(fig, cfg.figure_name)
    plt.close(fig)
    return fig, outputs


if __name__ == "__main__":
    build()
