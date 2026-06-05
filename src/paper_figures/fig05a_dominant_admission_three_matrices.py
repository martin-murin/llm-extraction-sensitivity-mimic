from __future__ import annotations

# Release documentation:
# Builds publication figure module `fig05a_dominant_admission_three_matrices`.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs Figure 5a.
# Usage: `python -m src.paper_figures.fig05a_dominant_admission_three_matrices` or `python scripts/build_paper_figures.py`.

# ruff: noqa: E501

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib import colors

from src.paper_figures.config import CMAP_AGREEMENT_HEATMAP
from src.paper_figures.plot_utils import apply_paper_style, save_figure


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_05a_dominant_admission_three_matrices"
    width_inches: float = 11.2
    height_inches: float = 5.1
    top_n_tags: int = 30
    min_tag_support: int = 10
    dominant_field: str = "dominant_admission_reason"
    offdiag_display_threshold: float = 1e-3
    offdiag_annotate_threshold: float = 0.20
    cmap: str = CMAP_AGREEMENT_HEATMAP
    colorbar_max: float = 1.0
    colorbar_min_log: float = 1e-3
    tick_fontsize: int = 3
    x_tick_rotation_deg: float = 58.0
    diag_fontsize: int = 5
    offdiag_fontsize: int = 4
    grid_linewidth: float = 0.2


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


def _compute_pair_matrices(
    common: pd.DataFrame, tags: list[str], cfg: FigConfig
) -> dict[str, dict[str, Any]]:
    tag_index = {tag: idx for idx, tag in enumerate(tags)}
    pair_data: dict[str, dict[str, Any]] = {}

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

        total = int(counts.sum())
        diag_mass = float(np.trace(counts) / total) if total > 0 else 0.0
        key = f"{left}-{right}"
        pair_data[key] = {
            "counts": counts,
            "rates": rates,
            "diag_counts": np.diag(counts).astype(int),
            "diag_mass": diag_mass,
            "n_pair_notes": total,
            "left": left,
            "right": right,
        }

    return pair_data


def _compute_pair_rates_for_tags(common: pd.DataFrame, tags: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Compute row-normalized pairwise confusion rates for a specific tag order."""
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
    """Load and shape dominant-admission confusion data.

    Returns:
        Dict with:
        - tags: ordered top-N dominant tags
        - pair_data: per-pair confusion counts/rates
        - n_common_hadm: complete A/B/C intersection size
    """
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
    # Match fig06b ordering exactly: rank by highest average confusion
    # (lowest diagonal agreement) across A-B / A-C / B-C, with a minimum support cutoff.
    all_tags = all_values.value_counts().index.tolist()
    full_pair = _compute_pair_rates_for_tags(common, all_tags)
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

    tag_confusion = [row for row in tag_confusion if row[2] >= cfg.min_tag_support]
    tag_confusion.sort(key=lambda x: (-x[1], -x[2], x[0]))
    top_tags = [t for t, _c, _n in tag_confusion[: cfg.top_n_tags]]

    pair_data = _compute_pair_matrices(common, top_tags, cfg)
    return {"tags": top_tags, "pair_data": pair_data, "n_common_hadm": len(common)}


def render(prepped: dict[str, Any], cfg: FigConfig = CFG) -> plt.Figure:
    """Build the three dominant-admission confusion matrices."""
    apply_paper_style()

    tags: list[str] = prepped["tags"]
    display_tags = list(tags)
    pair_data: dict[str, dict[str, Any]] = prepped["pair_data"]

    fig, axes = plt.subplots(
        1, 3, figsize=(cfg.width_inches, cfg.height_inches), constrained_layout=False, sharey=True
    )
    fig.subplots_adjust(left=0.14, right=0.905, bottom=0.26, top=0.83, wspace=0.015)

    cmap = plt.get_cmap(cfg.cmap).copy()
    norm = colors.LogNorm(vmin=cfg.colorbar_min_log, vmax=cfg.colorbar_max)
    eye_mask = np.eye(len(tags), dtype=bool)

    for ax_idx, (ax, (left, right)) in enumerate(zip(axes, PAIR_KEYS, strict=True)):
        key = f"{left}-{right}"
        payload = pair_data[key]
        rates = payload["rates"].copy()

        shown = np.full_like(rates, np.nan, dtype=np.float64)
        # Always show diagonal agreement rates.
        shown[eye_mask] = rates[eye_mask]
        # Show only sufficiently large off-diagonal confusion rates.
        offdiag_visible = (~eye_mask) & (rates >= cfg.offdiag_display_threshold)
        shown[offdiag_visible] = rates[offdiag_visible]

        im = ax.imshow(shown, cmap=cmap, norm=norm, interpolation="none", aspect="equal")

        for i in range(len(tags) + 1):
            ax.axhline(i - 0.5, color="#dddddd", linewidth=cfg.grid_linewidth)
            ax.axvline(i - 0.5, color="#dddddd", linewidth=cfg.grid_linewidth)

        # Intentionally no per-cell text annotations. This panel is color-only to
        # emphasize row-normalized confusion patterns and avoid label clutter.

        ax.set_xticks(np.arange(len(tags)))
        ax.set_yticks(np.arange(len(tags)))
        ax.set_xticklabels(
            display_tags,
            rotation=cfg.x_tick_rotation_deg,
            ha="right",
            rotation_mode="anchor",
            fontsize=cfg.tick_fontsize,
        )
        if ax_idx == 0:
            ax.set_yticklabels(display_tags, fontsize=cfg.tick_fontsize)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(
            f"{left} vs {right}\n"
            f"diag mass={payload['diag_mass'] * 100:.1f}% (n={payload['n_pair_notes']})",
            fontsize=8,
        )

    cbar = fig.colorbar(
        im,
        ax=list(axes),
        fraction=0.025,
        pad=0.015,
    )
    cbar.set_label(
        f"Row-normalized rate (log scale, {cfg.colorbar_min_log:.0e} to {cfg.colorbar_max:.1f})",
        fontsize=7,
    )
    cbar.ax.tick_params(labelsize=6)

    return fig


def build(cfg: FigConfig = CFG) -> plt.Figure:
    """Build and save Figure 6a."""
    prepped = prepare_data(cfg)
    fig = render(prepped, cfg)
    save_figure(fig, cfg.figure_name)
    return fig


if __name__ == "__main__":
    build()
