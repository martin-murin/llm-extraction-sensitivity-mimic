from __future__ import annotations

# Release documentation:
# Provides shared helpers/configuration for publication figure modules.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Supports publication figure generation.

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.paper_figures.config import CMAP_AGREEMENT_HEATMAP, DOUBLE_COLUMN_WIDTH
from src.paper_figures.plot_utils import apply_paper_style, save_figure

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_RESPONSES_DIR = REPO_ROOT / "data" / "raw_responses"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

PAIR_KEYS: tuple[tuple[str, str], ...] = (("A", "B"), ("A", "C"), ("B", "C"))


@dataclass(frozen=True)
class EnumPairFigConfig:
    figure_name: str
    field: str
    pretty_name: str
    labels: tuple[str, ...]
    width_inches: float = DOUBLE_COLUMN_WIDTH
    height_inches: float = 3.2
    tick_fontsize: float = 7.0
    grid_linewidth: float = 0.25


def _read_split_ids(path: Path) -> set[int]:
    frame = pd.read_csv(path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").dropna().astype("int64").tolist())


def _read_variant_field(
    run_id: str,
    field: str,
    allowed_hadm_ids: set[int],
    allowed_labels: set[str],
) -> dict[int, str]:
    out: dict[int, str] = {}
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
            value = str(feats.get(field, "not_documented"))
            out[hadm_id] = value if value in allowed_labels else "not_documented"
    return out


def _load_pooled_enum_frame(cfg: EnumPairFigConfig) -> pd.DataFrame:
    ids_1k = _read_split_ids(SPLITS_DIR / "methodology_1k.csv")
    ids_500 = _read_split_ids(SPLITS_DIR / "methodology_5k_audit_500.csv")
    ids_ext = _read_split_ids(SPLITS_DIR / "extended_5k.csv")
    allowed_labels = set(cfg.labels)

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

    by_variant: dict[str, dict[int, str]] = {}
    for variant, parts in run_map.items():
        merged: dict[int, str] = {}
        for run_id, allowed_ids in parts:
            merged.update(_read_variant_field(run_id, cfg.field, allowed_ids, allowed_labels))
        by_variant[variant] = merged

    common_ids = sorted(set.intersection(*(set(by_variant[v]) for v in ("A", "B", "C"))))
    if not common_ids:
        raise RuntimeError(f"No shared A/B/C hadm_id rows for S07 field '{cfg.field}'.")

    return pd.DataFrame(
        {
            "hadm_id": common_ids,
            "A": [by_variant["A"][hid] for hid in common_ids],
            "B": [by_variant["B"][hid] for hid in common_ids],
            "C": [by_variant["C"][hid] for hid in common_ids],
        }
    )


def _compute_confusion(
    frame: pd.DataFrame,
    labels: tuple[str, ...],
    left: str,
    right: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    label_index = {label: idx for idx, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    left_values = frame[left].astype(str).tolist()
    right_values = frame[right].astype(str).tolist()
    for lv, rv in zip(left_values, right_values, strict=True):
        if lv not in label_index or rv not in label_index:
            continue
        counts[label_index[lv], label_index[rv]] += 1

    row_totals = counts.sum(axis=1, keepdims=True)
    rates = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_totals > 0,
    )
    return counts, rates, int(counts.sum())


def build_enum_pairwise_figure(cfg: EnumPairFigConfig) -> Figure:
    apply_paper_style()
    pooled = _load_pooled_enum_frame(cfg)

    fig = plt.figure(figsize=(cfg.width_inches, cfg.height_inches))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.045], wspace=0.20)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    all_rates: list[np.ndarray] = []
    totals: list[int] = []
    for left, right in PAIR_KEYS:
        _counts, rates, total = _compute_confusion(pooled, cfg.labels, left, right)
        all_rates.append(rates)
        totals.append(total)

    stack = np.concatenate([r.flatten() for r in all_rates], axis=0)
    positive = stack[stack > 0]
    vmin = max(float(np.min(positive)) if positive.size else 1e-3, 1e-3)
    norm = LogNorm(vmin=vmin, vmax=1.0)
    cmap = plt.get_cmap(CMAP_AGREEMENT_HEATMAP)

    im = None
    for i, ((left, right), rates, total) in enumerate(zip(PAIR_KEYS, all_rates, totals, strict=True)):
        ax = axes[i]
        im = ax.imshow(rates, cmap=cmap, norm=norm, interpolation="none", aspect="equal")
        n = len(cfg.labels)
        for g in range(n + 1):
            ax.axhline(g - 0.5, color="#d6d6d6", linewidth=cfg.grid_linewidth)
            ax.axvline(g - 0.5, color="#d6d6d6", linewidth=cfg.grid_linewidth)

        idx = np.arange(n)
        ax.set_xticks(idx)
        ax.set_xticklabels(cfg.labels, rotation=35, ha="right", fontsize=cfg.tick_fontsize)
        ax.set_yticks(idx)
        if i == 0:
            ax.set_yticklabels(cfg.labels, fontsize=cfg.tick_fontsize)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel(cfg.pretty_name, fontsize=8)

    if im is None:
        raise RuntimeError("Failed to render S07 enum confusion heatmap image.")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Row-normalized rate (log scale)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.subplots_adjust(left=0.11, right=0.95, top=0.98, bottom=0.25)
    save_figure(fig, cfg.figure_name, supplement=True)
    return fig

