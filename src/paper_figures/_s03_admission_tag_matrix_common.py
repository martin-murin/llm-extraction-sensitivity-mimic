from __future__ import annotations

# Release documentation:
# Provides shared helpers/configuration for publication figure modules.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Supports publication figure generation.

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.paper_figures import config
from src.paper_figures.data_loaders import get_admission_tag_vocabulary
from src.paper_figures.plot_utils import apply_paper_style, save_figure

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_RESPONSES_DIR = REPO_ROOT / "data" / "raw_responses"
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def _read_split_ids(path: Path) -> set[int]:
    frame = pd.read_csv(path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").dropna().astype("int64").tolist())


def _read_parse_ok_dominant(
    run_id: str,
    dominant_field: str,
    allowed_hadm_ids: set[int] | None = None,
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
            if allowed_hadm_ids is not None and hadm_id not in allowed_hadm_ids:
                continue
            feats = payload.get("features_json")
            if not isinstance(feats, dict):
                continue
            out[hadm_id] = str(feats.get(dominant_field, ""))
    return out


def load_cross_variant_common(dominant_field: str = "dominant_admission_reason") -> pd.DataFrame:
    """Load A/B/C small-model pooled dominant labels on methodology_1k+500+extended_5k."""
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

    by_variant: dict[str, dict[int, str]] = {}
    for variant, parts in run_map.items():
        merged: dict[int, str] = {}
        for run_id, allowed_ids in parts:
            merged.update(_read_parse_ok_dominant(run_id, dominant_field, allowed_ids))
        by_variant[variant] = merged

    common_ids = sorted(set.intersection(*(set(by_variant[v].keys()) for v in ("A", "B", "C"))))
    if not common_ids:
        raise RuntimeError("No shared A/B/C hadm_id rows for cross-variant S03.")

    return pd.DataFrame(
        {
            "hadm_id": common_ids,
            "A": [by_variant["A"][hid] for hid in common_ids],
            "B": [by_variant["B"][hid] for hid in common_ids],
            "C": [by_variant["C"][hid] for hid in common_ids],
        }
    )


def load_model_size_common_for_variant(
    variant: str,
    dominant_field: str = "dominant_admission_reason",
) -> pd.DataFrame:
    """Load same-variant small/full dominant labels on paired methodology_1500."""
    variant_u = variant.upper()
    if variant_u not in {"A", "B", "C"}:
        raise ValueError(f"Unsupported variant: {variant}")

    small_map = {
        "A": ("methodology_1k_a", "methodology_5k_a_subset500"),
        "B": ("methodology_1k_b", "methodology_5k_audit_b"),
        "C": ("methodology_1k_c", "methodology_5k_audit_c"),
    }
    full_map = {
        "A": ("paired_gold_methodology_1k_a", "paired_gold_methodology_5k_audit_a"),
        "B": ("paired_gold_methodology_1k_b", "paired_gold_methodology_5k_audit_b"),
        "C": ("paired_gold_methodology_1k_c", "paired_gold_methodology_5k_audit_c"),
    }
    ids_1k = _read_split_ids(SPLITS_DIR / "methodology_1k.csv")
    ids_500 = _read_split_ids(SPLITS_DIR / "methodology_5k_audit_500.csv")

    small_1k, small_500 = small_map[variant_u]
    full_1k, full_500 = full_map[variant_u]

    small: dict[int, str] = {}
    small.update(_read_parse_ok_dominant(small_1k, dominant_field, ids_1k))
    small.update(_read_parse_ok_dominant(small_500, dominant_field, ids_500))

    full: dict[int, str] = {}
    full.update(_read_parse_ok_dominant(full_1k, dominant_field, ids_1k))
    full.update(_read_parse_ok_dominant(full_500, dominant_field, ids_500))

    common_ids = sorted(set(small.keys()) & set(full.keys()))
    if not common_ids:
        raise RuntimeError(f"No shared hadm_id rows for model-size S03 variant {variant_u}.")

    return pd.DataFrame(
        {
            "hadm_id": common_ids,
            "small": [small[hid] for hid in common_ids],
            "full": [full[hid] for hid in common_ids],
        }
    )


def order_tags_by_support(df_pairs: pd.DataFrame, left_col: str, right_col: str) -> list[str]:
    """Order 47 tags by pooled support in the two compared columns."""
    tags = get_admission_tag_vocabulary()
    counts = (
        pd.concat([df_pairs[left_col].astype(str), df_pairs[right_col].astype(str)], axis=0)
        .value_counts()
        .to_dict()
    )
    return sorted(tags, key=lambda t: (-int(counts.get(t, 0)), t))


def compute_confusion(
    df_pairs: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    tags: list[str],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    tag_index = {tag: idx for idx, tag in enumerate(tags)}
    left_values = df_pairs[left_col].astype(str)
    right_values = df_pairs[right_col].astype(str)
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
    return counts, rates, diag_mass, total


def render_single_matrix(
    *,
    figure_name: str,
    tags: list[str],
    rates: np.ndarray,
    tick_fontsize: float = 4.8,
    width_inches: float = 8.0,
    height_inches: float = 7.4,
) -> plt.Figure:
    """Render one 47x47 row-normalized confusion heatmap with a dedicated colorbar."""
    apply_paper_style()
    fig = plt.figure(figsize=(width_inches, height_inches))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.04], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    shown = rates.copy().astype(np.float64)
    # Keep weak off-diagonal signal visible with log scaling.
    min_positive = shown[shown > 0]
    vmin = float(np.min(min_positive)) if min_positive.size else 1e-3
    vmin = max(vmin, 1e-3)
    norm = colors.LogNorm(vmin=vmin, vmax=1.0)
    cmap = plt.get_cmap(config.CMAP_AGREEMENT_HEATMAP).copy()
    im = ax.imshow(shown, cmap=cmap, norm=norm, interpolation="none", aspect="equal")

    n = len(tags)
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="#dddddd", linewidth=0.18)
        ax.axvline(i - 0.5, color="#dddddd", linewidth=0.18)

    idx = np.arange(n)
    ax.set_xticks(idx)
    ax.set_yticks(idx)
    ax.set_xticklabels(tags, rotation=58, ha="right", rotation_mode="anchor", fontsize=tick_fontsize)
    ax.set_yticklabels(tags, fontsize=tick_fontsize)
    ax.set_xlabel("")
    ax.set_ylabel("")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Row-normalized rate (log scale)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.subplots_adjust(left=0.20, right=0.94, bottom=0.26, top=0.98)
    save_figure(fig, figure_name, supplement=True)
    return fig
