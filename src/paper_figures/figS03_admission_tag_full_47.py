from __future__ import annotations

# Release documentation:
# Builds publication figure module `figS03_admission_tag_full_47`.
#
# Reads: data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Writes: data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Backs Supplement Figure S3.
# Usage: `python -m src.paper_figures.figS03_admission_tag_full_47` or `python scripts/build_paper_figures.py`.

# ruff: noqa: E501

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.config import CMAP_AGREEMENT_HEATMAP, FULL_PAGE_WIDTH
from src.paper_figures.data_loaders import (
    get_admission_tag_vocabulary,
    load_methodology_5k_audit_extractions,
)
from src.paper_figures.plot_utils import apply_paper_style, save_figure


# =============================================================================
# CONFIGURATION - edit values here for cosmetic changes
# =============================================================================


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_S03_admission_tag_full_47"
    width_inches: float = FULL_PAGE_WIDTH * 3.0
    height_inches: float = FULL_PAGE_WIDTH * 1.15
    title: str = "Admission-Tag Confusion (Full 47 Tags, 5k-Audit Subset)"

    pair_order: tuple[tuple[str, str], tuple[str, str], tuple[str, str]] = (
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    )
    sort_tags_by_prevalence: bool = True

    cmap: str = CMAP_AGREEMENT_HEATMAP
    fixed_vmax: float | None = None
    quantile_vmax: float = 0.99
    min_vmax: float = 0.20
    max_vmax: float = 1.0

    show_every_nth_label: int = 1
    tick_label_size: float = 4.2


CFG = FigConfig()


# =============================================================================
# DATA PREP
# =============================================================================


def _safe_tag_set(value: Any, allowed: set[str]) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(x) for x in value if str(x) in allowed}
    return set()


def prepare_data() -> dict[str, Any]:
    """Load methodology 5k-audit variants and build pairwise 47x47 confusion matrices."""
    df = load_methodology_5k_audit_extractions().copy()
    vocab = get_admission_tag_vocabulary()
    allowed = set(vocab)

    required = {"hadm_id", "variant", "admission_reason_tags"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for tag confusion: {sorted(missing)}")

    by_variant: dict[str, dict[int, set[str]]] = {}
    for variant in ["A", "B", "C"]:
        sub = df[df["variant"] == variant]
        rows: dict[int, set[str]] = {}
        for _, row in sub.iterrows():
            hadm = int(row["hadm_id"])
            rows[hadm] = _safe_tag_set(row.get("admission_reason_tags"), allowed)
        by_variant[variant] = rows

    common_hadm = set(by_variant["A"]) & set(by_variant["B"]) & set(by_variant["C"])
    if not common_hadm:
        raise ValueError("No common hadm_id values across variants A/B/C.")

    tag_counts: dict[str, int] = {tag: 0 for tag in vocab}
    for variant in ["A", "B", "C"]:
        for hadm in common_hadm:
            for tag in by_variant[variant][hadm]:
                tag_counts[tag] += 1

    if CFG.sort_tags_by_prevalence:
        ordered_tags = sorted(vocab, key=lambda t: (-tag_counts.get(t, 0), t))
    else:
        ordered_tags = list(vocab)

    tag_idx = {tag: idx for idx, tag in enumerate(ordered_tags)}
    n_tags = len(ordered_tags)

    matrices: dict[str, np.ndarray] = {}
    diagonal_masses: dict[str, float] = {}

    for left, right in CFG.pair_order:
        confusion = np.zeros((n_tags, n_tags), dtype=np.int64)
        diag = np.zeros(n_tags, dtype=np.int64)

        for hadm in common_hadm:
            left_tags = by_variant[left][hadm]
            right_tags = by_variant[right][hadm]

            for tag in left_tags & right_tags:
                diag[tag_idx[tag]] += 1

            left_only = left_tags - right_tags
            right_only = right_tags - left_tags
            for left_tag in left_only:
                i = tag_idx[left_tag]
                for right_tag in right_only:
                    confusion[i, tag_idx[right_tag]] += 1

        row_offdiag = confusion.sum(axis=1).astype(np.float64)
        row_rates = np.divide(
            confusion.astype(np.float64),
            row_offdiag[:, None],
            out=np.zeros_like(confusion, dtype=np.float64),
            where=row_offdiag[:, None] > 0,
        )

        diag_counts = diag.astype(np.float64)
        denom = diag_counts + row_offdiag
        diag_rates = np.divide(
            diag_counts,
            denom,
            out=np.zeros_like(diag_counts, dtype=np.float64),
            where=denom > 0,
        )
        for i in range(n_tags):
            row_rates[i, i] = diag_rates[i]

        key = f"{left}-{right}"
        matrices[key] = row_rates

        total_diag = float(diag_counts.sum())
        total_conf = float(confusion.sum())
        diagonal_masses[key] = (total_diag / (total_diag + total_conf)) if (total_diag + total_conf) > 0 else 0.0

    all_nonzero = np.concatenate([mat[mat > 0.0] for mat in matrices.values()])
    if CFG.fixed_vmax is not None:
        vmax = CFG.fixed_vmax
    elif all_nonzero.size:
        vmax = float(np.quantile(all_nonzero, CFG.quantile_vmax))
        vmax = max(CFG.min_vmax, min(CFG.max_vmax, vmax))
    else:
        vmax = CFG.max_vmax

    return {
        "tags": ordered_tags,
        "matrices": matrices,
        "diagonal_masses": diagonal_masses,
        "vmax": vmax,
    }


# =============================================================================
# RENDERING
# =============================================================================


def render(data: dict[str, Any], cfg: FigConfig = CFG) -> plt.Figure:
    """Build the figure from prepared data."""
    apply_paper_style()

    tags: list[str] = data["tags"]
    matrices: dict[str, np.ndarray] = data["matrices"]
    diagonal_masses: dict[str, float] = data["diagonal_masses"]
    vmax: float = float(data["vmax"])

    fig = plt.figure(figsize=(cfg.width_inches, cfg.height_inches))
    # Use a dedicated colorbar column to avoid overlap with the third matrix.
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.06],
        wspace=0.16,
    )
    axes_arr = np.array([fig.add_subplot(gs[0, i]) for i in range(3)])
    cax = fig.add_subplot(gs[0, 3])

    im = None
    step = max(1, cfg.show_every_nth_label)
    idx_ticks = np.arange(0, len(tags), step)
    labels = [tags[i] for i in idx_ticks]

    for ax, (left, right) in zip(axes_arr, cfg.pair_order, strict=True):
        key = f"{left}-{right}"
        mat = matrices[key]
        im = ax.imshow(
            mat,
            cmap=cfg.cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )

        diag_mass = diagonal_masses[key] * 100.0
        ax.set_title(f"{left} vs {right}\nDiagonal mass: {diag_mass:.1f}%")
        ax.set_xlabel(f"Tag selected by {right} (and not by {left})")
        ax.set_ylabel(f"Tag selected by {left} (and not by {right})")

        ax.set_xticks(idx_ticks)
        ax.set_yticks(idx_ticks)
        ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=cfg.tick_label_size)
        ax.set_yticklabels(labels, fontsize=cfg.tick_label_size)

    if im is not None:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Row-normalized off-diagonal confusion rate / diagonal agreement rate")

    fig.suptitle(cfg.title, y=0.995)
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.18, top=0.92)
    return fig


# =============================================================================
# MAIN
# =============================================================================


def build() -> plt.Figure:
    """Build and save the figure. Called by orchestrator."""
    data = prepare_data()
    fig = render(data)
    save_figure(fig, CFG.figure_name, supplement=True)
    return fig


if __name__ == "__main__":
    build()
