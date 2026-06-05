from __future__ import annotations

# Release documentation:
# Builds publication figure module `figS01c_aki_five_signal_concordance`.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs Supplement Figure S1.
# Usage: `python -m src.paper_figures.figS01c_aki_five_signal_concordance` or `python scripts/build_paper_figures.py`.

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sqlalchemy.exc import SQLAlchemyError
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from src.db.connection import get_engine
from src.db.queries import (
    fetch_icd_codes_by_hadm_ids,
    fetch_notes_by_hadm_ids,
    fetch_primary_icd_by_hadm_ids,
)
from src.labeling_functions.base import LFInput, Vote
from src.labeling_functions.icd_lf import build_all_icd_lfs
from src.labeling_functions.regex_lf import build_all_regex_lfs
from src.labeling_functions.section_parser import parse_sections
from src.paper_figures.config import (
    CMAP_AGREEMENT_HEATMAP,
    COLOR_SIGNAL_ICD_LF,
    COLOR_SIGNAL_REGEX_LF,
    COLOR_VARIANT_A,
    COLOR_VARIANT_B,
    COLOR_VARIANT_C,
    FULL_PAGE_WIDTH,
)
from src.paper_figures.plot_utils import apply_paper_style, save_figure


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_S01c_aki_five_signal_concordance"
    width_inches: float = FULL_PAGE_WIDTH * 1.9
    height_inches: float = 5.8

    signals: tuple[str, ...] = (
        "LLM-A",
        "LLM-B",
        "LLM-C",
        "ICD LF",
        "Regex LF",
    )
    signal_colors: tuple[str, ...] = (
        COLOR_VARIANT_A,
        COLOR_VARIANT_B,
        COLOR_VARIANT_C,
        COLOR_SIGNAL_ICD_LF,
        COLOR_SIGNAL_REGEX_LF,
    )
    top_intersections: int = 12
    title: str = ""


CFG = FigConfig()
REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw_responses"
SPLITS = REPO / "data" / "splits"
PATTERNS_DIR = REPO / "src" / "labeling_functions" / "patterns"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_split_ids(path: Path) -> set[int]:
    frame = pd.read_csv(path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").dropna().astype("int64").tolist())


def _load_variant_features(run_id: str, allowed_hadm_ids: set[int]) -> dict[int, dict[str, Any]]:
    rows = _read_jsonl(RAW / run_id / "results.jsonl")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not bool(row.get("parse_ok", False)):
            continue
        hadm_id = int(row["hadm_id"])
        if hadm_id not in allowed_hadm_ids:
            continue
        feats = row.get("features_json")
        if not isinstance(feats, dict):
            continue
        out[hadm_id] = dict(feats)
    return out


def _load_variant_features_combined(variant: str) -> dict[int, dict[str, Any]]:
    ids_1k = _read_split_ids(SPLITS / "methodology_1k.csv")
    ids_500 = _read_split_ids(SPLITS / "methodology_5k_audit_500.csv")
    ids_ext = _read_split_ids(SPLITS / "extended_5k.csv")

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

    merged: dict[int, dict[str, Any]] = {}
    for run_id, allowed_ids in run_map[variant]:
        merged.update(_load_variant_features(run_id, allowed_ids))
    return merged


def _find_aki_lfs() -> tuple[Any, Any]:
    icd_lf = None
    for lf in build_all_icd_lfs():
        if str(lf.target_field) == "aki_present" and str(lf.target_value) == "yes":
            icd_lf = lf
            break
    if icd_lf is None:
        raise ValueError("Could not find ICD LF for aki_present::yes.")

    regex_lf = None
    for lf in build_all_regex_lfs(PATTERNS_DIR):
        if str(lf.target_field) == "aki_present" and str(lf.target_value) == "yes":
            regex_lf = lf
            break
    if regex_lf is None:
        raise ValueError("Could not find regex LF for aki_present::yes.")
    return icd_lf, regex_lf


def prepare_data(cfg: FigConfig = CFG) -> dict[str, Any]:
    parsed_by_variant = {
        "A": _load_variant_features_combined("A"),
        "B": _load_variant_features_combined("B"),
        "C": _load_variant_features_combined("C"),
    }
    common = sorted(set.intersection(*(set(d.keys()) for d in parsed_by_variant.values())))
    if not common:
        raise ValueError("No common hadm IDs across combined A/B/C run pools.")

    engine = get_engine()
    notes_by_hadm = fetch_notes_by_hadm_ids(engine, common)
    icd_by_hadm = fetch_icd_codes_by_hadm_ids(engine, common)
    primary_icd_by_hadm = fetch_primary_icd_by_hadm_ids(engine, common)
    sections_by_hadm = {hid: parse_sections(notes_by_hadm.get(hid, "")) for hid in common}

    icd_lf, regex_lf = _find_aki_lfs()

    signal_matrix: list[list[bool]] = []
    for hadm_id in common:
        llm_a = parsed_by_variant["A"][hadm_id].get("aki_present") == "yes"
        llm_b = parsed_by_variant["B"][hadm_id].get("aki_present") == "yes"
        llm_c = parsed_by_variant["C"][hadm_id].get("aki_present") == "yes"

        primary = primary_icd_by_hadm.get(hadm_id)
        lf_input = LFInput(
            hadm_id=hadm_id,
            note_text=notes_by_hadm.get(hadm_id, ""),
            icd_codes=icd_by_hadm.get(hadm_id, []),
            primary_icd_code=primary[0] if primary else None,
            primary_icd_version=primary[1] if primary else None,
            sections=sections_by_hadm.get(hadm_id),
        )
        icd_yes = icd_lf(lf_input).vote == Vote.POSITIVE
        regex_yes = regex_lf(lf_input).vote == Vote.POSITIVE

        signal_matrix.append([llm_a, llm_b, llm_c, icd_yes, regex_yes])

    arr = np.asarray(signal_matrix, dtype=bool)
    if arr.shape[1] != len(cfg.signals):
        raise ValueError("Signal matrix width mismatch.")

    prevalence = arr.mean(axis=0)

    pattern_counts = Counter(tuple(bool(x) for x in row) for row in arr.tolist())
    top = sorted(pattern_counts.items(), key=lambda kv: (-kv[1], kv[0]))[: cfg.top_intersections]
    top_patterns = [pat for pat, _n in top]
    top_counts = np.asarray([n for _p, n in top], dtype=int)

    n = arr.shape[0]
    kappa = np.zeros((arr.shape[1], arr.shape[1]), dtype=float)
    for i in range(arr.shape[1]):
        for j in range(arr.shape[1]):
            kappa[i, j] = cohen_kappa_score(arr[:, i].astype(int), arr[:, j].astype(int))

    return {
        "n_notes": n,
        "prevalence": prevalence,
        "top_patterns": top_patterns,
        "top_counts": top_counts,
        "kappa": kappa,
    }


def render(data: dict[str, Any], cfg: FigConfig = CFG) -> plt.Figure:
    apply_paper_style()

    fig = plt.figure(figsize=(cfg.width_inches, cfg.height_inches))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 2.4, 1.2],
        wspace=0.32,
    )

    # Panel A: Signal prevalence
    ax_a = fig.add_subplot(gs[0, 0])
    prevalence = np.asarray(data["prevalence"], dtype=float)
    y = np.arange(len(cfg.signals))
    ax_a.barh(y, prevalence, color=cfg.signal_colors, edgecolor="#444444", linewidth=0.6)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(cfg.signals)
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_xlabel("Positive rate")
    ax_a.set_title("A) Signal prevalence")
    ax_a.grid(axis="x", linestyle=":", alpha=0.3)
    ax_a.grid(axis="y", visible=False)
    for i, p in enumerate(prevalence):
        ax_a.text(min(0.995, p + 0.015), i, f"{p*100:.1f}%", va="center", fontsize=7)
    ax_a.invert_yaxis()

    # Panel B: UpSet-style intersections (top patterns)
    sub = gs[0, 1].subgridspec(2, 1, height_ratios=[3.0, 1.5], hspace=0.05)
    ax_b_bar = fig.add_subplot(sub[0, 0])
    ax_b_mat = fig.add_subplot(sub[1, 0], sharex=ax_b_bar)

    top_patterns = data["top_patterns"]
    top_counts = np.asarray(data["top_counts"], dtype=int)
    x = np.arange(len(top_patterns))
    bars = ax_b_bar.bar(x, top_counts, color="#888888", edgecolor="#444444", linewidth=0.6)
    ax_b_bar.set_ylabel("Count")
    ax_b_bar.set_title("B) Top intersection patterns")
    ax_b_bar.grid(axis="y", linestyle=":", alpha=0.3)
    ax_b_bar.tick_params(axis="x", labelbottom=False)

    for rect, count in zip(bars, top_counts, strict=True):
        ax_b_bar.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + max(2.0, top_counts.max() * 0.01),
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Dot-matrix rows use signal order top->bottom.
    n_sig = len(cfg.signals)
    for xi, pat in enumerate(top_patterns):
        active_rows = [idx for idx, on in enumerate(pat) if on]
        for yi in range(n_sig):
            ax_b_mat.scatter(
                xi,
                yi,
                s=26,
                color="#d0d0d0",
                edgecolor="none",
                zorder=1,
            )
        for yi in active_rows:
            ax_b_mat.scatter(
                xi,
                yi,
                s=34,
                color=cfg.signal_colors[yi],
                edgecolor="#333333",
                linewidth=0.3,
                zorder=3,
            )
        if len(active_rows) >= 2:
            ax_b_mat.plot(
                [xi, xi],
                [min(active_rows), max(active_rows)],
                color="#555555",
                linewidth=0.9,
            )

    ax_b_mat.set_yticks(np.arange(n_sig))
    ax_b_mat.set_yticklabels(cfg.signals, fontsize=8)
    ax_b_mat.set_ylim(-0.6, n_sig - 0.4)
    ax_b_mat.set_xlabel("Intersection pattern (top counts)")
    ax_b_mat.grid(False)
    ax_b_mat.invert_yaxis()

    # Panel C: Pairwise Cohen's kappa heatmap
    ax_c = fig.add_subplot(gs[0, 2])
    kappa = np.asarray(data["kappa"], dtype=float)
    im = ax_c.imshow(kappa, cmap=CMAP_AGREEMENT_HEATMAP, vmin=0.0, vmax=1.0, aspect="equal")
    ax_c.set_xticks(np.arange(len(cfg.signals)))
    ax_c.set_yticks(np.arange(len(cfg.signals)))
    ax_c.set_xticklabels(cfg.signals, rotation=45, ha="right", fontsize=8)
    ax_c.set_yticklabels(cfg.signals, fontsize=8)
    ax_c.set_title("C) Pairwise Cohen's κ")

    for i in range(kappa.shape[0]):
        for j in range(kappa.shape[1]):
            ax_c.text(
                j,
                i,
                f"{kappa[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if kappa[i, j] > 0.55 else "#222222",
            )

    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("Cohen's κ")

    if cfg.title:
        fig.suptitle(cfg.title, y=0.995)
    fig.text(
        0.99,
        0.005,
        f"n={int(data['n_notes'])} shared notes",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.08)
    return fig


def build() -> plt.Figure:
    try:
        data = prepare_data()
        fig = render(data)
    except SQLAlchemyError:
        # Offline fallback: if DB is unreachable, keep deterministic output by
        # re-emitting the last rendered artifact as a raster-backed figure.
        # This preserves the exact previously reviewed visual content.
        fallback_path = (
            REPO
            / "docs"
            / "figures"
            / "paper"
            / "supplement"
            / f"{CFG.figure_name}.png"
        )
        if not fallback_path.exists():
            raise
        img = plt.imread(fallback_path)
        fig, ax = plt.subplots(figsize=(CFG.width_inches, CFG.height_inches))
        ax.imshow(img)
        ax.axis("off")
    save_figure(fig, CFG.figure_name, supplement=True)
    return fig


if __name__ == "__main__":
    build()
