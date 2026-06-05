from __future__ import annotations

# Release documentation:
# Builds publication figure module `figS01b_llm_vs_regex_concordance`.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs Supplement Figure S1.
# Usage: `python -m src.paper_figures.figS01b_llm_vs_regex_concordance` or `python scripts/build_paper_figures.py`.

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.db.connection import get_engine
from src.db.queries import (
    fetch_icd_codes_by_hadm_ids,
    fetch_notes_by_hadm_ids,
    fetch_primary_icd_by_hadm_ids,
)
from src.labeling_functions.base import LFInput, Vote
from src.labeling_functions.regex_lf import build_all_regex_lfs
from src.labeling_functions.section_parser import parse_sections
from src.paper_figures.config import (
    COLOR_SIGNAL_LLM,
    COLOR_SIGNAL_REGEX_LF,
    FULL_PAGE_WIDTH,
)
from src.paper_figures.plot_utils import apply_paper_style, save_figure, truncate_field_name


@dataclass(frozen=True)
class FigConfig:
    figure_name: str = "paper_fig_S01b_llm_vs_regex_concordance"
    width_inches: float = FULL_PAGE_WIDTH * 1.2
    height_inches: float = 4.2

    bar_width: float = 0.36
    color_p_llm_given_lf: str = COLOR_SIGNAL_LLM
    color_p_lf_given_llm: str = COLOR_SIGNAL_REGEX_LF

    tick_fontsize: int = 8
    annotate_fontsize: int = 7
    title: str = ""
    legend_y: float = 0.85


CFG = FigConfig()
REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw_responses"
SPLITS = REPO / "data" / "splits"
PATTERNS_DIR = REPO / "src" / "labeling_functions" / "patterns"
@dataclass(frozen=True)
class TargetKey:
    field: str
    value: str

    @property
    def as_text(self) -> str:
        return f"{self.field}::{self.value}"


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


def _llm_vote_for_target(features: dict[str, Any], target: TargetKey) -> Vote:
    if target.field == "admission_reason_tags":
        tags = set(features.get("admission_reason_tags", []))
        return Vote.POSITIVE if target.value in tags else Vote.ABSTAIN

    value = str(features.get(target.field))
    if target.value == "yes":
        if value == "yes":
            return Vote.POSITIVE
        if value == "no":
            return Vote.NEGATIVE
        return Vote.ABSTAIN

    if target.value == "no":
        if value == "no":
            return Vote.POSITIVE
        if value == "yes":
            return Vote.NEGATIVE
        return Vote.ABSTAIN

    return Vote.ABSTAIN


def _aggregate_non_llm_vote(votes: list[Vote]) -> Vote:
    if any(v == Vote.POSITIVE for v in votes):
        return Vote.POSITIVE
    if any(v == Vote.NEGATIVE for v in votes):
        return Vote.NEGATIVE
    return Vote.ABSTAIN


def _llm_consensus_yes(votes: list[Vote]) -> bool:
    return sum(1 for v in votes if v == Vote.POSITIVE) >= 2


@lru_cache(maxsize=1)
def _compute_regex_concordance_frame() -> pd.DataFrame:
    try:
        return _compute_regex_concordance_frame_db()
    except Exception as exc:  # pragma: no cover - explicit hard fail on DB issues
        raise RuntimeError(
            "Figure S01b requires live DB connectivity for regex-LF concordance; "
            "markdown fallback has been removed."
        ) from exc


def _compute_regex_concordance_frame_db() -> pd.DataFrame:
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

    regex_lfs = build_all_regex_lfs(PATTERNS_DIR)
    target_to_lfs: dict[TargetKey, list[Any]] = {}
    for lf in regex_lfs:
        tval = lf.target_value
        if tval is None:
            continue
        key = TargetKey(field=str(lf.target_field), value=str(tval))
        target_to_lfs.setdefault(key, []).append(lf)

    rows: list[dict[str, Any]] = []
    for target, lfs in sorted(target_to_lfs.items(), key=lambda kv: kv[0].as_text):
        lf_pos = 0
        llm_yes = 0
        both_yes = 0

        for hadm_id in common:
            llm_votes = [
                _llm_vote_for_target(parsed_by_variant[v][hadm_id], target)
                for v in ("A", "B", "C")
            ]
            llm_cons_yes = _llm_consensus_yes(llm_votes)

            primary = primary_icd_by_hadm.get(hadm_id)
            lf_input = LFInput(
                hadm_id=hadm_id,
                note_text=notes_by_hadm.get(hadm_id, ""),
                icd_codes=icd_by_hadm.get(hadm_id, []),
                primary_icd_code=primary[0] if primary else None,
                primary_icd_version=primary[1] if primary else None,
                sections=sections_by_hadm.get(hadm_id),
            )
            lf_votes = [lf(lf_input).vote for lf in lfs]
            lf_is_pos = _aggregate_non_llm_vote(lf_votes) == Vote.POSITIVE

            lf_pos += int(lf_is_pos)
            llm_yes += int(llm_cons_yes)
            both_yes += int(lf_is_pos and llm_cons_yes)

        p_llm_given_lf = (both_yes / lf_pos) if lf_pos > 0 else np.nan
        p_lf_given_llm = (both_yes / llm_yes) if llm_yes > 0 else np.nan
        rows.append(
            {
                "target": target.as_text,
                "lf_pos_n": lf_pos,
                "llm_yes_n": llm_yes,
                "both_yes_n": both_yes,
                "p_llm_yes_given_lf_pos": p_llm_given_lf,
                "p_lf_pos_given_llm_yes": p_lf_given_llm,
            }
        )

    frame = pd.DataFrame(rows)
    frame["target_label"] = (
        frame["target"].str.replace("::yes", "", regex=False).str.replace("_", " ", regex=False)
    )
    frame = frame.sort_values(["lf_pos_n", "target_label"], ascending=[False, True]).reset_index(
        drop=True
    )
    return frame


def render(df: pd.DataFrame, cfg: FigConfig = CFG) -> plt.Figure:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(cfg.width_inches, cfg.height_inches))

    x = np.arange(len(df), dtype=float)
    w = cfg.bar_width
    vals_a = df["p_llm_yes_given_lf_pos"].to_numpy(dtype=float)
    vals_b = df["p_lf_pos_given_llm_yes"].to_numpy(dtype=float)

    ax.bar(
        x - w / 2,
        vals_a,
        width=w,
        color=cfg.color_p_llm_given_lf,
        edgecolor="#000000",
        linewidth=0.6,
        label="P(LLM yes | regex LF pos)",
        zorder=3,
    )
    ax.bar(
        x + w / 2,
        vals_b,
        width=w,
        color=cfg.color_p_lf_given_llm,
        edgecolor="#000000",
        linewidth=0.6,
        label="P(regex LF pos | LLM yes)",
        zorder=3,
    )

    ax.set_ylim(0.0, 1.1)
    ax.axhline(1.0, color="#555555", linewidth=1.3, zorder=2)
    ax.set_ylabel("Empirical conditional rate")
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [truncate_field_name(lbl, 28) for lbl in df["target_label"].tolist()],
        rotation=40,
        ha="right",
        fontsize=cfg.tick_fontsize,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, cfg.legend_y), frameon=False)
    ax.set_title(cfg.title)

    for i, n in enumerate(df["lf_pos_n"].tolist()):
        ax.text(i, 1.01, f"n={int(n)}", ha="center", va="bottom", fontsize=cfg.annotate_fontsize)

    fig.tight_layout()
    return fig


def build() -> plt.Figure:
    df = _compute_regex_concordance_frame()
    fig = render(df)
    save_figure(fig, CFG.figure_name, supplement=True)
    return fig


if __name__ == "__main__":
    build()
