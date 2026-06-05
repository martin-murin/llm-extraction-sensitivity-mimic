from __future__ import annotations

# Release documentation:
# Provides shared helpers for claim-registry recomputation.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Supports paper claim recomputation and receipt verification.

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
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

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw_responses"
SPLITS = REPO / "data" / "splits"
PATTERNS_DIR = REPO / "src" / "labeling_functions" / "patterns"


@dataclass(frozen=True)
class PooledLFContext:
    parsed_by_variant: dict[str, dict[int, dict[str, Any]]]
    common_hadm_ids: list[int]
    notes_by_hadm: dict[int, str]
    icd_by_hadm: dict[int, list[tuple[str, int]]]
    primary_icd_by_hadm: dict[int, tuple[str, int]]
    sections_by_hadm: dict[int, dict[str, str]]


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


def _safe_kappa(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0
    val = float(cohen_kappa_score(a.astype(int), b.astype(int)))
    return 0.0 if np.isnan(val) else val


def _llm_vote_for_target(features: dict[str, Any], *, target_field: str, target_value: str) -> Vote:
    if target_field == "admission_reason_tags":
        tags = set(features.get("admission_reason_tags", []))
        return Vote.POSITIVE if target_value in tags else Vote.ABSTAIN

    value = str(features.get(target_field))
    if target_value == "yes":
        if value == "yes":
            return Vote.POSITIVE
        if value == "no":
            return Vote.NEGATIVE
        return Vote.ABSTAIN
    if target_value == "no":
        if value == "no":
            return Vote.POSITIVE
        if value == "yes":
            return Vote.NEGATIVE
        return Vote.ABSTAIN
    return Vote.ABSTAIN


def _llm_consensus_yes(votes: list[Vote]) -> bool:
    return sum(1 for v in votes if v == Vote.POSITIVE) >= 2


def _aggregate_non_llm_vote(votes: list[Vote]) -> Vote:
    if any(v == Vote.POSITIVE for v in votes):
        return Vote.POSITIVE
    if any(v == Vote.NEGATIVE for v in votes):
        return Vote.NEGATIVE
    return Vote.ABSTAIN


def _target_lfs_icd() -> dict[tuple[str, str], list[Any]]:
    out: dict[tuple[str, str], list[Any]] = {}
    for lf in build_all_icd_lfs():
        tval = lf.target_value
        if tval is None:
            continue
        key = (str(lf.target_field), str(tval))
        out.setdefault(key, []).append(lf)
    return out


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


def build_pooled_context() -> PooledLFContext:
    parsed_by_variant = {
        "A": _load_variant_features_combined("A"),
        "B": _load_variant_features_combined("B"),
        "C": _load_variant_features_combined("C"),
    }
    common = sorted(set.intersection(*(set(d.keys()) for d in parsed_by_variant.values())))
    if not common:
        raise ValueError("No shared hadm_id intersection across pooled A/B/C runs.")

    engine = get_engine()
    notes_by_hadm = fetch_notes_by_hadm_ids(engine, common)
    icd_by_hadm = fetch_icd_codes_by_hadm_ids(engine, common)
    primary_icd_by_hadm = fetch_primary_icd_by_hadm_ids(engine, common)
    sections_by_hadm = {hid: parse_sections(notes_by_hadm.get(hid, "")) for hid in common}

    return PooledLFContext(
        parsed_by_variant=parsed_by_variant,
        common_hadm_ids=common,
        notes_by_hadm=notes_by_hadm,
        icd_by_hadm=icd_by_hadm,
        primary_icd_by_hadm=primary_icd_by_hadm,
        sections_by_hadm=sections_by_hadm,
    )


def compute_aki_claim_metrics(ctx: PooledLFContext) -> dict[str, float | int]:
    icd_lf, regex_lf = _find_aki_lfs()
    signal_rows: list[list[bool]] = []

    for hadm_id in ctx.common_hadm_ids:
        llm_a = ctx.parsed_by_variant["A"][hadm_id].get("aki_present") == "yes"
        llm_b = ctx.parsed_by_variant["B"][hadm_id].get("aki_present") == "yes"
        llm_c = ctx.parsed_by_variant["C"][hadm_id].get("aki_present") == "yes"
        primary = ctx.primary_icd_by_hadm.get(hadm_id)
        lf_input = LFInput(
            hadm_id=hadm_id,
            note_text=ctx.notes_by_hadm.get(hadm_id, ""),
            icd_codes=ctx.icd_by_hadm.get(hadm_id, []),
            primary_icd_code=primary[0] if primary else None,
            primary_icd_version=primary[1] if primary else None,
            sections=ctx.sections_by_hadm.get(hadm_id),
        )
        icd_yes = icd_lf(lf_input).vote == Vote.POSITIVE
        regex_yes = regex_lf(lf_input).vote == Vote.POSITIVE
        signal_rows.append([llm_a, llm_b, llm_c, icd_yes, regex_yes])

    arr = np.asarray(signal_rows, dtype=bool)
    llm_a = arr[:, 0]
    llm_b = arr[:, 1]
    llm_c = arr[:, 2]
    icd = arr[:, 3]
    regex = arr[:, 4]

    llm_cons_any = llm_a | llm_b | llm_c
    all_llm_positive = llm_a & llm_b & llm_c

    return {
        "aki_llm_a_prevalence_pct": float(llm_a.mean() * 100.0),
        "aki_llm_c_prevalence_pct": float(llm_c.mean() * 100.0),
        "aki_icd_lf_prevalence_pct": float(icd.mean() * 100.0),
        "aki_regex_lf_prevalence_pct": float(regex.mean() * 100.0),
        "aki_llm_vs_icd_kappa_a": _safe_kappa(llm_a, icd),
        "aki_llm_vs_icd_kappa_b": _safe_kappa(llm_b, icd),
        "aki_icd_only_no_llm_count": int(np.sum(icd & (~llm_cons_any))),
        "aki_all_signals_negative_count": int(
            np.sum((~llm_a) & (~llm_b) & (~llm_c) & (~icd) & (~regex))
        ),
        "aki_all_llm_positive_no_icd_count": int(np.sum(all_llm_positive & (~icd))),
        "cross_variant_pooled_n": int(arr.shape[0]),
    }


def compute_icd_target_rates(
    ctx: PooledLFContext,
    targets: dict[str, tuple[str, str]],
) -> dict[str, float]:
    icd_target_lfs = _target_lfs_icd()
    out: dict[str, float] = {}

    for short_name, (field, value) in targets.items():
        key = (field, value)
        lfs = icd_target_lfs.get(key, [])
        if not lfs:
            raise ValueError(f"No ICD LFs found for target {field}::{value}.")

        lf_pos = 0
        llm_yes = 0
        both_yes = 0

        for hadm_id in ctx.common_hadm_ids:
            llm_votes = [
                _llm_vote_for_target(
                    ctx.parsed_by_variant[variant][hadm_id],
                    target_field=field,
                    target_value=value,
                )
                for variant in ("A", "B", "C")
            ]
            llm_cons_yes = _llm_consensus_yes(llm_votes)

            primary = ctx.primary_icd_by_hadm.get(hadm_id)
            lf_input = LFInput(
                hadm_id=hadm_id,
                note_text=ctx.notes_by_hadm.get(hadm_id, ""),
                icd_codes=ctx.icd_by_hadm.get(hadm_id, []),
                primary_icd_code=primary[0] if primary else None,
                primary_icd_version=primary[1] if primary else None,
                sections=ctx.sections_by_hadm.get(hadm_id),
            )
            lf_votes = [lf(lf_input).vote for lf in lfs]
            lf_is_pos = _aggregate_non_llm_vote(lf_votes) == Vote.POSITIVE

            lf_pos += int(lf_is_pos)
            llm_yes += int(llm_cons_yes)
            both_yes += int(lf_is_pos and llm_cons_yes)

        out[f"{short_name}_p_llm_yes_given_icd"] = float(both_yes / lf_pos) if lf_pos > 0 else 0.0
        out[f"{short_name}_p_icd_given_llm_yes"] = float(both_yes / llm_yes) if llm_yes > 0 else 0.0

    return out
