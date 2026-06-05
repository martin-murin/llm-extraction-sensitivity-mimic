from __future__ import annotations

# Release documentation:
# Computes claim-registry values for admission tags.
#
# Reads: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Writes: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Backs paper claim registry entries for admission tags.

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import spearmanr

from paper.claims.scripts._common import (
    claim_entry,
    require_input_files,
)
from paper.claims.scripts._receipt import build_receipt, merge_into_claims_json, now_utc_iso
from src.paper_figures.data_loaders import load_paired_full_extractions
from src.schema.vocabulary import ADMISSION_REASON_TAGS

CLAIMS_PATH = Path(__file__).resolve().parent.parent / "claims.json"
SCRIPT_PATH = Path(__file__).resolve()
REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw_responses"
SPLITS = REPO / "data" / "splits"

INPUT_FILES = [
    "data/raw_responses/methodology_1k_a/results.jsonl",
    "data/raw_responses/methodology_1k_b/results.jsonl",
    "data/raw_responses/methodology_1k_c/results.jsonl",
    "data/raw_responses/methodology_5k_a_subset500/results.jsonl",
    "data/raw_responses/methodology_5k_audit_b/results.jsonl",
    "data/raw_responses/methodology_5k_audit_c/results.jsonl",
    "data/raw_responses/production_v1/results.jsonl",
    "data/raw_responses/extended_5k_b/results.jsonl",
    "data/raw_responses/extended_5k_c/results.jsonl",
    "data/splits/methodology_1k.csv",
    "data/splits/methodology_5k_audit_500.csv",
    "data/splits/extended_5k.csv",
    "data/raw_responses/paired_gold_methodology_1k_a/results.jsonl",
    "data/raw_responses/paired_gold_methodology_1k_b/results.jsonl",
    "data/raw_responses/paired_gold_methodology_1k_c/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_a/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_b/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_c/results.jsonl",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_split_ids(path: Path) -> set[int]:
    df = pd.read_csv(path)
    return set(pd.to_numeric(df["hadm_id"], errors="coerce").dropna().astype("int64").tolist())


def _load_variant(run_id: str, allowed: set[int]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(RAW / run_id / "results.jsonl"):
        if not bool(row.get("parse_ok", False)):
            continue
        hid = int(row["hadm_id"])
        if hid not in allowed:
            continue
        feats = row.get("features_json")
        if isinstance(feats, dict):
            out[hid] = feats
    return out


def _combined_variant_features() -> tuple[dict[str, dict[int, dict[str, Any]]], list[int]]:
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
    by_var: dict[str, dict[int, dict[str, Any]]] = {}
    for v, parts in run_map.items():
        merged: dict[int, dict[str, Any]] = {}
        for run, allowed in parts:
            merged.update(_load_variant(run, allowed))
        by_var[v] = merged
    common = sorted(set.intersection(*(set(by_var[v].keys()) for v in ("A", "B", "C"))))
    return by_var, common


def _normalize_tags(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    vocab = set(ADMISSION_REASON_TAGS)
    for t in raw:
        if not isinstance(t, str):
            continue
        x = t.strip()
        if not x or x not in vocab or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _pair_tag_jaccard_pct(
    by_var: dict[str, dict[int, dict[str, Any]]],
    common_hadm_ids: list[int],
    left_variant: str,
    right_variant: str,
) -> float:
    per_note: list[float] = []
    for hid in common_hadm_ids:
        lt = set(_normalize_tags(by_var[left_variant][hid].get("admission_reason_tags", [])))
        rt = set(_normalize_tags(by_var[right_variant][hid].get("admission_reason_tags", [])))
        union = lt | rt
        if not union:
            per_note.append(1.0)
        else:
            per_note.append(float(len(lt & rt) / len(union)))
    return float(np.nanmean(np.asarray(per_note, dtype=float)) * 100.0) if per_note else float("nan")


def _paired_full_agreements_pct() -> tuple[float, dict[str, float], float, dict[str, float]]:
    """Compute paired full-model agreements across A-B/A-C/B-C.

    Returns:
        (
            mean_admission_tag_jaccard_pct,
            {"AB": tag_jaccard_pct, "AC": tag_jaccard_pct, "BC": tag_jaccard_pct},
            mean_dominant_diagonal_mass_pct,
            {"AB": dominant_pct, "AC": dominant_pct, "BC": dominant_pct},
        )
    """
    frame = load_paired_full_extractions("full").copy()
    if frame.empty:
        raise ValueError("Paired full-model extraction frame is empty.")

    by_var = {
        v: frame[frame["variant"] == v].drop_duplicates(subset=["hadm_id"], keep="last").copy()
        for v in ("A", "B", "C")
    }
    common = sorted(set.intersection(*(set(by_var[v]["hadm_id"]) for v in ("A", "B", "C"))))
    if not common:
        raise ValueError("No shared hadm_id intersection across A/B/C in paired full-model data.")

    for v in ("A", "B", "C"):
        by_var[v] = by_var[v][by_var[v]["hadm_id"].isin(common)].set_index("hadm_id")

    def _tag_jaccard(vl: str, vr: str) -> float:
        per_note: list[float] = []
        left = by_var[vl]
        right = by_var[vr]
        for hid in common:
            lt = set(_normalize_tags(left.at[hid, "admission_reason_tags"]))
            rt = set(_normalize_tags(right.at[hid, "admission_reason_tags"]))
            union = lt | rt
            if not union:
                per_note.append(1.0)
            else:
                per_note.append(float(len(lt & rt) / len(union)))
        return float(np.nanmean(np.asarray(per_note, dtype=float))) if per_note else float("nan")

    def _dominant_diag_mass(vl: str, vr: str) -> float:
        left = by_var[vl]
        right = by_var[vr]
        agree = 0
        valid = 0
        for hid in common:
            lv = left.at[hid, "dominant_admission_reason"]
            rv = right.at[hid, "dominant_admission_reason"]
            if pd.isna(lv) or pd.isna(rv):
                continue
            valid += 1
            if str(lv) == str(rv):
                agree += 1
        return (agree / valid) if valid > 0 else float("nan")

    tag_pair = {
        "AB": _tag_jaccard("A", "B"),
        "AC": _tag_jaccard("A", "C"),
        "BC": _tag_jaccard("B", "C"),
    }
    tag_vals = np.asarray([tag_pair["AB"], tag_pair["AC"], tag_pair["BC"]], dtype=float)
    dom_pair = {
        "AB": _dominant_diag_mass("A", "B"),
        "AC": _dominant_diag_mass("A", "C"),
        "BC": _dominant_diag_mass("B", "C"),
    }
    dom_vals = np.asarray([dom_pair["AB"], dom_pair["AC"], dom_pair["BC"]], dtype=float)
    return (
        float(np.nanmean(tag_vals) * 100.0),
        {k: float(v * 100.0) for k, v in tag_pair.items()},
        float(np.nanmean(dom_vals) * 100.0),
        {k: float(v * 100.0) for k, v in dom_pair.items()},
    )


def _model_size_dominant_within_variant_pct() -> dict[str, float]:
    """Compute same-variant dominant-tag agreement across model sizes on paired 1500 notes.

    Returns:
        {"AA": pct, "BB": pct, "CC": pct}
    """
    small = load_paired_full_extractions("small").copy()
    full = load_paired_full_extractions("full").copy()
    if small.empty or full.empty:
        raise ValueError("Paired small/full extraction frames are empty.")

    out: dict[str, float] = {}
    for variant in ("A", "B", "C"):
        sm = (
            small[small["variant"] == variant]
            .drop_duplicates(subset=["hadm_id"], keep="last")
            .set_index("hadm_id")
        )
        fu = (
            full[full["variant"] == variant]
            .drop_duplicates(subset=["hadm_id"], keep="last")
            .set_index("hadm_id")
        )
        common = sorted(set(sm.index) & set(fu.index))
        if not common:
            raise ValueError(f"No shared hadm_id for model-size dominant comparison variant {variant}.")
        agree = 0
        valid = 0
        for hid in common:
            sv = sm.at[hid, "dominant_admission_reason"]
            fv = fu.at[hid, "dominant_admission_reason"]
            if pd.isna(sv) or pd.isna(fv):
                continue
            valid += 1
            if str(sv) == str(fv):
                agree += 1
        out[f"{variant}{variant}"] = (100.0 * agree / valid) if valid > 0 else float("nan")
    return out


def _model_size_admission_tag_jaccard_within_variant_pct() -> dict[str, float]:
    """Compute same-variant admission-tag Jaccard across model sizes on paired 1500 notes.

    Returns:
        {"AA": pct, "BB": pct, "CC": pct}
    """
    small = load_paired_full_extractions("small").copy()
    full = load_paired_full_extractions("full").copy()
    if small.empty or full.empty:
        raise ValueError("Paired small/full extraction frames are empty.")

    out: dict[str, float] = {}
    for variant in ("A", "B", "C"):
        sm = (
            small[small["variant"] == variant]
            .drop_duplicates(subset=["hadm_id"], keep="last")
            .set_index("hadm_id")
        )
        fu = (
            full[full["variant"] == variant]
            .drop_duplicates(subset=["hadm_id"], keep="last")
            .set_index("hadm_id")
        )
        common = sorted(set(sm.index) & set(fu.index))
        if not common:
            raise ValueError(
                f"No shared hadm_id for model-size admission-tag comparison variant {variant}."
            )
        per_note: list[float] = []
        for hid in common:
            st = set(_normalize_tags(sm.at[hid, "admission_reason_tags"]))
            ft = set(_normalize_tags(fu.at[hid, "admission_reason_tags"]))
            union = st | ft
            if not union:
                per_note.append(1.0)
            else:
                per_note.append(float(len(st & ft) / len(union)))
        out[f"{variant}{variant}"] = (
            float(np.nanmean(np.asarray(per_note, dtype=float)) * 100.0)
            if per_note
            else float("nan")
        )
    return out


def _paired_residual_mass_and_confusion_metrics() -> dict[str, float]:
    """Compute paired-1500 residual-mass and multi-label confusion-rate metrics.

    All metrics use the strict shared hadm_id intersection across:
    small/full x variants A/B/C on the paired methodology_1500 sample.

    Mass metrics are A/B/C-averaged shares of total admission-tag firing mass.
    Confusion-rate metrics are normalized multi-label presence confusion rates:
      confusion_rate(tag) = 100 * mean_pair xor_mass(tag) / prevalence_any_variant(tag)
    where xor_mass(tag) is the fraction of notes where exactly one side of a
    pair contains tag membership.
    """
    small = load_paired_full_extractions("small").copy()
    full = load_paired_full_extractions("full").copy()
    if small.empty or full.empty:
        raise ValueError("Paired small/full extraction frames are empty.")

    by_model: dict[str, dict[str, pd.DataFrame]] = {"small": {}, "full": {}}
    for model_name, frame in (("small", small), ("full", full)):
        for variant in ("A", "B", "C"):
            by_model[model_name][variant] = (
                frame[frame["variant"] == variant]
                .drop_duplicates(subset=["hadm_id"], keep="last")
                .set_index("hadm_id")
            )

    common = sorted(
        set(by_model["small"]["A"].index)
        & set(by_model["small"]["B"].index)
        & set(by_model["small"]["C"].index)
        & set(by_model["full"]["A"].index)
        & set(by_model["full"]["B"].index)
        & set(by_model["full"]["C"].index)
    )
    if not common:
        raise ValueError("No shared hadm_id for paired residual-mass/confusion metrics.")

    residual_family = {
        "cardiac_other",
        "respiratory_failure_other",
        "gi_other",
        "gu_other",
        "infection_other",
        "neuro_other",
        "endocrine_other",
        "trauma_other",
        "symptom_workup_other",
        "other",
    }
    mass_tags = [
        "elective_procedure_non_oncology",
        "heme_onc_complication",
        "trauma_fracture",
    ]
    confusion_tags = [
        "neuro_other",
        "gi_other",
        "gu_other",
        "infection_other",
        "respiratory_failure_other",
        "cardiac_other",
        "trauma_other",
    ]

    def _avg_mass_share_pct(model_name: str, tag_spec: str | set[str]) -> float:
        shares: list[float] = []
        for variant in ("A", "B", "C"):
            total = 0
            hit = 0
            sub = by_model[model_name][variant]
            for hid in common:
                tags = _normalize_tags(sub.at[hid, "admission_reason_tags"])
                total += len(tags)
                if isinstance(tag_spec, set):
                    hit += sum(1 for t in tags if t in tag_spec)
                else:
                    hit += sum(1 for t in tags if t == tag_spec)
            shares.append((100.0 * hit / total) if total > 0 else float("nan"))
        return float(np.nanmean(np.asarray(shares, dtype=float)))

    def _normalized_confusion_rate_pct(model_name: str, tag: str) -> float:
        dom = by_model[model_name]
        prevalence_any = 0
        pair_xor_counts = {"AB": 0, "AC": 0, "BC": 0}
        n_notes = len(common)
        for hid in common:
            has_a = tag in set(_normalize_tags(dom["A"].at[hid, "admission_reason_tags"]))
            has_b = tag in set(_normalize_tags(dom["B"].at[hid, "admission_reason_tags"]))
            has_c = tag in set(_normalize_tags(dom["C"].at[hid, "admission_reason_tags"]))
            if has_a or has_b or has_c:
                prevalence_any += 1
            if has_a != has_b:
                pair_xor_counts["AB"] += 1
            if has_a != has_c:
                pair_xor_counts["AC"] += 1
            if has_b != has_c:
                pair_xor_counts["BC"] += 1
        prevalence = (prevalence_any / n_notes) if n_notes > 0 else 0.0
        xor_mass = float(
            np.mean(
                np.asarray(
                    [
                        pair_xor_counts["AB"] / n_notes,
                        pair_xor_counts["AC"] / n_notes,
                        pair_xor_counts["BC"] / n_notes,
                    ],
                    dtype=float,
                )
            )
        )
        if prevalence <= 0:
            return float("nan")
        return float(100.0 * xor_mass / prevalence)

    out: dict[str, float] = {
        "residual_mass_small_pct": _avg_mass_share_pct("small", residual_family),
        "residual_mass_full_pct": _avg_mass_share_pct("full", residual_family),
        "other_tag_mass_small_pct": _avg_mass_share_pct("small", "other"),
        "other_tag_mass_full_pct": _avg_mass_share_pct("full", "other"),
    }
    for tag in mass_tags:
        out[f"{tag}_mass_small_pct"] = _avg_mass_share_pct("small", tag)
        out[f"{tag}_mass_full_pct"] = _avg_mass_share_pct("full", tag)
    for tag in confusion_tags:
        out[f"{tag}_confusion_rate_small_pct"] = _normalized_confusion_rate_pct("small", tag)
        out[f"{tag}_confusion_rate_full_pct"] = _normalized_confusion_rate_pct("full", tag)
    return out


def compute_all() -> dict:
    require_input_files(INPUT_FILES)
    by_var, common = _combined_variant_features()
    n = len(common)
    if n == 0:
        raise ValueError("No shared A/B/C hadm_id intersection for admission tag claims.")

    rows: list[dict[str, Any]] = []
    for v in ("A", "B", "C"):
        note_tag_counts: list[int] = []
        tag_note_hits: dict[str, int] = {t: 0 for t in ADMISSION_REASON_TAGS}
        dominant_vals: list[str] = []
        total_tag_firings = 0
        for hid in common:
            feats = by_var[v][hid]
            tags = _normalize_tags(feats.get("admission_reason_tags", []))
            note_tag_counts.append(len(tags))
            total_tag_firings += len(tags)
            for t in tags:
                tag_note_hits[t] += 1
            dominant_vals.append(str(feats.get("dominant_admission_reason", "")))
        for t in ADMISSION_REASON_TAGS:
            rows.append(
                {
                    "variant": v,
                    "tag": t,
                    "prevalence_pct": (100.0 * tag_note_hits[t] / n),
                    "share_pct": (100.0 * tag_note_hits[t] / total_tag_firings)
                    if total_tag_firings
                    else 0.0,
                }
            )
        rows.append(
            {
                "variant": v,
                "tag": "__mean_tags_per_note__",
                "prevalence_pct": float(np.mean(note_tag_counts)),
                "share_pct": float(np.percentile(note_tag_counts, 75)),
                "max_tags": int(max(note_tag_counts) if note_tag_counts else 0),
            }
        )

    df = pd.DataFrame(rows)

    # Dominant diagonal mass from 6500 pooled set.
    dominant = pd.DataFrame(
        {
            "hadm_id": common,
            "A": [str(by_var["A"][hid].get("dominant_admission_reason", "")) for hid in common],
            "B": [str(by_var["B"][hid].get("dominant_admission_reason", "")) for hid in common],
            "C": [str(by_var["C"][hid].get("dominant_admission_reason", "")) for hid in common],
        }
    )
    ab_dom = float((dominant["A"] == dominant["B"]).mean() * 100.0)
    ac_dom = float((dominant["A"] == dominant["C"]).mean() * 100.0)
    bc_dom = float((dominant["B"] == dominant["C"]).mean() * 100.0)

    # Per-tag prevalence-vs-confusion relationship on pooled cross-variant n=6500 sample.
    # prevalence(T): share of notes where any variant assigns dominant tag T.
    # confusion_mass(T): mean over pairs (A-B, A-C, B-C) of fraction where one side is T
    # and the other side is not T.
    tag_stats_rows: list[dict[str, float]] = []
    for tag in ADMISSION_REASON_TAGS:
        prevalence = (
            ((dominant["A"] == tag) | (dominant["B"] == tag) | (dominant["C"] == tag)).mean()
        )
        pair_masses = []
        for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
            pair_mass = (
                (((dominant[left] == tag) & (dominant[right] != tag))
                | ((dominant[right] == tag) & (dominant[left] != tag))).mean()
            )
            pair_masses.append(float(pair_mass))
        confusion_mass = float(np.mean(pair_masses))
        tag_stats_rows.append(
            {
                "prevalence": float(prevalence),
                "confusion_mass": confusion_mass,
            }
        )
    tag_stats = pd.DataFrame(tag_stats_rows)
    spearman_rho, spearman_pvalue = spearmanr(
        tag_stats["prevalence"],
        tag_stats["confusion_mass"],
    )
    (
        paired_full_tag_jaccard_pct,
        paired_full_tag_pairs_pct,
        paired_full_dom_diag_mass_pct,
        paired_full_dom_pairs_pct,
    ) = _paired_full_agreements_pct()
    model_size_within_variant_pct = _model_size_dominant_within_variant_pct()
    model_size_tag_jaccard_pct = _model_size_admission_tag_jaccard_within_variant_pct()
    residual_metrics = _paired_residual_mass_and_confusion_metrics()

    # Multi-label admission-tag Jaccard (pair-level) from pooled raw run outputs.
    tag_jaccard_small = {
        "A-B": _pair_tag_jaccard_pct(by_var, common, "A", "B"),
        "A-C": _pair_tag_jaccard_pct(by_var, common, "A", "C"),
        "B-C": _pair_tag_jaccard_pct(by_var, common, "B", "C"),
    }

    def _val(variant: str, tag: str, col: str = "prevalence_pct") -> float:
        row = df[(df["variant"] == variant) & (df["tag"] == tag)]
        if row.empty:
            return 0.0
        return float(row.iloc[0][col])

    timestamp = now_utc_iso()
    receipt = build_receipt(SCRIPT_PATH, "compute_all", INPUT_FILES)
    claims = {
        "mean_tags_per_note_a": claim_entry(
            value=_val("A", "__mean_tags_per_note__"),
            format_default=".2f",
            description="Mean number of admission tags per note, variant A",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "mean_tags_per_note_b": claim_entry(
            value=_val("B", "__mean_tags_per_note__"),
            format_default=".2f",
            description="Mean number of admission tags per note, variant B",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "mean_tags_per_note_c": claim_entry(
            value=_val("C", "__mean_tags_per_note__"),
            format_default=".2f",
            description="Mean number of admission tags per note, variant C",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "max_tags_per_note_observed": claim_entry(
            value=int(
                max(
                    int(
                        df[(df["variant"] == v) & (df["tag"] == "__mean_tags_per_note__")][
                            "max_tags"
                        ].iloc[0]
                    )
                    for v in ("A", "B", "C")
                )
            ),
            format_default="d",
            description="Maximum number of tags assigned to any single note",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "a_other_prevalence_pct": claim_entry(
            value=_val("A", "other"),
            format_default=".2f",
            unit="%",
            description="Prevalence of tag 'other' in variant A",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "b_other_prevalence_pct": claim_entry(
            value=_val("B", "other"),
            format_default=".2f",
            unit="%",
            description="Prevalence of tag 'other' in variant B",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "c_other_prevalence_pct": claim_entry(
            value=_val("C", "other"),
            format_default=".2f",
            unit="%",
            description="Prevalence of tag 'other' in variant C",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "c_symptom_workup_other_prevalence_pct": claim_entry(
            value=_val("C", "symptom_workup_other"),
            format_default=".2f",
            unit="%",
            description="Prevalence of tag symptom_workup_other in variant C",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "a_symptom_workup_other_prevalence_pct": claim_entry(
            value=_val("A", "symptom_workup_other"),
            format_default=".2f",
            unit="%",
            description="Prevalence of tag symptom_workup_other in variant A",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "b_symptom_workup_other_prevalence_pct": claim_entry(
            value=_val("B", "symptom_workup_other"),
            format_default=".2f",
            unit="%",
            description="Prevalence of tag symptom_workup_other in variant B",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "c_symptom_workup_other_normalized_share_pct": claim_entry(
            value=_val("C", "symptom_workup_other", "share_pct"),
            format_default=".2f",
            unit="%",
            description="Normalized share of symptom_workup_other tag firings in variant C",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "a_symptom_workup_other_normalized_share_pct": claim_entry(
            value=_val("A", "symptom_workup_other", "share_pct"),
            format_default=".2f",
            unit="%",
            description="Normalized share of symptom_workup_other tag firings in variant A",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "b_symptom_workup_other_normalized_share_pct": claim_entry(
            value=_val("B", "symptom_workup_other", "share_pct"),
            format_default=".2f",
            unit="%",
            description="Normalized share of symptom_workup_other tag firings in variant B",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "a_infection_other_prevalence_pct": claim_entry(
            value=_val("A", "infection_other"),
            format_default=".2f",
            unit="%",
            description="Prevalence of tag infection_other in variant A",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "b_infection_other_prevalence_pct": claim_entry(
            value=_val("B", "infection_other"),
            format_default=".2f",
            unit="%",
            description="Prevalence of tag infection_other in variant B",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "c_infection_other_prevalence_pct": claim_entry(
            value=_val("C", "infection_other"),
            format_default=".2f",
            unit="%",
            description="Prevalence of tag infection_other in variant C",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "ab_dominant_diagonal_mass_pct": claim_entry(
            value=ab_dom,
            format_default=".2f",
            unit="%",
            description="Dominant admission tag agreement rate (A vs B)",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "ac_dominant_diagonal_mass_pct": claim_entry(
            value=ac_dom,
            format_default=".2f",
            unit="%",
            description="Dominant admission tag agreement rate (A vs C)",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "bc_dominant_diagonal_mass_pct": claim_entry(
            value=bc_dom,
            format_default=".2f",
            unit="%",
            description="Dominant admission tag agreement rate (B vs C)",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "ab_admission_tag_jaccard_pct": claim_entry(
            value=tag_jaccard_small["A-B"],
            format_default=".2f",
            unit="%",
            description="Admission-tag set agreement (mean per-note Jaccard), variant A vs B",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "ac_admission_tag_jaccard_pct": claim_entry(
            value=tag_jaccard_small["A-C"],
            format_default=".2f",
            unit="%",
            description="Admission-tag set agreement (mean per-note Jaccard), variant A vs C",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "bc_admission_tag_jaccard_pct": claim_entry(
            value=tag_jaccard_small["B-C"],
            format_default=".2f",
            unit="%",
            description="Admission-tag set agreement (mean per-note Jaccard), variant B vs C",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "paired_full_admission_tag_jaccard_pct": claim_entry(
            value=paired_full_tag_jaccard_pct,
            format_default=".2f",
            unit="%",
            description=(
                "Mean pairwise admission-tag set agreement (mean per-note Jaccard) on paired full-model "
                "methodology_1500 set (A-B, A-C, B-C)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "ab_full_admission_tag_jaccard_pct": claim_entry(
            value=paired_full_tag_pairs_pct["AB"],
            format_default=".2f",
            unit="%",
            description=(
                "Admission-tag set agreement (mean per-note Jaccard) on paired full-model methodology_1500 "
                "(variant A vs B)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "ac_full_admission_tag_jaccard_pct": claim_entry(
            value=paired_full_tag_pairs_pct["AC"],
            format_default=".2f",
            unit="%",
            description=(
                "Admission-tag set agreement (mean per-note Jaccard) on paired full-model methodology_1500 "
                "(variant A vs C)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "bc_full_admission_tag_jaccard_pct": claim_entry(
            value=paired_full_tag_pairs_pct["BC"],
            format_default=".2f",
            unit="%",
            description=(
                "Admission-tag set agreement (mean per-note Jaccard) on paired full-model methodology_1500 "
                "(variant B vs C)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "paired_full_dominant_diagonal_mass_pct": claim_entry(
            value=paired_full_dom_diag_mass_pct,
            format_default=".2f",
            unit="%",
            description=(
                "Mean pairwise dominant-admission diagonal mass on paired full-model "
                "methodology_1500 set (A-B, A-C, B-C)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "ab_full_dominant_diagonal_mass_pct": claim_entry(
            value=paired_full_dom_pairs_pct["AB"],
            format_default=".2f",
            unit="%",
            description=(
                "Dominant-admission diagonal mass on paired full-model methodology_1500 "
                "(variant A vs B)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "ac_full_dominant_diagonal_mass_pct": claim_entry(
            value=paired_full_dom_pairs_pct["AC"],
            format_default=".2f",
            unit="%",
            description=(
                "Dominant-admission diagonal mass on paired full-model methodology_1500 "
                "(variant A vs C)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "bc_full_dominant_diagonal_mass_pct": claim_entry(
            value=paired_full_dom_pairs_pct["BC"],
            format_default=".2f",
            unit="%",
            description=(
                "Dominant-admission diagonal mass on paired full-model methodology_1500 "
                "(variant B vs C)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "aa_model_size_dominant_diagonal_mass_pct": claim_entry(
            value=model_size_within_variant_pct["AA"],
            format_default=".2f",
            unit="%",
            description=(
                "Dominant-admission diagonal mass on paired methodology_1500, variant A "
                "(small model vs full model)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "aa_model_size_admission_tag_jaccard_pct": claim_entry(
            value=model_size_tag_jaccard_pct["AA"],
            format_default=".2f",
            unit="%",
            description=(
                "Admission-tag set agreement (mean per-note Jaccard) on paired methodology_1500, variant A "
                "(small model vs full model)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "aa_model_size_offdiagonal_mass_pct": claim_entry(
            value=100.0 - model_size_within_variant_pct["AA"],
            format_default=".2f",
            unit="%",
            description=(
                "Dominant-admission off-diagonal mass on paired methodology_1500, variant A "
                "(small model vs full model); computed as 100 minus diagonal mass"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "bb_model_size_dominant_diagonal_mass_pct": claim_entry(
            value=model_size_within_variant_pct["BB"],
            format_default=".2f",
            unit="%",
            description=(
                "Dominant-admission diagonal mass on paired methodology_1500, variant B "
                "(small model vs full model)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "bb_model_size_admission_tag_jaccard_pct": claim_entry(
            value=model_size_tag_jaccard_pct["BB"],
            format_default=".2f",
            unit="%",
            description=(
                "Admission-tag set agreement (mean per-note Jaccard) on paired methodology_1500, variant B "
                "(small model vs full model)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "bb_model_size_offdiagonal_mass_pct": claim_entry(
            value=100.0 - model_size_within_variant_pct["BB"],
            format_default=".2f",
            unit="%",
            description=(
                "Dominant-admission off-diagonal mass on paired methodology_1500, variant B "
                "(small model vs full model); computed as 100 minus diagonal mass"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "cc_model_size_dominant_diagonal_mass_pct": claim_entry(
            value=model_size_within_variant_pct["CC"],
            format_default=".2f",
            unit="%",
            description=(
                "Dominant-admission diagonal mass on paired methodology_1500, variant C "
                "(small model vs full model)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "cc_model_size_admission_tag_jaccard_pct": claim_entry(
            value=model_size_tag_jaccard_pct["CC"],
            format_default=".2f",
            unit="%",
            description=(
                "Admission-tag set agreement (mean per-note Jaccard) on paired methodology_1500, variant C "
                "(small model vs full model)"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "cc_model_size_offdiagonal_mass_pct": claim_entry(
            value=100.0 - model_size_within_variant_pct["CC"],
            format_default=".2f",
            unit="%",
            description=(
                "Dominant-admission off-diagonal mass on paired methodology_1500, variant C "
                "(small model vs full model); computed as 100 minus diagonal mass"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "tag_confusion_prevalence_spearman_rho": claim_entry(
            value=float(spearman_rho),
            format_default=".4f",
            description=(
                "Spearman correlation between per-tag dominant-label prevalence and "
                "per-tag mean cross-variant confusion mass across the 47-tag vocabulary"
            ),
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "tag_confusion_prevalence_spearman_pvalue": claim_entry(
            value=float(spearman_pvalue),
            format_default=".2e",
            description=(
                "P-value for Spearman correlation between per-tag dominant-label prevalence "
                "and per-tag mean cross-variant confusion mass across the 47-tag vocabulary"
            ),
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "residual_mass_small_pct": claim_entry(
            value=residual_metrics["residual_mass_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "A/B/C-averaged share of total admission-tag firing mass on the residual family "
                "(other + *_other tags), small model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "residual_mass_full_pct": claim_entry(
            value=residual_metrics["residual_mass_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "A/B/C-averaged share of total admission-tag firing mass on the residual family "
                "(other + *_other tags), full model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "other_tag_mass_small_pct": claim_entry(
            value=residual_metrics["other_tag_mass_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "A/B/C-averaged share of total admission-tag firing mass on tag 'other', "
                "small model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "other_tag_mass_full_pct": claim_entry(
            value=residual_metrics["other_tag_mass_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "A/B/C-averaged share of total admission-tag firing mass on tag 'other', "
                "full model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "elective_procedure_non_oncology_mass_small_pct": claim_entry(
            value=residual_metrics["elective_procedure_non_oncology_mass_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "A/B/C-averaged share of total admission-tag firing mass on tag "
                "elective_procedure_non_oncology, small model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "elective_procedure_non_oncology_mass_full_pct": claim_entry(
            value=residual_metrics["elective_procedure_non_oncology_mass_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "A/B/C-averaged share of total admission-tag firing mass on tag "
                "elective_procedure_non_oncology, full model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "heme_onc_complication_mass_small_pct": claim_entry(
            value=residual_metrics["heme_onc_complication_mass_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "A/B/C-averaged share of total admission-tag firing mass on tag "
                "heme_onc_complication, small model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "heme_onc_complication_mass_full_pct": claim_entry(
            value=residual_metrics["heme_onc_complication_mass_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "A/B/C-averaged share of total admission-tag firing mass on tag "
                "heme_onc_complication, full model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "trauma_fracture_mass_small_pct": claim_entry(
            value=residual_metrics["trauma_fracture_mass_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "A/B/C-averaged share of total admission-tag firing mass on tag trauma_fracture, "
                "small model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "trauma_fracture_mass_full_pct": claim_entry(
            value=residual_metrics["trauma_fracture_mass_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "A/B/C-averaged share of total admission-tag firing mass on tag trauma_fracture, "
                "full model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "neuro_other_confusion_rate_small_pct": claim_entry(
            value=residual_metrics["neuro_other_confusion_rate_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag neuro_other, small model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "neuro_other_confusion_rate_full_pct": claim_entry(
            value=residual_metrics["neuro_other_confusion_rate_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag neuro_other, full model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "gi_other_confusion_rate_small_pct": claim_entry(
            value=residual_metrics["gi_other_confusion_rate_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag gi_other, small model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "gi_other_confusion_rate_full_pct": claim_entry(
            value=residual_metrics["gi_other_confusion_rate_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag gi_other, full model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "gu_other_confusion_rate_small_pct": claim_entry(
            value=residual_metrics["gu_other_confusion_rate_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag gu_other, small model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "gu_other_confusion_rate_full_pct": claim_entry(
            value=residual_metrics["gu_other_confusion_rate_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag gu_other, full model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "infection_other_confusion_rate_small_pct": claim_entry(
            value=residual_metrics["infection_other_confusion_rate_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag infection_other, small model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "infection_other_confusion_rate_full_pct": claim_entry(
            value=residual_metrics["infection_other_confusion_rate_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag infection_other, full model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "respiratory_failure_other_confusion_rate_small_pct": claim_entry(
            value=residual_metrics["respiratory_failure_other_confusion_rate_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag respiratory_failure_other, "
                "small model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "respiratory_failure_other_confusion_rate_full_pct": claim_entry(
            value=residual_metrics["respiratory_failure_other_confusion_rate_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag respiratory_failure_other, "
                "full model, paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "cardiac_other_confusion_rate_small_pct": claim_entry(
            value=residual_metrics["cardiac_other_confusion_rate_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag cardiac_other, small model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "cardiac_other_confusion_rate_full_pct": claim_entry(
            value=residual_metrics["cardiac_other_confusion_rate_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag cardiac_other, full model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "trauma_other_confusion_rate_small_pct": claim_entry(
            value=residual_metrics["trauma_other_confusion_rate_small_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag trauma_other, small model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "trauma_other_confusion_rate_full_pct": claim_entry(
            value=residual_metrics["trauma_other_confusion_rate_full_pct"],
            format_default=".2f",
            unit="%",
            description=(
                "Normalized multi-label presence confusion rate for tag trauma_other, full model, "
                "paired methodology_1500"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
    }
    return claims


def main() -> int:
    new_claims = compute_all()
    n = merge_into_claims_json(CLAIMS_PATH, new_claims)
    claims_obj = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    legacy_keys = [
        key
        for key in list(claims_obj.keys())
        if ("admission_tag" in key and "diagonal_mass" in key and "dominant" not in key)
    ]
    removed = 0
    for key in legacy_keys:
        if key in claims_obj:
            claims_obj.pop(key, None)
            removed += 1
    if removed:
        CLAIMS_PATH.write_text(json.dumps(claims_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Updated {n} claims in {CLAIMS_PATH}")
    if removed:
        print(f"Removed {removed} legacy admission-tag diagonal-mass claim keys from {CLAIMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
