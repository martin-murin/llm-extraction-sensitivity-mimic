from __future__ import annotations

# Release documentation:
# Provides shared helpers for claim-registry recomputation.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Supports paper claim recomputation and receipt verification.

import json
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import pandas as pd

from src.schema.fields import LLMNoteFeatures

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw_responses"
SPLITS = REPO / "data" / "splits"

TRISTATE_SET = {"yes", "no", "not_documented"}
PAIR_KEYS: tuple[tuple[str, str], ...] = (("A", "B"), ("A", "C"), ("B", "C"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_variant(run_id: str, *, allowed_hadm_ids: set[int] | None = None) -> dict[int, dict[str, Any]]:
    path = RAW / run_id / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing run results file: {path}")
    parsed: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if not bool(row.get("parse_ok", False)):
            continue
        feats = row.get("features_json")
        if not isinstance(feats, dict):
            continue
        hadm_id = int(row["hadm_id"])
        if allowed_hadm_ids is not None and hadm_id not in allowed_hadm_ids:
            continue
        parsed[hadm_id] = feats
    return parsed


def _read_split_ids(filename: str) -> set[int]:
    path = SPLITS / filename
    frame = pd.read_csv(path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").dropna().astype("int64").tolist())


def _is_tristate_field(annotation: Any) -> bool:
    return get_origin(annotation) is Literal and set(get_args(annotation)) == TRISTATE_SET


def _tristate_fields() -> list[str]:
    out: list[str] = []
    for name, field in LLMNoteFeatures.model_fields.items():
        if _is_tristate_field(field.annotation):
            out.append(name)
    return sorted(out)


def _normalize(value: Any) -> str | None:
    if isinstance(value, str) and value in TRISTATE_SET:
        return value
    return None


def _original_category(left: str | None, right: str | None) -> str | None:
    if left is None or right is None:
        return None
    if left == right:
        return "full_agreement"
    pair = {left, right}
    if pair == {"no", "not_documented"}:
        return "soft_no_not_documented"
    if pair == {"yes", "not_documented"}:
        return "soft_yes_not_documented"
    if pair == {"yes", "no"}:
        return "hard_yes_no"
    return None


def _collapsed_category(left: str | None, right: str | None) -> str:
    if left is None or right is None:
        return "null_disagreement"
    left_collapsed = "yes" if left == "yes" else "not_yes"
    right_collapsed = "yes" if right == "yes" else "not_yes"
    if left_collapsed == right_collapsed:
        return "full_agreement_collapsed"
    return "residual_yes_vs_not_yes_disagreement"


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def compute_cross_variant_pooled_disagreement_metrics() -> dict[str, float | int]:
    """Recompute disagreement metrics from pooled small-model cross-variant runs.

    Run mapping (1k + 500 + extended_5k pools):
    - A: methodology_1k_a + methodology_5k_a_subset500 + production_v1(on extended_5k split)
    - B: methodology_1k_b + methodology_5k_audit_b + extended_5k_b
    - C: methodology_1k_c + methodology_5k_audit_c + extended_5k_c
    """
    ids_1k = _read_split_ids("methodology_1k.csv")
    ids_500 = _read_split_ids("methodology_5k_audit_500.csv")
    ids_ext = _read_split_ids("extended_5k.csv")

    def _load_union(parts: list[tuple[str, set[int]]]) -> dict[int, dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        for run_id, allowed_ids in parts:
            merged.update(_load_variant(run_id, allowed_hadm_ids=allowed_ids))
        return merged

    by_variant = {
        "A": _load_union(
            [("methodology_1k_a", ids_1k), ("methodology_5k_a_subset500", ids_500), ("production_v1", ids_ext)]
        ),
        "B": _load_union(
            [("methodology_1k_b", ids_1k), ("methodology_5k_audit_b", ids_500), ("extended_5k_b", ids_ext)]
        ),
        "C": _load_union(
            [("methodology_1k_c", ids_1k), ("methodology_5k_audit_c", ids_500), ("extended_5k_c", ids_ext)]
        ),
    }
    hadm_ids = sorted(set(by_variant["A"]) & set(by_variant["B"]) & set(by_variant["C"]))
    if not hadm_ids:
        raise ValueError("No shared parse_ok hadm_id intersection for pooled cross-variant sample.")

    fields = _tristate_fields()
    if not fields:
        raise ValueError("No TriState fields found in schema.")

    original_counts = {
        "full_agreement": 0,
        "soft_no_not_documented": 0,
        "soft_yes_not_documented": 0,
        "hard_yes_no": 0,
    }
    collapsed_counts = {
        "full_agreement_collapsed": 0,
        "residual_yes_vs_not_yes_disagreement": 0,
        "null_disagreement": 0,
    }
    crosstab = {
        "soft_no_not_documented->full_agreement_collapsed": 0,
    }

    for field in fields:
        for left_variant, right_variant in PAIR_KEYS:
            for hadm_id in hadm_ids:
                left = _normalize(by_variant[left_variant][hadm_id].get(field))
                right = _normalize(by_variant[right_variant][hadm_id].get(field))

                original = _original_category(left, right)
                collapsed = _collapsed_category(left, right)
                collapsed_counts[collapsed] += 1

                if original is not None:
                    original_counts[original] += 1
                    if (
                        original == "soft_no_not_documented"
                        and collapsed == "full_agreement_collapsed"
                    ):
                        crosstab["soft_no_not_documented->full_agreement_collapsed"] += 1

    original_disagreements = (
        original_counts["soft_no_not_documented"]
        + original_counts["soft_yes_not_documented"]
        + original_counts["hard_yes_no"]
    )
    collapsed_disagreements = (
        collapsed_counts["residual_yes_vs_not_yes_disagreement"]
        + collapsed_counts["null_disagreement"]
    )
    dissolved = crosstab["soft_no_not_documented->full_agreement_collapsed"]
    preserved = max(0, original_disagreements - dissolved)

    return {
        "tri_state_fields": len(fields),
        "shared_parse_ok_hadm_ids": len(hadm_ids),
        "total_pairwise_comparisons": len(fields) * len(hadm_ids) * len(PAIR_KEYS),
        "disagreement_count_full_tristate": original_disagreements,
        "disagreement_count_collapsed": collapsed_disagreements,
        "disagreement_dissolved_pct": _pct(dissolved, original_disagreements),
        "disagreement_residual_pct": _pct(preserved, original_disagreements),
        "disagreement_soft_no_vs_not_documented_pct": _pct(
            original_counts["soft_no_not_documented"], original_disagreements
        ),
        "disagreement_soft_yes_vs_not_documented_pct": _pct(
            original_counts["soft_yes_not_documented"], original_disagreements
        ),
        "disagreement_hard_pct": _pct(original_counts["hard_yes_no"], original_disagreements),
        "disagreement_soft_pct": _pct(
            original_counts["soft_no_not_documented"] + original_counts["soft_yes_not_documented"],
            original_disagreements,
        ),
    }


def compute_methodology5k_disagreement_metrics() -> dict[str, float | int]:
    """Back-compat alias; disagreement claims now use pooled cross-variant sample."""
    return compute_cross_variant_pooled_disagreement_metrics()
