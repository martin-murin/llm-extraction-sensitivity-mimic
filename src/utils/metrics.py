"""Shared metric helpers for paper claims and figure computations.

This module provides a canonical implementation of filtered median cross-variant
kappa for the project's TriState fields, with optional binary collapse:
yes vs not_yes (no + not_documented).
"""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from src.schema.fields import LLMNoteFeatures
from src.utils.threeway_kappa import low_base_rate_flag

TRISTATE_DOMAIN = {"yes", "no", "not_documented"}


def _literal_values(annotation: Any) -> set[str] | None:
    origin = get_origin(annotation)
    if origin is Literal:
        return {arg for arg in get_args(annotation) if isinstance(arg, str)}
    args = get_args(annotation)
    if not args:
        return None
    out: set[str] = set()
    saw_literal = False
    for arg in args:
        if arg is type(None):
            continue
        sub = _literal_values(arg)
        if sub is None:
            return None
        out |= sub
        saw_literal = True
    return out if saw_literal else None


def tristate_fields_base() -> list[str]:
    """Return TriState fields in the base LLMNoteFeatures schema."""
    out: list[str] = []
    for name, field in LLMNoteFeatures.model_fields.items():
        values = _literal_values(field.annotation)
        if values == TRISTATE_DOMAIN:
            out.append(name)
    return sorted(out)


def _normalize_tristate_token(value: Any, collapse: bool) -> str:
    if value is None:
        return "null"
    norm = str(value).strip().lower()
    if norm == "yes":
        return "yes"
    if norm == "no":
        return "not_yes" if collapse else "no"
    if norm == "not_documented":
        return "not_yes" if collapse else "not_documented"
    if norm in {"", "none", "null"}:
        return "null"
    return "not_yes" if collapse else "not_documented"


def _encode_tristate(value: Any, collapse: bool) -> int:
    token = _normalize_tristate_token(value, collapse=collapse)
    if collapse:
        mapping = {"yes": 1, "not_yes": 0, "null": 2}
    else:
        mapping = {"yes": 1, "no": -1, "not_documented": 0, "null": 2}
    return mapping[token]


def _safe_kappa(left: list[int], right: list[int]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    if len(set(left)) == 1 and len(set(right)) == 1 and left[0] == right[0]:
        return 1.0
    try:
        value = float(cohen_kappa_score(left, right))
    except Exception:
        return 0.0
    if np.isnan(value):
        return 0.0
    return value


def filtered_median_kappa(
    df: pd.DataFrame,
    fields: list[str] | Literal["tristate"] = "tristate",
    collapse: bool = False,
) -> float:
    """Compute filtered median cross-variant kappa across fields.

    Args:
        df: Long-format frame containing columns:
            - hadm_id
            - variant (A/B/C)
            - one column per field being evaluated
        fields: list of field names or literal "tristate" for base TriState list
        collapse: if True, apply yes-vs-not_yes collapse for TriState encoding

    Returns:
        Median of per-field mean pairwise kappa values (A-B, A-C, B-C),
        filtering out low-base-rate fields via `low_base_rate_flag`.
    """
    required = {"hadm_id", "variant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"filtered_median_kappa missing required columns: {sorted(missing)}")

    if fields == "tristate":
        field_list = tristate_fields_base()
    else:
        field_list = list(fields)
    if not field_list:
        raise ValueError("No fields provided to filtered_median_kappa.")

    parsed = df.copy()
    parsed["variant"] = parsed["variant"].astype(str).str.upper()
    parsed["hadm_id"] = pd.to_numeric(parsed["hadm_id"], errors="coerce")
    parsed = parsed.dropna(subset=["hadm_id"]).copy()
    parsed["hadm_id"] = parsed["hadm_id"].astype("int64")

    by_variant: dict[str, pd.DataFrame] = {
        v: parsed[parsed["variant"] == v].drop_duplicates(subset=["hadm_id"], keep="last").copy()
        for v in ("A", "B", "C")
    }
    if any(frame.empty for frame in by_variant.values()):
        raise ValueError("filtered_median_kappa requires non-empty A/B/C variant slices.")

    common = sorted(
        set(by_variant["A"]["hadm_id"])
        & set(by_variant["B"]["hadm_id"])
        & set(by_variant["C"]["hadm_id"])
    )
    if not common:
        raise ValueError("No shared hadm_id across A/B/C variants.")

    for v in ("A", "B", "C"):
        by_variant[v] = by_variant[v][by_variant[v]["hadm_id"].isin(common)].set_index("hadm_id")

    kappas: list[float] = []
    for field in field_list:
        if field not in by_variant["A"].columns:
            continue
        va = [_encode_tristate(x, collapse=collapse) for x in by_variant["A"][field].tolist()]
        vb = [_encode_tristate(x, collapse=collapse) for x in by_variant["B"][field].tolist()]
        vc = [_encode_tristate(x, collapse=collapse) for x in by_variant["C"][field].tolist()]

        k_ab = _safe_kappa(va, vb)
        k_ac = _safe_kappa(va, vc)
        k_bc = _safe_kappa(vb, vc)
        kappa_mean = float(np.mean([k_ab, k_ac, k_bc]))

        n_positive_total = int(sum(v == 1 for v in va) + sum(v == 1 for v in vb) + sum(v == 1 for v in vc))
        if not low_base_rate_flag(n_positive_total=n_positive_total):
            kappas.append(kappa_mean)

    if not kappas:
        raise ValueError("No non-low-base-rate fields remained for filtered median kappa.")
    return float(np.median(np.asarray(kappas, dtype=np.float64)))
