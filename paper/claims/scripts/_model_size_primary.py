from __future__ import annotations

# Release documentation:
# Provides shared helpers for claim-registry recomputation.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Supports paper claim recomputation and receipt verification.

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from src.paper_figures.data_loaders import get_tristate_field_list, load_paired_full_extractions


PAIR_KEYS: tuple[tuple[str, str], ...] = (("A", "B"), ("A", "C"), ("B", "C"))


@dataclass(frozen=True)
class PairedModelSizeData:
    small_by_variant: dict[str, pd.DataFrame]
    full_by_variant: dict[str, pd.DataFrame]
    common_hadm_ids: list[int]
    tri_fields: list[str]
    included_fields: list[str]


def _normalize_token(value: Any) -> str:
    if value is None:
        return "null"
    token = str(value).strip().lower()
    if token in {"yes", "no", "not_documented"}:
        return token
    if token in {"", "none", "null"}:
        return "null"
    return "not_documented"


def _encode_tristate(value: Any, *, collapse: bool) -> int:
    token = _normalize_token(value)
    if collapse:
        if token == "yes":
            return 1
        if token in {"no", "not_documented"}:
            return 0
        return 2
    mapping = {"yes": 1, "no": -1, "not_documented": 0, "null": 2}
    return mapping[token]


def _is_yes(value: Any) -> int:
    return 1 if _normalize_token(value) == "yes" else 0


def _safe_kappa(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.shape != right.shape:
        return float("nan")
    if np.unique(left).size == 1 and np.unique(right).size == 1:
        return 1.0 if np.array_equal(left, right) else 0.0
    value = float(cohen_kappa_score(left, right))
    if np.isnan(value):
        return 1.0 if np.array_equal(left, right) else 0.0
    return value


def _by_variant(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for variant in ("A", "B", "C"):
        sub = (
            frame[frame["variant"] == variant]
            .drop_duplicates(subset=["hadm_id"], keep="last")
            .copy()
        )
        if sub.empty:
            raise ValueError(f"Missing variant {variant} rows in paired extraction frame.")
        out[variant] = sub.set_index("hadm_id")
    return out


def load_paired_model_size_data(base_rate_threshold: int = 10) -> PairedModelSizeData:
    """Load strict paired 1500-note model-size data and included TriState field set."""
    small = load_paired_full_extractions("small").copy()
    full = load_paired_full_extractions("full").copy()
    if small.empty or full.empty:
        raise ValueError("Paired extraction frames are empty.")

    small_by_variant = _by_variant(small)
    full_by_variant = _by_variant(full)
    common_hadm_ids = sorted(
        set(small_by_variant["A"].index)
        & set(small_by_variant["B"].index)
        & set(small_by_variant["C"].index)
        & set(full_by_variant["A"].index)
        & set(full_by_variant["B"].index)
        & set(full_by_variant["C"].index)
    )
    if not common_hadm_ids:
        raise ValueError("No shared hadm_id across paired small/full A/B/C runs.")

    tri_fields = get_tristate_field_list("base")
    included_fields: list[str] = []
    for field in tri_fields:
        small_yes = sum(
            int(np.sum([_is_yes(v) for v in small_by_variant[var].loc[common_hadm_ids, field].tolist()]))
            for var in ("A", "B", "C")
        )
        full_yes = sum(
            int(np.sum([_is_yes(v) for v in full_by_variant[var].loc[common_hadm_ids, field].tolist()]))
            for var in ("A", "B", "C")
        )
        if small_yes >= base_rate_threshold and full_yes >= base_rate_threshold:
            included_fields.append(field)

    return PairedModelSizeData(
        small_by_variant=small_by_variant,
        full_by_variant=full_by_variant,
        common_hadm_ids=common_hadm_ids,
        tri_fields=tri_fields,
        included_fields=included_fields,
    )


def pooled_kappa_levels(data: PairedModelSizeData) -> tuple[float, float, float, float]:
    """Return pooled median kappa levels on included fields:
    (small_tri, full_tri, small_collapsed, full_collapsed).
    """
    if not data.included_fields:
        raise ValueError("No fields passed dual-model base-rate inclusion filter.")

    small_tri: list[float] = []
    full_tri: list[float] = []
    small_col: list[float] = []
    full_col: list[float] = []
    common = data.common_hadm_ids

    for field in data.included_fields:
        for left, right in PAIR_KEYS:
            s_left_raw = data.small_by_variant[left].loc[common, field].tolist()
            s_right_raw = data.small_by_variant[right].loc[common, field].tolist()
            f_left_raw = data.full_by_variant[left].loc[common, field].tolist()
            f_right_raw = data.full_by_variant[right].loc[common, field].tolist()

            s_tri_l = np.asarray([_encode_tristate(v, collapse=False) for v in s_left_raw], dtype=np.int8)
            s_tri_r = np.asarray([_encode_tristate(v, collapse=False) for v in s_right_raw], dtype=np.int8)
            f_tri_l = np.asarray([_encode_tristate(v, collapse=False) for v in f_left_raw], dtype=np.int8)
            f_tri_r = np.asarray([_encode_tristate(v, collapse=False) for v in f_right_raw], dtype=np.int8)
            s_col_l = np.asarray([_encode_tristate(v, collapse=True) for v in s_left_raw], dtype=np.int8)
            s_col_r = np.asarray([_encode_tristate(v, collapse=True) for v in s_right_raw], dtype=np.int8)
            f_col_l = np.asarray([_encode_tristate(v, collapse=True) for v in f_left_raw], dtype=np.int8)
            f_col_r = np.asarray([_encode_tristate(v, collapse=True) for v in f_right_raw], dtype=np.int8)

            small_tri.append(_safe_kappa(s_tri_l, s_tri_r))
            full_tri.append(_safe_kappa(f_tri_l, f_tri_r))
            small_col.append(_safe_kappa(s_col_l, s_col_r))
            full_col.append(_safe_kappa(f_col_l, f_col_r))

    return (
        float(np.median(np.asarray(small_tri, dtype=np.float64))),
        float(np.median(np.asarray(full_tri, dtype=np.float64))),
        float(np.median(np.asarray(small_col, dtype=np.float64))),
        float(np.median(np.asarray(full_col, dtype=np.float64))),
    )


def per_field_model_size_deltas(data: PairedModelSizeData) -> tuple[dict[str, float], dict[str, float]]:
    """Return per-field model-size deltas in pp for all 17 TriState fields.

    Delta per field is median over pairwise deltas:
    median_{A-B,A-C,B-C}(kappa_full - kappa_small)*100.
    """
    full_delta_pp: dict[str, float] = {}
    collapsed_delta_pp: dict[str, float] = {}
    common = data.common_hadm_ids

    for field in data.tri_fields:
        pair_full: list[float] = []
        pair_collapsed: list[float] = []
        for left, right in PAIR_KEYS:
            s_left_raw = data.small_by_variant[left].loc[common, field].tolist()
            s_right_raw = data.small_by_variant[right].loc[common, field].tolist()
            f_left_raw = data.full_by_variant[left].loc[common, field].tolist()
            f_right_raw = data.full_by_variant[right].loc[common, field].tolist()

            s_tri_l = np.asarray([_encode_tristate(v, collapse=False) for v in s_left_raw], dtype=np.int8)
            s_tri_r = np.asarray([_encode_tristate(v, collapse=False) for v in s_right_raw], dtype=np.int8)
            f_tri_l = np.asarray([_encode_tristate(v, collapse=False) for v in f_left_raw], dtype=np.int8)
            f_tri_r = np.asarray([_encode_tristate(v, collapse=False) for v in f_right_raw], dtype=np.int8)
            s_col_l = np.asarray([_encode_tristate(v, collapse=True) for v in s_left_raw], dtype=np.int8)
            s_col_r = np.asarray([_encode_tristate(v, collapse=True) for v in s_right_raw], dtype=np.int8)
            f_col_l = np.asarray([_encode_tristate(v, collapse=True) for v in f_left_raw], dtype=np.int8)
            f_col_r = np.asarray([_encode_tristate(v, collapse=True) for v in f_right_raw], dtype=np.int8)

            k_s_tri = _safe_kappa(s_tri_l, s_tri_r)
            k_f_tri = _safe_kappa(f_tri_l, f_tri_r)
            k_s_col = _safe_kappa(s_col_l, s_col_r)
            k_f_col = _safe_kappa(f_col_l, f_col_r)

            pair_full.append((k_f_tri - k_s_tri) * 100.0)
            pair_collapsed.append((k_f_col - k_s_col) * 100.0)

        full_delta_pp[field] = float(np.median(np.asarray(pair_full, dtype=np.float64)))
        collapsed_delta_pp[field] = float(np.median(np.asarray(pair_collapsed, dtype=np.float64)))

    return full_delta_pp, collapsed_delta_pp


def mean_pairwise_agreements(data: PairedModelSizeData) -> tuple[float, float, float, float]:
    """Return mean pairwise agreements (small_tag_jaccard, full_tag_jaccard, small_primary, full_primary)."""
    common = data.common_hadm_ids

    def _tag_mass(by_variant: dict[str, pd.DataFrame], left: str, right: str) -> float:
        per_note: list[float] = []
        for hid in common:
            left_tags = by_variant[left].at[hid, "admission_reason_tags"]
            right_tags = by_variant[right].at[hid, "admission_reason_tags"]
            lt = set(map(str, left_tags if isinstance(left_tags, (list, tuple, np.ndarray)) else []))
            rt = set(map(str, right_tags if isinstance(right_tags, (list, tuple, np.ndarray)) else []))
            union = lt | rt
            if not union:
                per_note.append(1.0)
            else:
                per_note.append(float(len(lt & rt) / len(union)))
        return float(np.nanmean(np.asarray(per_note, dtype=float))) if per_note else float("nan")

    def _primary_mass(by_variant: dict[str, pd.DataFrame], left: str, right: str) -> float:
        agree = 0
        valid = 0
        for hid in common:
            lv = by_variant[left].at[hid, "dominant_admission_reason"]
            rv = by_variant[right].at[hid, "dominant_admission_reason"]
            if pd.isna(lv) or pd.isna(rv):
                continue
            valid += 1
            if str(lv) == str(rv):
                agree += 1
        return float(agree / valid) if valid > 0 else float("nan")

    small_tag_vals = np.asarray(
        [_tag_mass(data.small_by_variant, "A", "B"), _tag_mass(data.small_by_variant, "A", "C"), _tag_mass(data.small_by_variant, "B", "C")],
        dtype=float,
    )
    full_tag_vals = np.asarray(
        [_tag_mass(data.full_by_variant, "A", "B"), _tag_mass(data.full_by_variant, "A", "C"), _tag_mass(data.full_by_variant, "B", "C")],
        dtype=float,
    )
    small_primary_vals = np.asarray(
        [_primary_mass(data.small_by_variant, "A", "B"), _primary_mass(data.small_by_variant, "A", "C"), _primary_mass(data.small_by_variant, "B", "C")],
        dtype=float,
    )
    full_primary_vals = np.asarray(
        [_primary_mass(data.full_by_variant, "A", "B"), _primary_mass(data.full_by_variant, "A", "C"), _primary_mass(data.full_by_variant, "B", "C")],
        dtype=float,
    )

    return (
        float(np.nanmean(small_tag_vals)),
        float(np.nanmean(full_tag_vals)),
        float(np.nanmean(small_primary_vals)),
        float(np.nanmean(full_primary_vals)),
    )


def mean_pairwise_diagonal_masses(data: PairedModelSizeData) -> tuple[float, float, float, float]:
    """Backward-compatible alias for mean_pairwise_agreements."""
    return mean_pairwise_agreements(data)


def per_variant_cross_model_kappa(
    data: PairedModelSizeData,
) -> dict[str, tuple[float, float]]:
    """Return per-variant small-vs-full kappa medians on included fields.

    For each variant (A/B/C), computes per-field kappa between small and full
    outputs on the same hadm_ids, then takes median over included fields.
    Returns:
        {
            "A": (tri_state_median, collapsed_median),
            "B": (...),
            "C": (...),
        }
    """
    if not data.included_fields:
        raise ValueError("No fields passed dual-model base-rate inclusion filter.")

    out: dict[str, tuple[float, float]] = {}
    common = data.common_hadm_ids

    for variant in ("A", "B", "C"):
        tri_vals: list[float] = []
        col_vals: list[float] = []
        for field in data.included_fields:
            small_raw = data.small_by_variant[variant].loc[common, field].tolist()
            full_raw = data.full_by_variant[variant].loc[common, field].tolist()

            s_tri = np.asarray([_encode_tristate(v, collapse=False) for v in small_raw], dtype=np.int8)
            f_tri = np.asarray([_encode_tristate(v, collapse=False) for v in full_raw], dtype=np.int8)
            s_col = np.asarray([_encode_tristate(v, collapse=True) for v in small_raw], dtype=np.int8)
            f_col = np.asarray([_encode_tristate(v, collapse=True) for v in full_raw], dtype=np.int8)

            tri_vals.append(_safe_kappa(s_tri, f_tri))
            col_vals.append(_safe_kappa(s_col, f_col))

        out[variant] = (
            float(np.median(np.asarray(tri_vals, dtype=np.float64))),
            float(np.median(np.asarray(col_vals, dtype=np.float64))),
        )

    return out
