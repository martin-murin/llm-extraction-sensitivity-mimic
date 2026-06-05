from __future__ import annotations

# Release documentation:
# Runs staged pipeline step `96_paired_bootstrap_ci.py`.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs Figure 2 and model-size confidence-interval claims.
# Usage: `python scripts/96_paired_bootstrap_ci.py` unless the script's argparse help says otherwise.

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from src.paper_figures.data_loaders import get_tristate_field_list


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw_responses"
CLAIMS_PATH = REPO_ROOT / "paper" / "claims" / "claims.json"
DEFAULT_OUTPUT = REPO_ROOT / "codex_outputs" / "paired_bootstrap_ci.md"

VARIANTS = ("A", "B", "C")
PAIRS = (("A", "B"), ("A", "C"), ("B", "C"))
MODELS = ("small", "full")
SCHEMES = ("tristate", "collapsed")

RUN_MAP: dict[str, dict[str, tuple[str, str]]] = {
    "small": {
        "A": ("methodology_1k_a", "methodology_5k_a_subset500"),
        "B": ("methodology_1k_b", "methodology_5k_audit_b"),
        "C": ("methodology_1k_c", "methodology_5k_audit_c"),
    },
    "full": {
        "A": ("paired_gold_methodology_1k_a", "paired_gold_methodology_5k_audit_a"),
        "B": ("paired_gold_methodology_1k_b", "paired_gold_methodology_5k_audit_b"),
        "C": ("paired_gold_methodology_1k_c", "paired_gold_methodology_5k_audit_c"),
    },
}


@dataclass(frozen=True)
class AggregateStats:
    pooled_delta_tristate_pp: float
    pooled_delta_collapsed_pp: float
    perfield_delta_tristate_pp: float
    perfield_delta_collapsed_pp: float


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_run(run_id: str) -> dict[int, dict[str, Any]]:
    path = RAW_DIR / run_id / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing run results: {path}")
    out: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if not bool(row.get("parse_ok", False)):
            continue
        feats = row.get("features_json")
        if not isinstance(feats, dict):
            continue
        out[int(row["hadm_id"])] = feats
    return out


def _merge_disjoint(left: dict[int, dict[str, Any]], right: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    overlap = set(left) & set(right)
    if overlap:
        raise RuntimeError(f"Expected disjoint split runs; found overlap n={len(overlap)}")
    merged = dict(left)
    merged.update(right)
    return merged


def _normalize_tristate(value: Any) -> str:
    if value is None:
        return "null"
    token = str(value).strip().lower()
    if token in {"yes", "no", "not_documented"}:
        return token
    if token in {"", "none", "null"}:
        return "null"
    return "not_documented"


def _encode_tristate(value: Any, *, collapsed: bool) -> int:
    token = _normalize_tristate(value)
    if collapsed:
        if token == "yes":
            return 1
        if token in {"no", "not_documented"}:
            return 0
        return 2  # null
    mapping = {"yes": 1, "no": -1, "not_documented": 0, "null": 2}
    return mapping[token]


def _is_yes(value: Any) -> int:
    return 1 if _normalize_tristate(value) == "yes" else 0


def _safe_kappa(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.shape != right.shape:
        return float("nan")
    if np.unique(left).size == 1 and np.unique(right).size == 1:
        return 1.0 if np.array_equal(left, right) else 0.0
    val = float(cohen_kappa_score(left, right))
    if np.isnan(val):
        return 1.0 if np.array_equal(left, right) else 0.0
    return val


def _percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(clean, alpha / 2))
    hi = float(np.quantile(clean, 1 - alpha / 2))
    return lo, hi


def _build_model_variant_maps() -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    out: dict[str, dict[str, dict[int, dict[str, Any]]]] = {m: {} for m in MODELS}
    for model in MODELS:
        for variant in VARIANTS:
            run_1k, run_500 = RUN_MAP[model][variant]
            left = _load_run(run_1k)
            right = _load_run(run_500)
            out[model][variant] = _merge_disjoint(left, right)
    return out


def _shared_hadm_ids(by_model_variant: dict[str, dict[str, dict[int, dict[str, Any]]]]) -> list[int]:
    keys: list[set[int]] = []
    for model in MODELS:
        for variant in VARIANTS:
            keys.append(set(by_model_variant[model][variant].keys()))
    return sorted(set.intersection(*keys))


def _encode_arrays(
    by_model_variant: dict[str, dict[str, dict[int, dict[str, Any]]]],
    hadm_ids: list[int],
    fields: list[str],
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, dict[str, np.ndarray]]],
]:
    encoded: dict[str, dict[str, dict[str, np.ndarray]]] = {
        scheme: {model: {} for model in MODELS} for scheme in SCHEMES
    }
    yes_flags: dict[str, dict[str, dict[str, np.ndarray]]] = {
        model: {variant: {} for variant in VARIANTS} for model in MODELS
    }

    for model in MODELS:
        for variant in VARIANTS:
            parsed = by_model_variant[model][variant]
            for field in fields:
                raw = [parsed[h].get(field) for h in hadm_ids]
                yes_flags[model][variant][field] = np.asarray([_is_yes(v) for v in raw], dtype=np.int8)
                tri_vals = np.asarray([_encode_tristate(v, collapsed=False) for v in raw], dtype=np.int8)
                col_vals = np.asarray([_encode_tristate(v, collapsed=True) for v in raw], dtype=np.int8)
                encoded["tristate"][model][f"{variant}::{field}"] = tri_vals
                encoded["collapsed"][model][f"{variant}::{field}"] = col_vals
    return encoded, yes_flags


def _included_fields_fixed(
    yes_flags: dict[str, dict[str, dict[str, np.ndarray]]],
    fields: list[str],
    threshold: int,
) -> list[str]:
    included: list[str] = []
    for field in fields:
        small_pos = sum(int(np.sum(yes_flags["small"][v][field])) for v in VARIANTS)
        full_pos = sum(int(np.sum(yes_flags["full"][v][field])) for v in VARIANTS)
        if small_pos >= threshold and full_pos >= threshold:
            included.append(field)
    return included


def _compute_stats_for_sample(
    *,
    sample_idx: np.ndarray,
    encoded: dict[str, dict[str, dict[str, np.ndarray]]],
    yes_flags: dict[str, dict[str, dict[str, np.ndarray]]],
    fields: list[str],
    filter_mode: str,
    fixed_included_fields: list[str],
    threshold: int,
) -> tuple[AggregateStats, dict[str, float], dict[str, float], int]:
    if filter_mode not in {"fixed", "dynamic"}:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")

    if filter_mode == "fixed":
        included_fields = fixed_included_fields
    else:
        included_fields = []
        for field in fields:
            small_pos = sum(int(np.sum(yes_flags["small"][v][field][sample_idx])) for v in VARIANTS)
            full_pos = sum(int(np.sum(yes_flags["full"][v][field][sample_idx])) for v in VARIANTS)
            if small_pos >= threshold and full_pos >= threshold:
                included_fields.append(field)

    if not included_fields:
        nan_stats = AggregateStats(np.nan, np.nan, np.nan, np.nan)
        return nan_stats, {f: np.nan for f in fields}, {f: np.nan for f in fields}, 0

    per_field_delta_tri: dict[str, float] = {}
    per_field_delta_col: dict[str, float] = {}
    pooled_small_tri: list[float] = []
    pooled_full_tri: list[float] = []
    pooled_small_col: list[float] = []
    pooled_full_col: list[float] = []

    for field in included_fields:
        tri_pair_deltas: list[float] = []
        col_pair_deltas: list[float] = []
        for lv, rv in PAIRS:
            s_tri_l = encoded["tristate"]["small"][f"{lv}::{field}"][sample_idx]
            s_tri_r = encoded["tristate"]["small"][f"{rv}::{field}"][sample_idx]
            f_tri_l = encoded["tristate"]["full"][f"{lv}::{field}"][sample_idx]
            f_tri_r = encoded["tristate"]["full"][f"{rv}::{field}"][sample_idx]
            s_col_l = encoded["collapsed"]["small"][f"{lv}::{field}"][sample_idx]
            s_col_r = encoded["collapsed"]["small"][f"{rv}::{field}"][sample_idx]
            f_col_l = encoded["collapsed"]["full"][f"{lv}::{field}"][sample_idx]
            f_col_r = encoded["collapsed"]["full"][f"{rv}::{field}"][sample_idx]

            k_s_tri = _safe_kappa(s_tri_l, s_tri_r)
            k_f_tri = _safe_kappa(f_tri_l, f_tri_r)
            k_s_col = _safe_kappa(s_col_l, s_col_r)
            k_f_col = _safe_kappa(f_col_l, f_col_r)

            pooled_small_tri.append(k_s_tri)
            pooled_full_tri.append(k_f_tri)
            pooled_small_col.append(k_s_col)
            pooled_full_col.append(k_f_col)
            tri_pair_deltas.append((k_f_tri - k_s_tri) * 100.0)
            col_pair_deltas.append((k_f_col - k_s_col) * 100.0)

        per_field_delta_tri[field] = float(np.median(np.asarray(tri_pair_deltas, dtype=np.float64)))
        per_field_delta_col[field] = float(np.median(np.asarray(col_pair_deltas, dtype=np.float64)))

    # Populate non-included fields as NaN for consistent output tables.
    for field in fields:
        if field not in per_field_delta_tri:
            per_field_delta_tri[field] = float("nan")
        if field not in per_field_delta_col:
            per_field_delta_col[field] = float("nan")

    pooled_delta_tri = (
        float(np.median(np.asarray(pooled_full_tri, dtype=np.float64)))
        - float(np.median(np.asarray(pooled_small_tri, dtype=np.float64)))
    ) * 100.0
    pooled_delta_col = (
        float(np.median(np.asarray(pooled_full_col, dtype=np.float64)))
        - float(np.median(np.asarray(pooled_small_col, dtype=np.float64)))
    ) * 100.0

    perfield_delta_tri = float(
        np.median(np.asarray([per_field_delta_tri[f] for f in included_fields], dtype=np.float64))
    )
    perfield_delta_col = float(
        np.median(np.asarray([per_field_delta_col[f] for f in included_fields], dtype=np.float64))
    )

    agg = AggregateStats(
        pooled_delta_tristate_pp=pooled_delta_tri,
        pooled_delta_collapsed_pp=pooled_delta_col,
        perfield_delta_tristate_pp=perfield_delta_tri,
        perfield_delta_collapsed_pp=perfield_delta_col,
    )
    return agg, per_field_delta_tri, per_field_delta_col, len(included_fields)


def _markdown_table(rows: list[dict[str, str]], headers: list[str]) -> str:
    if not rows:
        return "_No rows._"
    head = "| " + " | ".join(headers) + " |"
    div = "|" + "|".join(["---"] * len(headers)) + "|"
    body = [
        "| " + " | ".join(str(row.get(h, "")).replace("|", "\\|") for h in headers) + " |"
        for row in rows
    ]
    return "\n".join([head, div, *body])


def _fmt(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def run(args: argparse.Namespace) -> None:
    fields = get_tristate_field_list("base")
    by_model_variant = _build_model_variant_maps()
    hadm_ids = _shared_hadm_ids(by_model_variant)
    n_notes = len(hadm_ids)
    if n_notes == 0:
        raise RuntimeError("No shared hadm_id across all model/variant paired runs.")

    encoded, yes_flags = _encode_arrays(by_model_variant, hadm_ids, fields)
    fixed_fields = _included_fields_fixed(yes_flags, fields, threshold=args.base_rate_threshold)

    # Primary (fixed-filter) point estimate on full sample (no resampling).
    full_idx = np.arange(n_notes, dtype=np.int64)
    point_fixed, point_field_tri, point_field_col, n_included_fixed = _compute_stats_for_sample(
        sample_idx=full_idx,
        encoded=encoded,
        yes_flags=yes_flags,
        fields=fields,
        filter_mode="fixed",
        fixed_included_fields=fixed_fields,
        threshold=args.base_rate_threshold,
    )

    rng = np.random.default_rng(args.seed)
    b = args.bootstrap_reps

    # Bootstrap arrays (fixed-filter primary analysis).
    boot_fixed_agg = np.empty((b, 4), dtype=np.float64)
    boot_fixed_tri = np.empty((b, len(fields)), dtype=np.float64)
    boot_fixed_col = np.empty((b, len(fields)), dtype=np.float64)

    # Optional sensitivity (dynamic filter handling).
    boot_dynamic_agg = np.empty((b, 4), dtype=np.float64)
    boot_dynamic_n_fields = np.empty(b, dtype=np.int64)

    field_to_idx = {f: i for i, f in enumerate(fields)}

    for i in range(b):
        sample_idx = rng.integers(0, n_notes, size=n_notes, dtype=np.int64)

        agg_f, per_tri_f, per_col_f, _ = _compute_stats_for_sample(
            sample_idx=sample_idx,
            encoded=encoded,
            yes_flags=yes_flags,
            fields=fields,
            filter_mode="fixed",
            fixed_included_fields=fixed_fields,
            threshold=args.base_rate_threshold,
        )
        boot_fixed_agg[i, :] = [
            agg_f.pooled_delta_tristate_pp,
            agg_f.pooled_delta_collapsed_pp,
            agg_f.perfield_delta_tristate_pp,
            agg_f.perfield_delta_collapsed_pp,
        ]
        for f in fields:
            j = field_to_idx[f]
            boot_fixed_tri[i, j] = per_tri_f[f]
            boot_fixed_col[i, j] = per_col_f[f]

        agg_d, _, _, n_inc_d = _compute_stats_for_sample(
            sample_idx=sample_idx,
            encoded=encoded,
            yes_flags=yes_flags,
            fields=fields,
            filter_mode="dynamic",
            fixed_included_fields=fixed_fields,
            threshold=args.base_rate_threshold,
        )
        boot_dynamic_agg[i, :] = [
            agg_d.pooled_delta_tristate_pp,
            agg_d.pooled_delta_collapsed_pp,
            agg_d.perfield_delta_tristate_pp,
            agg_d.perfield_delta_collapsed_pp,
        ]
        boot_dynamic_n_fields[i] = n_inc_d

    # CI summaries (fixed-filter primary).
    fixed_ci = {
        "pooled_delta_tristate_pp": _percentile_ci(boot_fixed_agg[:, 0]),
        "pooled_delta_collapsed_pp": _percentile_ci(boot_fixed_agg[:, 1]),
        "perfield_delta_tristate_pp": _percentile_ci(boot_fixed_agg[:, 2]),
        "perfield_delta_collapsed_pp": _percentile_ci(boot_fixed_agg[:, 3]),
    }

    # Per-field CIs.
    field_rows_tri: list[dict[str, str]] = []
    field_rows_col: list[dict[str, str]] = []
    tri_neg = 0
    tri_neg_ci_excl = 0
    tri_pos_ci_excl = 0
    tri_straddle = 0
    col_neg = 0
    col_neg_ci_excl = 0
    col_pos_ci_excl = 0
    col_straddle = 0

    for f in fields:
        j = field_to_idx[f]
        point_t = point_field_tri[f]
        lo_t, hi_t = _percentile_ci(boot_fixed_tri[:, j])
        excl_zero_t = np.isfinite(lo_t) and np.isfinite(hi_t) and ((lo_t > 0) or (hi_t < 0))
        sign_t = "negative" if point_t < 0 else "positive" if point_t > 0 else "zero"
        if sign_t == "negative":
            tri_neg += 1
            if hi_t < 0:
                tri_neg_ci_excl += 1
        elif sign_t == "positive" and lo_t > 0:
            tri_pos_ci_excl += 1
        if lo_t <= 0 <= hi_t:
            tri_straddle += 1
        field_rows_tri.append(
            {
                "field": f,
                "point_estimate_pp": _fmt(point_t, 2),
                "ci95_low_pp": _fmt(lo_t, 2),
                "ci95_high_pp": _fmt(hi_t, 2),
                "ci_excludes_zero": "yes" if excl_zero_t else "no",
            }
        )

        point_c = point_field_col[f]
        lo_c, hi_c = _percentile_ci(boot_fixed_col[:, j])
        excl_zero_c = np.isfinite(lo_c) and np.isfinite(hi_c) and ((lo_c > 0) or (hi_c < 0))
        sign_c = "negative" if point_c < 0 else "positive" if point_c > 0 else "zero"
        if sign_c == "negative":
            col_neg += 1
            if hi_c < 0:
                col_neg_ci_excl += 1
        elif sign_c == "positive" and lo_c > 0:
            col_pos_ci_excl += 1
        if lo_c <= 0 <= hi_c:
            col_straddle += 1
        field_rows_col.append(
            {
                "field": f,
                "point_estimate_pp": _fmt(point_c, 2),
                "ci95_low_pp": _fmt(lo_c, 2),
                "ci95_high_pp": _fmt(hi_c, 2),
                "ci_excludes_zero": "yes" if excl_zero_c else "no",
            }
        )

    # Sensitivity to filter handling.
    dynamic_ci = {
        "pooled_delta_tristate_pp": _percentile_ci(boot_dynamic_agg[:, 0]),
        "pooled_delta_collapsed_pp": _percentile_ci(boot_dynamic_agg[:, 1]),
        "perfield_delta_tristate_pp": _percentile_ci(boot_dynamic_agg[:, 2]),
        "perfield_delta_collapsed_pp": _percentile_ci(boot_dynamic_agg[:, 3]),
    }
    point_dynamic, _, _, n_included_dynamic_point = _compute_stats_for_sample(
        sample_idx=full_idx,
        encoded=encoded,
        yes_flags=yes_flags,
        fields=fields,
        filter_mode="dynamic",
        fixed_included_fields=fixed_fields,
        threshold=args.base_rate_threshold,
    )

    # Claim-value checks.
    claims = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    claim_checks = {
        "model_size_pooled_delta_tristate": float(claims["model_size_pooled_delta_tristate"]["value"]),
        "model_size_pooled_delta_collapsed": float(claims["model_size_pooled_delta_collapsed"]["value"]),
        "model_size_perfield_delta_tristate": float(claims["model_size_perfield_delta_tristate"]["value"]),
        "model_size_perfield_delta_collapsed": float(claims["model_size_perfield_delta_collapsed"]["value"]),
    }
    point_vals = {
        "model_size_pooled_delta_tristate": point_fixed.pooled_delta_tristate_pp,
        "model_size_pooled_delta_collapsed": point_fixed.pooled_delta_collapsed_pp,
        "model_size_perfield_delta_tristate": point_fixed.perfield_delta_tristate_pp,
        "model_size_perfield_delta_collapsed": point_fixed.perfield_delta_collapsed_pp,
    }

    agg_rows = [
        {
            "quantity": r"$\Delta\bar{\kappa}^{\mathrm{TriState}}$ (pooled delta, pp)",
            "point_estimate": _fmt(point_fixed.pooled_delta_tristate_pp, 2),
            "ci95_low": _fmt(fixed_ci["pooled_delta_tristate_pp"][0], 2),
            "ci95_high": _fmt(fixed_ci["pooled_delta_tristate_pp"][1], 2),
        },
        {
            "quantity": r"$\Delta\bar{\kappa}^{\mathrm{collapsed}}$ (pooled delta, pp)",
            "point_estimate": _fmt(point_fixed.pooled_delta_collapsed_pp, 2),
            "ci95_low": _fmt(fixed_ci["pooled_delta_collapsed_pp"][0], 2),
            "ci95_high": _fmt(fixed_ci["pooled_delta_collapsed_pp"][1], 2),
        },
        {
            "quantity": r"$\overline{\Delta\kappa}^{\,\mathrm{per\text{-}field}}$ (TriState, pp)",
            "point_estimate": _fmt(point_fixed.perfield_delta_tristate_pp, 2),
            "ci95_low": _fmt(fixed_ci["perfield_delta_tristate_pp"][0], 2),
            "ci95_high": _fmt(fixed_ci["perfield_delta_tristate_pp"][1], 2),
        },
        {
            "quantity": r"$\overline{\Delta\kappa}^{\,\mathrm{per\text{-}field}}$ (collapsed, pp)",
            "point_estimate": _fmt(point_fixed.perfield_delta_collapsed_pp, 2),
            "ci95_low": _fmt(fixed_ci["perfield_delta_collapsed_pp"][0], 2),
            "ci95_high": _fmt(fixed_ci["perfield_delta_collapsed_pp"][1], 2),
        },
    ]

    # Robustness of sign disagreement.
    pooled_ci_crosses_zero = fixed_ci["pooled_delta_tristate_pp"][0] <= 0 <= fixed_ci["pooled_delta_tristate_pp"][1]
    perfield_ci_crosses_zero = (
        fixed_ci["perfield_delta_tristate_pp"][0] <= 0 <= fixed_ci["perfield_delta_tristate_pp"][1]
    )

    sensitivity_rows = [
        {
            "quantity": "pooled_delta_tristate_pp",
            "fixed_point": _fmt(point_fixed.pooled_delta_tristate_pp, 2),
            "fixed_ci95": f"[{_fmt(fixed_ci['pooled_delta_tristate_pp'][0],2)}, {_fmt(fixed_ci['pooled_delta_tristate_pp'][1],2)}]",
            "dynamic_point": _fmt(point_dynamic.pooled_delta_tristate_pp, 2),
            "dynamic_ci95": f"[{_fmt(dynamic_ci['pooled_delta_tristate_pp'][0],2)}, {_fmt(dynamic_ci['pooled_delta_tristate_pp'][1],2)}]",
        },
        {
            "quantity": "pooled_delta_collapsed_pp",
            "fixed_point": _fmt(point_fixed.pooled_delta_collapsed_pp, 2),
            "fixed_ci95": f"[{_fmt(fixed_ci['pooled_delta_collapsed_pp'][0],2)}, {_fmt(fixed_ci['pooled_delta_collapsed_pp'][1],2)}]",
            "dynamic_point": _fmt(point_dynamic.pooled_delta_collapsed_pp, 2),
            "dynamic_ci95": f"[{_fmt(dynamic_ci['pooled_delta_collapsed_pp'][0],2)}, {_fmt(dynamic_ci['pooled_delta_collapsed_pp'][1],2)}]",
        },
        {
            "quantity": "perfield_delta_tristate_pp",
            "fixed_point": _fmt(point_fixed.perfield_delta_tristate_pp, 2),
            "fixed_ci95": f"[{_fmt(fixed_ci['perfield_delta_tristate_pp'][0],2)}, {_fmt(fixed_ci['perfield_delta_tristate_pp'][1],2)}]",
            "dynamic_point": _fmt(point_dynamic.perfield_delta_tristate_pp, 2),
            "dynamic_ci95": f"[{_fmt(dynamic_ci['perfield_delta_tristate_pp'][0],2)}, {_fmt(dynamic_ci['perfield_delta_tristate_pp'][1],2)}]",
        },
        {
            "quantity": "perfield_delta_collapsed_pp",
            "fixed_point": _fmt(point_fixed.perfield_delta_collapsed_pp, 2),
            "fixed_ci95": f"[{_fmt(fixed_ci['perfield_delta_collapsed_pp'][0],2)}, {_fmt(fixed_ci['perfield_delta_collapsed_pp'][1],2)}]",
            "dynamic_point": _fmt(point_dynamic.perfield_delta_collapsed_pp, 2),
            "dynamic_ci95": f"[{_fmt(dynamic_ci['perfield_delta_collapsed_pp'][0],2)}, {_fmt(dynamic_ci['perfield_delta_collapsed_pp'][1],2)}]",
        },
    ]

    claim_rows = []
    for key in [
        "model_size_pooled_delta_tristate",
        "model_size_pooled_delta_collapsed",
        "model_size_perfield_delta_tristate",
        "model_size_perfield_delta_collapsed",
    ]:
        claim_rows.append(
            {
                "claim_key": key,
                "claim_value_pp": _fmt(claim_checks[key], 2),
                "bootstrap_point_pp": _fmt(point_vals[key], 2),
                "difference_pp": _fmt(point_vals[key] - claim_checks[key], 2),
            }
        )

    lines = [
        "# Paired Bootstrap CIs for Model-Size Effect (Computation Only)",
        "",
        f"- generated_utc: {datetime.now(UTC).isoformat()}",
        f"- paired_sample_n_notes: {n_notes}",
        f"- tristate_field_count: {len(fields)}",
        f"- bootstrap_replicates: {b}",
        f"- bootstrap_seed: {args.seed}",
        f"- base_rate_threshold_positive_votes: {args.base_rate_threshold}",
        "",
        "## Resampling Design",
        "- Unit of resampling: note (hadm_id), sampled with replacement.",
        "- Pairing preserved: each bootstrap sample uses the same resampled note index set for small and full model outputs.",
        "- Primary filter handling: fixed field set based on full-sample positive-count threshold in **both** model sizes.",
        f"- Fixed included field count: {n_included_fixed} / {len(fields)}",
        f"- Dynamic-filter sensitivity full-sample included field count: {n_included_dynamic_point} / {len(fields)}",
        "- CI method: percentile (2.5th, 97.5th).",
        "",
        "## Aggregate Quantities (Primary: Fixed Filter)",
        _markdown_table(agg_rows, ["quantity", "point_estimate", "ci95_low", "ci95_high"]),
        "",
        "## Per-field $\\Delta\\kappa_f$ (TriState, pp)",
        _markdown_table(
            field_rows_tri,
            ["field", "point_estimate_pp", "ci95_low_pp", "ci95_high_pp", "ci_excludes_zero"],
        ),
        "",
        "## Per-field $\\Delta\\kappa_f$ (Collapsed, pp)",
        _markdown_table(
            field_rows_col,
            ["field", "point_estimate_pp", "ci95_low_pp", "ci95_high_pp", "ci_excludes_zero"],
        ),
        "",
        "## Direct Answers to Requested Questions",
        (
            f"1. TriState per-field $\\Delta\\kappa_f$: negative point estimates = {tri_neg}; "
            f"reliably negative (95% CI < 0) = {tri_neg_ci_excl}; "
            f"positive with 95% CI > 0 = {tri_pos_ci_excl}; "
            f"CI straddling zero = {tri_straddle}."
        ),
        (
            f"2. Collapsed per-field $\\Delta\\kappa_f$: negative point estimates = {col_neg}; "
            f"reliably negative (95% CI < 0) = {col_neg_ci_excl}; "
            f"positive with 95% CI > 0 = {col_pos_ci_excl}; "
            f"CI straddling zero = {col_straddle}."
        ),
        (
            "3. Does 95% CI on $\\overline{\\Delta\\kappa}^{\\,\\mathrm{per\\text{-}field}}$ "
            f"exclude zero? TriState: {'yes' if not perfield_ci_crosses_zero else 'no'}; "
            f"Collapsed: {'yes' if not (fixed_ci['perfield_delta_collapsed_pp'][0] <= 0 <= fixed_ci['perfield_delta_collapsed_pp'][1]) else 'no'}."
        ),
        (
            "4. Does 95% CI on $\\Delta\\bar{\\kappa}^{\\mathrm{TriState}}$ exclude zero? "
            f"{'yes' if not pooled_ci_crosses_zero else 'no'}."
        ),
        (
            "5. Is the sign disagreement robust (pooled positive vs per-field negative)? "
            f"Pooled TriState CI crosses zero: {'yes' if pooled_ci_crosses_zero else 'no'}; "
            f"Per-field TriState CI crosses zero: {'yes' if perfield_ci_crosses_zero else 'no'}."
        ),
        "",
        "## Filter-handling Sensitivity (Fixed vs Dynamic)",
        _markdown_table(
            sensitivity_rows,
            ["quantity", "fixed_point", "fixed_ci95", "dynamic_point", "dynamic_ci95"],
        ),
        (
            f"- Dynamic-filter bootstrap included-field count: median "
            f"{_fmt(float(np.median(boot_dynamic_n_fields)), 1)}, "
            f"range [{int(np.min(boot_dynamic_n_fields))}, {int(np.max(boot_dynamic_n_fields))}]."
        ),
        "",
        "## Claim Reproduction Check",
        _markdown_table(claim_rows, ["claim_key", "claim_value_pp", "bootstrap_point_pp", "difference_pp"]),
        "",
        "Interpretation note: discrepancies here indicate that registered claim values and raw paired-resample recomputation are using different aggregation/data paths.",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap CIs for model-size effects.")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--base-rate-threshold", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
