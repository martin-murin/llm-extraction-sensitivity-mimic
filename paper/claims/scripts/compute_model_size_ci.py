from __future__ import annotations

# Release documentation:
# Computes claim-registry values for model size ci.
#
# Reads: codex_outputs/paired_bootstrap_ci.md, data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl.
# Writes: codex_outputs/paired_bootstrap_ci.md, data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl.
# Backs paper claim registry entries for model size ci.

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

import numpy as np

from paper.claims.scripts._common import claim_entry, require_input_files
from paper.claims.scripts._model_size_primary import (
    PAIR_KEYS,
    _encode_tristate,
    _safe_kappa,
    load_paired_model_size_data,
    per_field_model_size_deltas,
    pooled_kappa_levels,
)
from paper.claims.scripts._receipt import build_receipt, merge_into_claims_json, now_utc_iso


CLAIMS_PATH = Path(__file__).resolve().parent.parent / "claims.json"
SCRIPT_PATH = Path(__file__).resolve()
REPO = Path(__file__).resolve().parents[3]
DIFF_REPORT_PATH = REPO / "codex_outputs" / "model_size_ci_diff_vs_paired_bootstrap_ci.md"
REFERENCE_BOOTSTRAP_REPORT = REPO / "codex_outputs" / "paired_bootstrap_ci.md"

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260523
BASE_RATE_THRESHOLD = 10
FILTER_MODE = "fixed_dual_model_yes_threshold"

INPUT_FILES = [
    "data/raw_responses/methodology_1k_a/results.jsonl",
    "data/raw_responses/methodology_1k_b/results.jsonl",
    "data/raw_responses/methodology_1k_c/results.jsonl",
    "data/raw_responses/methodology_5k_a_subset500/results.jsonl",
    "data/raw_responses/methodology_5k_audit_b/results.jsonl",
    "data/raw_responses/methodology_5k_audit_c/results.jsonl",
    "data/raw_responses/paired_gold_methodology_1k_a/results.jsonl",
    "data/raw_responses/paired_gold_methodology_1k_b/results.jsonl",
    "data/raw_responses/paired_gold_methodology_1k_c/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_a/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_b/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_c/results.jsonl",
    "src/schema/fields.py",
]


@dataclass(frozen=True)
class AggregatePoint:
    pooled_delta_tristate_pp: float
    pooled_delta_collapsed_pp: float
    perfield_delta_tristate_pp: float
    perfield_delta_collapsed_pp: float


def _percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(clean, alpha / 2.0)),
        float(np.quantile(clean, 1.0 - alpha / 2.0)),
    )


def _build_encoded_arrays(
    base_rate_threshold: int,
) -> tuple[
    list[str],
    list[str],
    dict[str, dict[str, dict[str, np.ndarray]]],
]:
    """Return (all_fields, included_fields, encoded[scheme][model_variant][field])."""
    data = load_paired_model_size_data(base_rate_threshold=base_rate_threshold)
    all_fields = list(data.tri_fields)
    included_fields = list(data.included_fields)
    common = data.common_hadm_ids

    encoded: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "tristate": {},
        "collapsed": {},
    }
    for scheme in ("tristate", "collapsed"):
        collapsed = scheme == "collapsed"
        for model_name, by_variant in (("small", data.small_by_variant), ("full", data.full_by_variant)):
            mv_key = model_name
            encoded[scheme][mv_key] = {}
            for variant in ("A", "B", "C"):
                for field in all_fields:
                    raw = by_variant[variant].loc[common, field].tolist()
                    arr = np.asarray(
                        [_encode_tristate(v, collapse=collapsed) for v in raw],
                        dtype=np.int8,
                    )
                    encoded[scheme][mv_key][f"{variant}::{field}"] = arr
    return all_fields, included_fields, encoded


def _bootstrap(
    *,
    all_fields: list[str],
    included_fields: list[str],
    encoded: dict[str, dict[str, dict[str, np.ndarray]]],
    reps: int,
    seed: int,
) -> tuple[
    AggregatePoint,
    dict[str, float],
    dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Run paired note-level bootstrap and return aggregate/per-field distributions."""
    # Point estimates from canonical full sample.
    data = load_paired_model_size_data(base_rate_threshold=BASE_RATE_THRESHOLD)
    full_delta_pp, collapsed_delta_pp = per_field_model_size_deltas(data)
    p_small_tri, p_full_tri, p_small_col, p_full_col = pooled_kappa_levels(data)
    point = AggregatePoint(
        pooled_delta_tristate_pp=(p_full_tri - p_small_tri) * 100.0,
        pooled_delta_collapsed_pp=(p_full_col - p_small_col) * 100.0,
        perfield_delta_tristate_pp=float(
            np.median(np.asarray([full_delta_pp[f] for f in included_fields], dtype=np.float64))
        ),
        perfield_delta_collapsed_pp=float(
            np.median(np.asarray([collapsed_delta_pp[f] for f in included_fields], dtype=np.float64))
        ),
    )

    n = next(iter(encoded["tristate"]["small"].values())).size
    rng = np.random.default_rng(seed)

    pooled_delta_tri = np.empty(reps, dtype=np.float64)
    pooled_delta_col = np.empty(reps, dtype=np.float64)
    perfield_delta_tri = np.empty(reps, dtype=np.float64)
    perfield_delta_col = np.empty(reps, dtype=np.float64)
    per_field_tri: dict[str, np.ndarray] = {field: np.empty(reps, dtype=np.float64) for field in all_fields}
    per_field_col: dict[str, np.ndarray] = {field: np.empty(reps, dtype=np.float64) for field in all_fields}

    for b in range(reps):
        idx = rng.integers(0, n, size=n, endpoint=False)

        small_tri_cells: list[float] = []
        full_tri_cells: list[float] = []
        small_col_cells: list[float] = []
        full_col_cells: list[float] = []
        field_tri_vals: dict[str, float] = {}
        field_col_vals: dict[str, float] = {}

        for field in all_fields:
            tri_pair_deltas: list[float] = []
            col_pair_deltas: list[float] = []
            for left, right in PAIR_KEYS:
                s_tri_l = encoded["tristate"]["small"][f"{left}::{field}"][idx]
                s_tri_r = encoded["tristate"]["small"][f"{right}::{field}"][idx]
                f_tri_l = encoded["tristate"]["full"][f"{left}::{field}"][idx]
                f_tri_r = encoded["tristate"]["full"][f"{right}::{field}"][idx]
                s_col_l = encoded["collapsed"]["small"][f"{left}::{field}"][idx]
                s_col_r = encoded["collapsed"]["small"][f"{right}::{field}"][idx]
                f_col_l = encoded["collapsed"]["full"][f"{left}::{field}"][idx]
                f_col_r = encoded["collapsed"]["full"][f"{right}::{field}"][idx]

                k_s_tri = _safe_kappa(s_tri_l, s_tri_r)
                k_f_tri = _safe_kappa(f_tri_l, f_tri_r)
                k_s_col = _safe_kappa(s_col_l, s_col_r)
                k_f_col = _safe_kappa(f_col_l, f_col_r)

                tri_pair_deltas.append((k_f_tri - k_s_tri) * 100.0)
                col_pair_deltas.append((k_f_col - k_s_col) * 100.0)

                if field in included_fields:
                    small_tri_cells.append(k_s_tri)
                    full_tri_cells.append(k_f_tri)
                    small_col_cells.append(k_s_col)
                    full_col_cells.append(k_f_col)

            field_tri_vals[field] = float(np.median(np.asarray(tri_pair_deltas, dtype=np.float64)))
            field_col_vals[field] = float(np.median(np.asarray(col_pair_deltas, dtype=np.float64)))
            per_field_tri[field][b] = field_tri_vals[field]
            per_field_col[field][b] = field_col_vals[field]

        pooled_small_tri = float(np.median(np.asarray(small_tri_cells, dtype=np.float64)))
        pooled_full_tri = float(np.median(np.asarray(full_tri_cells, dtype=np.float64)))
        pooled_small_col = float(np.median(np.asarray(small_col_cells, dtype=np.float64)))
        pooled_full_col = float(np.median(np.asarray(full_col_cells, dtype=np.float64)))

        pooled_delta_tri[b] = (pooled_full_tri - pooled_small_tri) * 100.0
        pooled_delta_col[b] = (pooled_full_col - pooled_small_col) * 100.0
        perfield_delta_tri[b] = float(
            np.median(np.asarray([field_tri_vals[f] for f in included_fields], dtype=np.float64))
        )
        perfield_delta_col[b] = float(
            np.median(np.asarray([field_col_vals[f] for f in included_fields], dtype=np.float64))
        )

    return (
        point,
        full_delta_pp,
        collapsed_delta_pp,
        pooled_delta_tri,
        pooled_delta_col,
        perfield_delta_tri,
        perfield_delta_col,
        per_field_tri,
        per_field_col,
    )


def _parse_markdown_table(text: str, heading: str) -> list[dict[str, str]]:
    """Parse first markdown table under a given H2 heading."""
    lines = text.splitlines()
    start = None
    target = f"## {heading}".strip()
    for i, line in enumerate(lines):
        if line.strip() == target:
            start = i
            break
    if start is None:
        return []

    table_start = None
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("|"):
            table_start = j
            break
    if table_start is None or table_start + 1 >= len(lines):
        return []

    header = [c.strip() for c in lines[table_start].strip().strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for k in range(table_start + 2, len(lines)):
        line = lines[k]
        if not line.startswith("|"):
            break
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts, strict=True)))
    return rows


def _to_float_maybe(value: str) -> float | None:
    raw = value.strip()
    if raw.upper() == "NA":
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    if not m:
        return None
    return float(m.group(0))


def _write_diff_report(
    *,
    point: AggregatePoint,
    pooled_delta_tri: np.ndarray,
    pooled_delta_col: np.ndarray,
    perfield_delta_tri: np.ndarray,
    perfield_delta_col: np.ndarray,
    full_delta_pp: dict[str, float],
    collapsed_delta_pp: dict[str, float],
    per_field_tri: dict[str, np.ndarray],
    per_field_col: dict[str, np.ndarray],
    all_fields: list[str],
) -> None:
    agg_current = {
        "pooled_delta_tristate_pp": (
            point.pooled_delta_tristate_pp,
            *_percentile_ci(pooled_delta_tri),
        ),
        "pooled_delta_collapsed_pp": (
            point.pooled_delta_collapsed_pp,
            *_percentile_ci(pooled_delta_col),
        ),
        "perfield_delta_tristate_pp": (
            point.perfield_delta_tristate_pp,
            *_percentile_ci(perfield_delta_tri),
        ),
        "perfield_delta_collapsed_pp": (
            point.perfield_delta_collapsed_pp,
            *_percentile_ci(perfield_delta_col),
        ),
    }

    ref_text = REFERENCE_BOOTSTRAP_REPORT.read_text(encoding="utf-8") if REFERENCE_BOOTSTRAP_REPORT.exists() else ""
    ref_agg_rows = _parse_markdown_table(ref_text, "Aggregate Quantities (Primary: Fixed Filter)")
    ref_tri_rows = _parse_markdown_table(ref_text, "Per-field $\\Delta\\kappa_f$ (TriState, pp)")
    ref_col_rows = _parse_markdown_table(ref_text, "Per-field $\\Delta\\kappa_f$ (Collapsed, pp)")

    ref_agg: dict[str, tuple[float | None, float | None, float | None]] = {}
    for row in ref_agg_rows:
        q = row.get("quantity", "")
        pe = _to_float_maybe(row.get("point_estimate", ""))
        lo = _to_float_maybe(row.get("ci95_low", ""))
        hi = _to_float_maybe(row.get("ci95_high", ""))
        if "TriState" in q and "pooled" in q:
            ref_agg["pooled_delta_tristate_pp"] = (pe, lo, hi)
        elif "collapsed" in q and "pooled" in q:
            ref_agg["pooled_delta_collapsed_pp"] = (pe, lo, hi)
        elif "per\\text{-}field" in q and "TriState" in q:
            ref_agg["perfield_delta_tristate_pp"] = (pe, lo, hi)
        elif "per\\text{-}field" in q and "collapsed" in q:
            ref_agg["perfield_delta_collapsed_pp"] = (pe, lo, hi)

    ref_tri: dict[str, tuple[float | None, float | None, float | None]] = {}
    for row in ref_tri_rows:
        field = row.get("field", "").strip()
        ref_tri[field] = (
            _to_float_maybe(row.get("point_estimate_pp", "")),
            _to_float_maybe(row.get("ci95_low_pp", "")),
            _to_float_maybe(row.get("ci95_high_pp", "")),
        )

    ref_col: dict[str, tuple[float | None, float | None, float | None]] = {}
    for row in ref_col_rows:
        field = row.get("field", "").strip()
        ref_col[field] = (
            _to_float_maybe(row.get("point_estimate_pp", "")),
            _to_float_maybe(row.get("ci95_low_pp", "")),
            _to_float_maybe(row.get("ci95_high_pp", "")),
        )

    lines: list[str] = []
    lines.append("# Model-Size CI Reproduction and Diff")
    lines.append("")
    lines.append(f"- seed: `{BOOTSTRAP_SEED}`")
    lines.append(f"- bootstrap_replicates: `{BOOTSTRAP_REPS}`")
    lines.append(f"- base_rate_threshold: `{BASE_RATE_THRESHOLD}`")
    lines.append(f"- filter_mode: `{FILTER_MODE}`")
    lines.append("")
    lines.append("## Aggregate Diff vs `paired_bootstrap_ci.md`")
    lines.append("")
    lines.append("| metric | current_point | current_ci95 | reference_point | reference_ci95 | diff_point |")
    lines.append("|---|---:|---|---:|---|---:|")
    for key, (cur_p, cur_lo, cur_hi) in agg_current.items():
        rp, rlo, rhi = ref_agg.get(key, (None, None, None))
        diff = (cur_p - rp) if rp is not None else float("nan")
        lines.append(
            "| {k} | {cp:.2f} | [{cl:.2f}, {ch:.2f}] | {rpv} | {rc} | {dpv} |".format(
                k=key,
                cp=cur_p,
                cl=cur_lo,
                ch=cur_hi,
                rpv="NA" if rp is None else f"{rp:.2f}",
                rc="NA" if rlo is None or rhi is None else f"[{rlo:.2f}, {rhi:.2f}]",
                dpv="NA" if not np.isfinite(diff) else f"{diff:.2f}",
            )
        )

    lines.append("")
    lines.append("## Per-field TriState Diff vs `paired_bootstrap_ci.md`")
    lines.append("")
    lines.append("| field | current_point | current_ci95 | reference_point | reference_ci95 |")
    lines.append("|---|---:|---|---:|---|")
    for field in all_fields:
        cur_p = full_delta_pp[field]
        cur_lo, cur_hi = _percentile_ci(per_field_tri[field])
        rp, rlo, rhi = ref_tri.get(field, (None, None, None))
        lines.append(
            "| `{f}` | {cp:.2f} | [{cl:.2f}, {ch:.2f}] | {rpv} | {rc} |".format(
                f=field,
                cp=cur_p,
                cl=cur_lo,
                ch=cur_hi,
                rpv="NA" if rp is None else f"{rp:.2f}",
                rc="NA" if rlo is None or rhi is None else f"[{rlo:.2f}, {rhi:.2f}]",
            )
        )

    lines.append("")
    lines.append("## Per-field Collapsed Diff vs `paired_bootstrap_ci.md`")
    lines.append("")
    lines.append("| field | current_point | current_ci95 | reference_point | reference_ci95 |")
    lines.append("|---|---:|---|---:|---|")
    for field in all_fields:
        cur_p = collapsed_delta_pp[field]
        cur_lo, cur_hi = _percentile_ci(per_field_col[field])
        rp, rlo, rhi = ref_col.get(field, (None, None, None))
        lines.append(
            "| `{f}` | {cp:.2f} | [{cl:.2f}, {ch:.2f}] | {rpv} | {rc} |".format(
                f=field,
                cp=cur_p,
                cl=cur_lo,
                ch=cur_hi,
                rpv="NA" if rp is None else f"{rp:.2f}",
                rc="NA" if rlo is None or rhi is None else f"[{rlo:.2f}, {rhi:.2f}]",
            )
        )

    lines.append("")
    lines.append(
        "Note: this script computes per-field CIs for all 17 TriState fields to align with "
        "Figure 01 bars; aggregate medians remain on the canonical filtered field set."
    )
    DIFF_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def compute_all() -> dict:
    require_input_files(INPUT_FILES)

    all_fields, included_fields, encoded = _build_encoded_arrays(BASE_RATE_THRESHOLD)
    (
        point,
        full_delta_pp,
        collapsed_delta_pp,
        pooled_delta_tri,
        pooled_delta_col,
        perfield_delta_tri,
        perfield_delta_col,
        per_field_tri,
        per_field_col,
    ) = _bootstrap(
        all_fields=all_fields,
        included_fields=included_fields,
        encoded=encoded,
        reps=BOOTSTRAP_REPS,
        seed=BOOTSTRAP_SEED,
    )

    timestamp = now_utc_iso()
    receipt = build_receipt(SCRIPT_PATH, "compute_all", INPUT_FILES)
    receipt["bootstrap_seed"] = BOOTSTRAP_SEED
    receipt["bootstrap_replicates"] = BOOTSTRAP_REPS
    receipt["bootstrap_filter_mode"] = FILTER_MODE
    receipt["base_rate_threshold"] = BASE_RATE_THRESHOLD
    receipt["included_field_count"] = len(included_fields)
    receipt["included_fields"] = list(included_fields)

    claims: dict[str, dict[str, Any]] = {}

    # Aggregate CI claims.
    agg_ci = {
        "model_size_pooled_delta_tristate": _percentile_ci(pooled_delta_tri),
        "model_size_pooled_delta_collapsed": _percentile_ci(pooled_delta_col),
        "model_size_perfield_delta_tristate": _percentile_ci(perfield_delta_tri),
        "model_size_perfield_delta_collapsed": _percentile_ci(perfield_delta_col),
    }
    for base_key, (ci_low, ci_high) in agg_ci.items():
        claims[f"{base_key}_ci_low_pp"] = claim_entry(
            value=ci_low,
            format_default=".2f",
            unit="pp",
            description=f"Bootstrap 95% CI lower bound for {base_key}",
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        )
        claims[f"{base_key}_ci_high_pp"] = claim_entry(
            value=ci_high,
            format_default=".2f",
            unit="pp",
            description=f"Bootstrap 95% CI upper bound for {base_key}",
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        )

    # Per-field CI claims for all 17 TriState fields.
    for field in all_fields:
        tri_lo, tri_hi = _percentile_ci(per_field_tri[field])
        col_lo, col_hi = _percentile_ci(per_field_col[field])
        claims[f"{field}_delta_full_ci_low_pp"] = claim_entry(
            value=tri_lo,
            format_default=".2f",
            unit="pp",
            description=f"Bootstrap 95% CI lower bound for {field} TriState model-size delta",
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        )
        claims[f"{field}_delta_full_ci_high_pp"] = claim_entry(
            value=tri_hi,
            format_default=".2f",
            unit="pp",
            description=f"Bootstrap 95% CI upper bound for {field} TriState model-size delta",
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        )
        claims[f"{field}_delta_collapsed_ci_low_pp"] = claim_entry(
            value=col_lo,
            format_default=".2f",
            unit="pp",
            description=f"Bootstrap 95% CI lower bound for {field} collapsed model-size delta",
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        )
        claims[f"{field}_delta_collapsed_ci_high_pp"] = claim_entry(
            value=col_hi,
            format_default=".2f",
            unit="pp",
            description=f"Bootstrap 95% CI upper bound for {field} collapsed model-size delta",
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        )

    _write_diff_report(
        point=point,
        pooled_delta_tri=pooled_delta_tri,
        pooled_delta_col=pooled_delta_col,
        perfield_delta_tri=perfield_delta_tri,
        perfield_delta_col=perfield_delta_col,
        full_delta_pp=full_delta_pp,
        collapsed_delta_pp=collapsed_delta_pp,
        per_field_tri=per_field_tri,
        per_field_col=per_field_col,
        all_fields=all_fields,
    )

    return claims


def main() -> int:
    new_claims = compute_all()
    n = merge_into_claims_json(CLAIMS_PATH, new_claims)
    print(f"Updated {n} claims in {CLAIMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
