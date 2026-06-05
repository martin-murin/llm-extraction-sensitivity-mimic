"""
Runs staged pipeline step `91_paired_collapse_analysis.py`.

Reads: codex_outputs/91_paired_collapse_per_field.md, codex_outputs/91_paired_collapse_accuracy.md, codex_outputs/91_reasoning_collapse.md.
Writes: codex_outputs/91_paired_collapse_per_field.md, codex_outputs/91_paired_collapse_accuracy.md, codex_outputs/91_reasoning_collapse.md, docs/figures/91_collapse_vs_full_per_field.png, docs/figures/91_per_variant_accuracy_collapse_vs_full.png.
Backs Figure 1 and collapse-vs-full agreement claims.
Usage: `python scripts/91_paired_collapse_analysis.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from typing import Any, Literal, get_args, get_origin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from src import config
from src.schema.fields import LLMNoteFeatures
from src.schema.vocabulary import ADMISSION_REASON_TAGS
from src.utils.threeway_kappa import low_base_rate_flag

SKIP_FIELDS: set[str] = {
    "reasoning",
    "primary_diagnosis_text",
    "new_meds_started_count",
    "meds_stopped_count",
}
ENUM_FIELDS: set[str] = {"functional_status", "mental_status", "discharge_condition_category"}
TRISTATE_DOMAIN: set[str] = {"yes", "no", "not_documented"}


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in rows:
        vals = [str(row.get(col, "")).replace("|", "\\|").replace("\n", " ") for col in columns]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider, *body])


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


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


def _literal_values(annotation: Any) -> set[str] | None:
    origin = get_origin(annotation)
    if origin is Literal:
        vals = {arg for arg in get_args(annotation) if isinstance(arg, str)}
        return set(vals)
    if origin in {None, list, dict, tuple, set}:
        return None

    args = get_args(annotation)
    if args:
        union_vals: set[str] = set()
        saw_literal = False
        for arg in args:
            if arg is type(None):
                continue
            sub_vals = _literal_values(arg)
            if sub_vals is None:
                return None
            union_vals |= sub_vals
            saw_literal = True
        return union_vals if saw_literal else None
    return None


def _detect_tristate_fields() -> list[str]:
    tristate_fields: list[str] = []
    for field_name, field_info in LLMNoteFeatures.model_fields.items():
        if field_name in SKIP_FIELDS:
            continue
        literal_values = _literal_values(field_info.annotation)
        if literal_values == TRISTATE_DOMAIN:
            tristate_fields.append(field_name)
    return sorted(tristate_fields)


def _field_names() -> list[str]:
    return [f for f in LLMNoteFeatures.model_fields if f not in SKIP_FIELDS]


def _ids_from_split(path: Path) -> set[int]:
    frame = pd.read_csv(path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").dropna().astype("int64").tolist())


def _load_results(run_id: str) -> dict[int, dict[str, Any]]:
    path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    parsed: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if bool(row.get("parse_ok", False)) and isinstance(row.get("features_json"), dict):
                parsed[int(row["hadm_id"])] = row["features_json"]
    return parsed


def _merge_variant_runs(
    first: dict[int, dict[str, Any]], second: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    overlap = set(first) & set(second)
    if overlap:
        raise RuntimeError(f"Unexpected hadm_id overlap while merging split runs: n={len(overlap)}")
    merged = dict(first)
    merged.update(second)
    return merged


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


def _agreement_percent(left: list[int], right: list[int]) -> float:
    if not left:
        return 0.0
    agree = sum(1 for lv, rv in zip(left, right, strict=False) if lv == rv)
    return float(agree / len(left) * 100.0)


def _encode_non_tristate_field(
    values: list[Any], reference: list[Any]
) -> tuple[list[int], list[int]]:
    left_vals = [str(v) for v in values]
    right_vals = [str(v) for v in reference]
    cats = sorted(set(left_vals) | set(right_vals))
    mapping = {cat: idx for idx, cat in enumerate(cats)}
    return [mapping[v] for v in left_vals], [mapping[v] for v in right_vals]


def _cross_variant_kappas(
    *,
    parsed_a: dict[int, dict[str, Any]],
    parsed_b: dict[int, dict[str, Any]],
    parsed_c: dict[int, dict[str, Any]],
    fields: list[str],
    tristate_fields: set[str],
    collapse_tristate: bool,
) -> list[dict[str, Any]]:
    overlap = sorted(set(parsed_a) & set(parsed_b) & set(parsed_c))
    if not overlap:
        return []

    rows: list[dict[str, Any]] = []
    for field in fields:
        if field == "admission_reason_tags":
            for tag in ADMISSION_REASON_TAGS:
                va = [
                    1 if tag in list(parsed_a[h].get("admission_reason_tags", [])) else 0
                    for h in overlap
                ]
                vb = [
                    1 if tag in list(parsed_b[h].get("admission_reason_tags", [])) else 0
                    for h in overlap
                ]
                vc = [
                    1 if tag in list(parsed_c[h].get("admission_reason_tags", [])) else 0
                    for h in overlap
                ]
                k_ab = _safe_kappa(va, vb)
                k_ac = _safe_kappa(va, vc)
                k_bc = _safe_kappa(vb, vc)
                n_positive_total = int(sum(va) + sum(vb) + sum(vc))
                rows.append(
                    {
                        "field": f"admission_reason_tags::{tag}",
                        "kappa_A_B": k_ab,
                        "kappa_A_C": k_ac,
                        "kappa_B_C": k_bc,
                        "kappa_mean": float(np.mean([k_ab, k_ac, k_bc])),
                        "n_positive_total": n_positive_total,
                        "low_base_rate_flag": bool(low_base_rate_flag(n_positive_total)),
                    }
                )
            continue

        if field in tristate_fields:
            va_raw = [parsed_a[h].get(field) for h in overlap]
            vb_raw = [parsed_b[h].get(field) for h in overlap]
            vc_raw = [parsed_c[h].get(field) for h in overlap]
            va = [_encode_tristate(v, collapse=collapse_tristate) for v in va_raw]
            vb = [_encode_tristate(v, collapse=collapse_tristate) for v in vb_raw]
            vc = [_encode_tristate(v, collapse=collapse_tristate) for v in vc_raw]
            n_positive_total = int(
                sum(1 for v in va_raw if _normalize_tristate_token(v, collapse=False) == "yes")
                + sum(1 for v in vb_raw if _normalize_tristate_token(v, collapse=False) == "yes")
                + sum(1 for v in vc_raw if _normalize_tristate_token(v, collapse=False) == "yes")
            )
        else:
            va_raw = [parsed_a[h].get(field, "not_documented") for h in overlap]
            vb_raw = [parsed_b[h].get(field, "not_documented") for h in overlap]
            vc_raw = [parsed_c[h].get(field, "not_documented") for h in overlap]
            cats = sorted(
                {str(v) for v in va_raw} | {str(v) for v in vb_raw} | {str(v) for v in vc_raw}
            )
            mapping = {cat: idx for idx, cat in enumerate(cats)}
            va = [mapping[str(v)] for v in va_raw]
            vb = [mapping[str(v)] for v in vb_raw]
            vc = [mapping[str(v)] for v in vc_raw]
            if field in ENUM_FIELDS:
                n_positive_total = int(
                    sum(1 for v in va_raw if str(v) != "not_documented")
                    + sum(1 for v in vb_raw if str(v) != "not_documented")
                    + sum(1 for v in vc_raw if str(v) != "not_documented")
                )
            else:
                n_positive_total = 0

        k_ab = _safe_kappa(va, vb)
        k_ac = _safe_kappa(va, vc)
        k_bc = _safe_kappa(vb, vc)
        rows.append(
            {
                "field": field,
                "kappa_A_B": k_ab,
                "kappa_A_C": k_ac,
                "kappa_B_C": k_bc,
                "kappa_mean": float(np.mean([k_ab, k_ac, k_bc])),
                "n_positive_total": n_positive_total,
                "low_base_rate_flag": bool(low_base_rate_flag(n_positive_total)),
            }
        )
    return rows


def _median_filtered(rows: list[dict[str, Any]]) -> float:
    values = [float(r["kappa_mean"]) for r in rows if not bool(r["low_base_rate_flag"])]
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _paired_accuracy_rows(
    *,
    variant: str,
    gold_parsed: dict[int, dict[str, Any]],
    nano_parsed: dict[int, dict[str, Any]],
    tristate_fields: set[str],
) -> list[dict[str, Any]]:
    overlap = sorted(set(gold_parsed) & set(nano_parsed))
    rows: list[dict[str, Any]] = []

    for field in sorted(tristate_fields):
        gold_vals = [gold_parsed[h].get(field) for h in overlap]
        nano_vals = [nano_parsed[h].get(field) for h in overlap]

        left_full = [_encode_tristate(v, collapse=False) for v in gold_vals]
        right_full = [_encode_tristate(v, collapse=False) for v in nano_vals]
        left_collapsed = [_encode_tristate(v, collapse=True) for v in gold_vals]
        right_collapsed = [_encode_tristate(v, collapse=True) for v in nano_vals]

        agree_collapsed_n = sum(
            1 for lv, rv in zip(left_collapsed, right_collapsed, strict=False) if lv == rv
        )
        ci_lo, ci_hi = _wilson_ci(agree_collapsed_n, len(overlap))

        rows.append(
            {
                "variant": variant.upper(),
                "field": field,
                "n_overlap": len(overlap),
                "agreement_full_pct": _agreement_percent(left_full, right_full),
                "agreement_collapsed_pct": _agreement_percent(left_collapsed, right_collapsed),
                "agreement_collapsed_ci95_low_pct": ci_lo * 100.0,
                "agreement_collapsed_ci95_high_pct": ci_hi * 100.0,
                "kappa_full": _safe_kappa(left_full, right_full),
                "kappa_collapsed": _safe_kappa(left_collapsed, right_collapsed),
            }
        )

    rows.sort(key=lambda row: (row["variant"], float(row["kappa_collapsed"])))
    return rows


def _variant_median_kappas(accuracy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in ["A", "B", "C"]:
        subset = [row for row in accuracy_rows if row["variant"] == variant]
        full_vals = [float(row["kappa_full"]) for row in subset]
        collapse_vals = [float(row["kappa_collapsed"]) for row in subset]
        out.append(
            {
                "variant": variant,
                "median_kappa_full": float(np.median(full_vals)) if full_vals else 0.0,
                "median_kappa_collapsed": float(np.median(collapse_vals)) if collapse_vals else 0.0,
            }
        )
    return out


def _reasoning_disagreement_rows(
    *,
    on_parsed: dict[int, dict[str, Any]],
    off_parsed: dict[int, dict[str, Any]],
    tristate_fields: set[str],
) -> tuple[int, list[dict[str, Any]], float, int]:
    overlap = sorted(set(on_parsed) & set(off_parsed))
    rows: list[dict[str, Any]] = []

    for field in sorted(tristate_fields):
        left = [_encode_tristate(on_parsed[h].get(field), collapse=True) for h in overlap]
        right = [_encode_tristate(off_parsed[h].get(field), collapse=True) for h in overlap]
        disagree = sum(1 for lv, rv in zip(left, right, strict=False) if lv != rv)
        rows.append(
            {
                "field": field,
                "n_overlap": len(overlap),
                "n_disagree": disagree,
                "disagree_rate_pct": (disagree / len(overlap) * 100.0) if overlap else 0.0,
                "agree_rate_pct": ((len(overlap) - disagree) / len(overlap) * 100.0)
                if overlap
                else 0.0,
            }
        )

    rows.sort(key=lambda row: float(row["disagree_rate_pct"]), reverse=True)
    total_cells = len(overlap) * len(tristate_fields)
    total_disagree = sum(int(row["n_disagree"]) for row in rows)
    total_disagree_rate = float(total_disagree / total_cells * 100.0) if total_cells else 0.0
    return len(overlap), rows, total_disagree_rate, total_disagree


def _reasoning_vs_gold_rows(
    *,
    on_parsed: dict[int, dict[str, Any]],
    off_parsed: dict[int, dict[str, Any]],
    gold_parsed: dict[int, dict[str, Any]],
    tristate_fields: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in sorted(tristate_fields):
        overlap_on = sorted(set(on_parsed) & set(gold_parsed))
        overlap_off = sorted(set(off_parsed) & set(gold_parsed))
        overlap_common = sorted(set(on_parsed) & set(off_parsed) & set(gold_parsed))

        on_full = [_encode_tristate(on_parsed[h].get(field), collapse=False) for h in overlap_on]
        gold_on_full = [
            _encode_tristate(gold_parsed[h].get(field), collapse=False) for h in overlap_on
        ]
        off_full = [_encode_tristate(off_parsed[h].get(field), collapse=False) for h in overlap_off]
        gold_off_full = [
            _encode_tristate(gold_parsed[h].get(field), collapse=False) for h in overlap_off
        ]

        on_col = [_encode_tristate(on_parsed[h].get(field), collapse=True) for h in overlap_on]
        gold_on_col = [
            _encode_tristate(gold_parsed[h].get(field), collapse=True) for h in overlap_on
        ]
        off_col = [_encode_tristate(off_parsed[h].get(field), collapse=True) for h in overlap_off]
        gold_off_col = [
            _encode_tristate(gold_parsed[h].get(field), collapse=True) for h in overlap_off
        ]

        on_col_common = [
            _encode_tristate(on_parsed[h].get(field), collapse=True) for h in overlap_common
        ]
        off_col_common = [
            _encode_tristate(off_parsed[h].get(field), collapse=True) for h in overlap_common
        ]
        gold_col_common = [
            _encode_tristate(gold_parsed[h].get(field), collapse=True) for h in overlap_common
        ]

        rows.append(
            {
                "field": field,
                "on_overlap": len(overlap_on),
                "off_overlap": len(overlap_off),
                "common_overlap": len(overlap_common),
                "on_agree_full_pct": _agreement_percent(on_full, gold_on_full),
                "off_agree_full_pct": _agreement_percent(off_full, gold_off_full),
                "on_agree_collapsed_pct": _agreement_percent(on_col, gold_on_col),
                "off_agree_collapsed_pct": _agreement_percent(off_col, gold_off_col),
                "on_kappa_full": _safe_kappa(on_full, gold_on_full),
                "off_kappa_full": _safe_kappa(off_full, gold_off_full),
                "on_kappa_collapsed": _safe_kappa(on_col, gold_on_col),
                "off_kappa_collapsed": _safe_kappa(off_col, gold_off_col),
                "on_minus_off_collapsed_agree_pp": _agreement_percent(on_col, gold_on_col)
                - _agreement_percent(off_col, gold_off_col),
                "on_minus_off_collapsed_kappa": _safe_kappa(on_col, gold_on_col)
                - _safe_kappa(off_col, gold_off_col),
                "on_agree_collapsed_common_pct": _agreement_percent(on_col_common, gold_col_common),
                "off_agree_collapsed_common_pct": _agreement_percent(
                    off_col_common, gold_col_common
                ),
            }
        )

    rows.sort(key=lambda row: abs(float(row["on_minus_off_collapsed_agree_pp"])), reverse=True)
    return rows


def _write_per_field_report(
    *,
    output_path: Path,
    tristate_fields: list[str],
    nano_full_rows: list[dict[str, Any]],
    gold_full_rows: list[dict[str, Any]],
    nano_collapse_rows: list[dict[str, Any]],
    gold_collapse_rows: list[dict[str, Any]],
) -> dict[str, float]:
    by_full_nano = {str(row["field"]): row for row in nano_full_rows}
    by_full_gold = {str(row["field"]): row for row in gold_full_rows}
    by_col_nano = {str(row["field"]): row for row in nano_collapse_rows}
    by_col_gold = {str(row["field"]): row for row in gold_collapse_rows}

    tri_rows: list[dict[str, Any]] = []
    for field in tristate_fields:
        if (
            field not in by_full_nano
            or field not in by_full_gold
            or field not in by_col_nano
            or field not in by_col_gold
        ):
            continue

        nano_full = float(by_full_nano[field]["kappa_mean"])
        gold_full = float(by_full_gold[field]["kappa_mean"])
        nano_col = float(by_col_nano[field]["kappa_mean"])
        gold_col = float(by_col_gold[field]["kappa_mean"])
        delta_full = (gold_full - nano_full) * 100.0
        delta_col = (gold_col - nano_col) * 100.0
        tri_rows.append(
            {
                "field": field,
                "nano_kappa_full": nano_full,
                "nano_kappa_collapsed": nano_col,
                "gold_kappa_full": gold_full,
                "gold_kappa_collapsed": gold_col,
                "delta_full_pp": delta_full,
                "delta_collapsed_pp": delta_col,
                "collapse_effect_on_delta_pp": delta_col - delta_full,
                "collapse_interpretation": (
                    "collapsed_delta_gt_full_delta"
                    if delta_col > delta_full
                    else "collapsed_delta_lt_or_eq_full_delta"
                ),
            }
        )

    tri_rows.sort(key=lambda row: float(row["delta_collapsed_pp"]), reverse=True)

    full_nano_med = _median_filtered(nano_full_rows)
    full_gold_med = _median_filtered(gold_full_rows)
    full_delta_med_pp = (full_gold_med - full_nano_med) * 100.0

    col_nano_med = _median_filtered(nano_collapse_rows)
    col_gold_med = _median_filtered(gold_collapse_rows)
    col_delta_med_pp = (col_gold_med - col_nano_med) * 100.0

    full_headline = (
        f"- Filtered median kappa (full): nano `{full_nano_med:.4f}`, "
        f"gold `{full_gold_med:.4f}`, delta `{full_delta_med_pp:+.2f}` pp"
    )
    collapsed_headline = (
        f"- Filtered median kappa (collapsed): nano `{col_nano_med:.4f}`, "
        f"gold `{col_gold_med:.4f}`, delta `{col_delta_med_pp:+.2f}` pp"
    )
    shift_headline = (
        "- Collapse shifts filtered median model-size effect by "
        f"`{(col_delta_med_pp - full_delta_med_pp):+.2f}` pp"
    )

    headline_lines = [
        "# Prompt 33 Task 1 — Paired TriState Collapse Per-field",
        "",
        "## Headline",
        full_headline,
        collapsed_headline,
        shift_headline,
        "",
        "## TriState Field Table",
        _markdown_table(
            [
                {
                    "field": row["field"],
                    "nano_kappa_full": f"{row['nano_kappa_full']:.4f}",
                    "nano_kappa_collapsed": f"{row['nano_kappa_collapsed']:.4f}",
                    "gold_kappa_full": f"{row['gold_kappa_full']:.4f}",
                    "gold_kappa_collapsed": f"{row['gold_kappa_collapsed']:.4f}",
                    "delta_full_pp": f"{row['delta_full_pp']:+.2f}",
                    "delta_collapsed_pp": f"{row['delta_collapsed_pp']:+.2f}",
                    "collapse_effect_on_delta_pp": f"{row['collapse_effect_on_delta_pp']:+.2f}",
                    "flag": row["collapse_interpretation"],
                }
                for row in tri_rows
            ],
            [
                "field",
                "nano_kappa_full",
                "nano_kappa_collapsed",
                "gold_kappa_full",
                "gold_kappa_collapsed",
                "delta_full_pp",
                "delta_collapsed_pp",
                "collapse_effect_on_delta_pp",
                "flag",
            ],
        ),
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(headline_lines), encoding="utf-8")

    return {
        "full_nano_median": full_nano_med,
        "full_gold_median": full_gold_med,
        "full_delta_median_pp": full_delta_med_pp,
        "collapsed_nano_median": col_nano_med,
        "collapsed_gold_median": col_gold_med,
        "collapsed_delta_median_pp": col_delta_med_pp,
        "collapse_shift_pp": col_delta_med_pp - full_delta_med_pp,
    }


def _write_accuracy_report(
    *,
    output_path: Path,
    accuracy_rows: list[dict[str, Any]],
) -> None:
    cardiac_rows = [row for row in accuracy_rows if row["field"] == "cardiac_rehab_referred"]
    home_rows = [row for row in accuracy_rows if row["field"] == "home_health_ordered"]

    def _median(values: list[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0

    cardiac_full_med = _median([float(row["kappa_full"]) for row in cardiac_rows])
    cardiac_col_med = _median([float(row["kappa_collapsed"]) for row in cardiac_rows])
    home_full_med = _median([float(row["kappa_full"]) for row in home_rows])
    home_col_med = _median([float(row["kappa_collapsed"]) for row in home_rows])

    table_rows = [
        {
            "variant": row["variant"],
            "field": row["field"],
            "n_overlap": row["n_overlap"],
            "agreement_full_pct": f"{float(row['agreement_full_pct']):.2f}",
            "agreement_collapsed_pct": f"{float(row['agreement_collapsed_pct']):.2f}",
            "agreement_collapsed_ci95": (
                f"[{float(row['agreement_collapsed_ci95_low_pct']):.2f}, "
                f"{float(row['agreement_collapsed_ci95_high_pct']):.2f}]"
            ),
            "kappa_full": f"{float(row['kappa_full']):.4f}",
            "kappa_collapsed": f"{float(row['kappa_collapsed']):.4f}",
            "delta_kappa_collapse_minus_full": (
                f"{(float(row['kappa_collapsed']) - float(row['kappa_full'])):+.4f}"
            ),
        }
        for row in accuracy_rows
    ]

    lines = [
        "# Prompt 33 Task 1 — Per-variant Accuracy Under TriState Collapse",
        "",
        "## Headline",
        (
            "- `cardiac_rehab_referred` median kappa across variants: "
            f"full `{cardiac_full_med:.4f}` -> collapsed `{cardiac_col_med:.4f}`"
        ),
        (
            "- `home_health_ordered` median kappa across variants: "
            f"full `{home_full_med:.4f}` -> collapsed `{home_col_med:.4f}`"
        ),
        "- Collapse uses `yes->yes`, `no/not_documented->not_yes`, and preserves `null->null`.",
        "",
        "## Table",
        _markdown_table(
            table_rows,
            [
                "variant",
                "field",
                "n_overlap",
                "agreement_full_pct",
                "agreement_collapsed_pct",
                "agreement_collapsed_ci95",
                "kappa_full",
                "kappa_collapsed",
                "delta_kappa_collapse_minus_full",
            ],
        ),
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_reasoning_report(
    *,
    output_path: Path,
    run_on: str,
    run_off: str,
    run_gold: str,
    n_overlap_on_off: int,
    tri_field_count: int,
    total_disagree_rate: float,
    total_disagree_n: int,
    disagreement_rows: list[dict[str, Any]],
    on_off_vs_gold_rows: list[dict[str, Any]],
) -> None:
    top12 = disagreement_rows[:12]
    shock_row = next((row for row in on_off_vs_gold_rows if row["field"] == "shock_present"), None)

    overview_rows: list[dict[str, Any]] = [
        {"metric": "timestamp_utc", "value": datetime.now(tz=UTC).isoformat()},
        {"metric": "run_on", "value": run_on},
        {"metric": "run_off", "value": run_off},
        {"metric": "run_gold", "value": run_gold},
        {"metric": "n_overlap_on_off_notes", "value": n_overlap_on_off},
        {"metric": "n_tristate_fields", "value": tri_field_count},
        {"metric": "n_collapsed_disagreements", "value": total_disagree_n},
        {"metric": "collapsed_cell_disagreement_rate_pct", "value": f"{total_disagree_rate:.3f}"},
    ]

    disagree_table = [
        {
            "field": row["field"],
            "n_overlap": row["n_overlap"],
            "n_disagree": row["n_disagree"],
            "disagree_rate_pct": f"{float(row['disagree_rate_pct']):.3f}",
            "agree_rate_pct": f"{float(row['agree_rate_pct']):.3f}",
        }
        for row in top12
    ]

    compare_table = [
        {
            "field": row["field"],
            "on_overlap": row["on_overlap"],
            "off_overlap": row["off_overlap"],
            "common_overlap": row["common_overlap"],
            "on_agree_full_pct": f"{float(row['on_agree_full_pct']):.3f}",
            "off_agree_full_pct": f"{float(row['off_agree_full_pct']):.3f}",
            "on_agree_collapsed_pct": f"{float(row['on_agree_collapsed_pct']):.3f}",
            "off_agree_collapsed_pct": f"{float(row['off_agree_collapsed_pct']):.3f}",
            "on_kappa_collapsed": f"{float(row['on_kappa_collapsed']):.4f}",
            "off_kappa_collapsed": f"{float(row['off_kappa_collapsed']):.4f}",
            "delta_on_minus_off_collapsed_agree_pp": (
                f"{float(row['on_minus_off_collapsed_agree_pp']):+.3f}"
            ),
        }
        for row in on_off_vs_gold_rows
    ]

    spotlight_lines: list[str] = []
    if shock_row is not None:
        spotlight_lines = [
            "## shock_present Spotlight",
            (
                "- Full agreement vs gold: "
                f"ON `{float(shock_row['on_agree_full_pct']):.3f}%` vs "
                f"OFF `{float(shock_row['off_agree_full_pct']):.3f}%`"
            ),
            (
                "- Collapsed agreement vs gold: "
                f"ON `{float(shock_row['on_agree_collapsed_pct']):.3f}%` vs "
                f"OFF `{float(shock_row['off_agree_collapsed_pct']):.3f}%`"
            ),
            (
                "- Collapsed kappa vs gold: "
                f"ON `{float(shock_row['on_kappa_collapsed']):.4f}` vs "
                f"OFF `{float(shock_row['off_kappa_collapsed']):.4f}`"
            ),
            "",
        ]

    lines = [
        "# Prompt 33 Task 4 — Reasoning ON/OFF Under TriState Collapse",
        "",
        "## Overview",
        _markdown_table(overview_rows, ["metric", "value"]),
        "",
        "## ON vs OFF Per-field Disagreement (Top 12, collapsed)",
        _markdown_table(
            disagree_table,
            ["field", "n_overlap", "n_disagree", "disagree_rate_pct", "agree_rate_pct"],
        ),
        "",
        "## ON/OFF vs Gold (collapsed and full side-by-side)",
        _markdown_table(
            compare_table,
            [
                "field",
                "on_overlap",
                "off_overlap",
                "common_overlap",
                "on_agree_full_pct",
                "off_agree_full_pct",
                "on_agree_collapsed_pct",
                "off_agree_collapsed_pct",
                "on_kappa_collapsed",
                "off_kappa_collapsed",
                "delta_on_minus_off_collapsed_agree_pp",
            ],
        ),
        "",
        *spotlight_lines,
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_delta_plot(
    *,
    output_path: Path,
    tri_fields: list[str],
    nano_full_rows: list[dict[str, Any]],
    gold_full_rows: list[dict[str, Any]],
    nano_collapse_rows: list[dict[str, Any]],
    gold_collapse_rows: list[dict[str, Any]],
) -> None:
    by_full_nano = {str(row["field"]): row for row in nano_full_rows}
    by_full_gold = {str(row["field"]): row for row in gold_full_rows}
    by_col_nano = {str(row["field"]): row for row in nano_collapse_rows}
    by_col_gold = {str(row["field"]): row for row in gold_collapse_rows}

    rows: list[tuple[str, float, float]] = []
    for field in tri_fields:
        if (
            field not in by_full_nano
            or field not in by_full_gold
            or field not in by_col_nano
            or field not in by_col_gold
        ):
            continue
        full_delta = (
            float(by_full_gold[field]["kappa_mean"]) - float(by_full_nano[field]["kappa_mean"])
        ) * 100.0
        collapse_delta = (
            float(by_col_gold[field]["kappa_mean"]) - float(by_col_nano[field]["kappa_mean"])
        ) * 100.0
        rows.append((field, full_delta, collapse_delta))

    rows.sort(key=lambda item: item[2], reverse=True)

    labels = [row[0] for row in rows]
    full_vals = np.asarray([row[1] for row in rows], dtype=np.float64)
    col_vals = np.asarray([row[2] for row in rows], dtype=np.float64)

    y = np.arange(len(labels))
    height = 0.38

    plt.figure(figsize=(12.5, max(8.5, len(labels) * 0.33)))
    plt.barh(y + height / 2.0, full_vals, height=height, color="#c7d3f3", label="Full delta")
    plt.barh(y - height / 2.0, col_vals, height=height, color="#2f5aa8", label="Collapsed delta")
    plt.axvline(0.0, color="black", linewidth=0.8)

    full_median = float(np.median(full_vals)) if full_vals.size else 0.0
    col_median = float(np.median(col_vals)) if col_vals.size else 0.0
    plt.axvline(full_median, color="#6f84c3", linestyle="--", linewidth=1.2, label="Full median")
    plt.axvline(
        col_median, color="#153b7a", linestyle="-.", linewidth=1.2, label="Collapsed median"
    )

    plt.yticks(y, labels, fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel("Model-size effect (gold kappa mean - nano kappa mean), pp")
    plt.title("TriState Fields: Full vs Collapsed Model-size Effect")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def _write_variant_accuracy_plot(
    *,
    output_path: Path,
    variant_medians: list[dict[str, Any]],
    accuracy_rows: list[dict[str, Any]],
) -> None:
    variants = [row["variant"] for row in variant_medians]
    full_vals = np.asarray(
        [float(row["median_kappa_full"]) for row in variant_medians], dtype=np.float64
    )
    col_vals = np.asarray(
        [float(row["median_kappa_collapsed"]) for row in variant_medians], dtype=np.float64
    )

    x = np.arange(len(variants))
    width = 0.36

    plt.figure(figsize=(9.5, 6.8))
    plt.bar(x - width / 2.0, full_vals, width=width, color="#c8d6ea", label="Full median kappa")
    plt.bar(x + width / 2.0, col_vals, width=width, color="#2b6aa6", label="Collapsed median kappa")

    plt.xticks(x, variants)
    plt.ylim(0.0, min(1.0, max(col_vals.max(initial=0.0), full_vals.max(initial=0.0)) + 0.2))
    plt.ylabel("Median kappa across TriState fields")
    plt.title("Per-variant nano-vs-gold Accuracy: Full vs Collapsed")
    plt.legend(loc="upper left")

    cardiac_by_variant = {
        row["variant"]: row for row in accuracy_rows if row["field"] == "cardiac_rehab_referred"
    }
    home_by_variant = {
        row["variant"]: row for row in accuracy_rows if row["field"] == "home_health_ordered"
    }
    annotation_lines: list[str] = []
    for variant in variants:
        cardiac = cardiac_by_variant.get(variant)
        home = home_by_variant.get(variant)
        if cardiac is None or home is None:
            continue
        annotation_lines.append(
            f"{variant}: cardiac {float(cardiac['kappa_full']):.3f}->"
            f"{float(cardiac['kappa_collapsed']):.3f}, "
            f"home {float(home['kappa_full']):.3f}->"
            f"{float(home['kappa_collapsed']):.3f}"
        )
    plt.gcf().text(
        0.02,
        0.01,
        "\n".join(annotation_lines),
        fontsize=8.5,
        family="monospace",
        va="bottom",
        ha="left",
    )

    plt.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt 33 Task 1+4: paired collapse analysis and reasoning collapse analysis"
    )
    parser.add_argument(
        "--output-per-field",
        default="codex_outputs/91_paired_collapse_per_field.md",
    )
    parser.add_argument(
        "--output-accuracy",
        default="codex_outputs/91_paired_collapse_accuracy.md",
    )
    parser.add_argument(
        "--output-reasoning",
        default="codex_outputs/91_reasoning_collapse.md",
    )
    parser.add_argument(
        "--figure-per-field",
        default="docs/figures/91_collapse_vs_full_per_field.png",
    )
    parser.add_argument(
        "--figure-variant-accuracy",
        default="docs/figures/91_per_variant_accuracy_collapse_vs_full.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    all_fields = _field_names()
    tristate_fields = _detect_tristate_fields()
    tristate_set = set(tristate_fields)

    split_1k_ids = _ids_from_split(config.SPLITS_DIR / "methodology_1k.csv")
    split_500_ids = _ids_from_split(config.SPLITS_DIR / "methodology_5k_audit_500.csv")

    gold_1k = {
        "a": _load_results("paired_gold_methodology_1k_a"),
        "b": _load_results("paired_gold_methodology_1k_b"),
        "c": _load_results("paired_gold_methodology_1k_c"),
    }
    gold_500 = {
        "a": _load_results("paired_gold_methodology_5k_audit_a"),
        "b": _load_results("paired_gold_methodology_5k_audit_b"),
        "c": _load_results("paired_gold_methodology_5k_audit_c"),
    }
    nano_1k = {
        "a": _load_results("methodology_1k_a"),
        "b": _load_results("methodology_1k_b"),
        "c": _load_results("methodology_1k_c"),
    }
    nano_500 = {
        "a": _load_results("production_v1"),
        "b": _load_results("methodology_5k_audit_b"),
        "c": _load_results("methodology_5k_audit_c"),
    }

    for variant in ["a", "b", "c"]:
        gold_1k[variant] = {k: v for k, v in gold_1k[variant].items() if k in split_1k_ids}
        gold_500[variant] = {k: v for k, v in gold_500[variant].items() if k in split_500_ids}
        nano_1k[variant] = {k: v for k, v in nano_1k[variant].items() if k in split_1k_ids}
        nano_500[variant] = {k: v for k, v in nano_500[variant].items() if k in split_500_ids}

    merged_gold = {
        "a": _merge_variant_runs(gold_1k["a"], gold_500["a"]),
        "b": _merge_variant_runs(gold_1k["b"], gold_500["b"]),
        "c": _merge_variant_runs(gold_1k["c"], gold_500["c"]),
    }
    merged_nano = {
        "a": _merge_variant_runs(nano_1k["a"], nano_500["a"]),
        "b": _merge_variant_runs(nano_1k["b"], nano_500["b"]),
        "c": _merge_variant_runs(nano_1k["c"], nano_500["c"]),
    }

    nano_full_rows = _cross_variant_kappas(
        parsed_a=merged_nano["a"],
        parsed_b=merged_nano["b"],
        parsed_c=merged_nano["c"],
        fields=all_fields,
        tristate_fields=tristate_set,
        collapse_tristate=False,
    )
    gold_full_rows = _cross_variant_kappas(
        parsed_a=merged_gold["a"],
        parsed_b=merged_gold["b"],
        parsed_c=merged_gold["c"],
        fields=all_fields,
        tristate_fields=tristate_set,
        collapse_tristate=False,
    )
    nano_collapse_rows = _cross_variant_kappas(
        parsed_a=merged_nano["a"],
        parsed_b=merged_nano["b"],
        parsed_c=merged_nano["c"],
        fields=all_fields,
        tristate_fields=tristate_set,
        collapse_tristate=True,
    )
    gold_collapse_rows = _cross_variant_kappas(
        parsed_a=merged_gold["a"],
        parsed_b=merged_gold["b"],
        parsed_c=merged_gold["c"],
        fields=all_fields,
        tristate_fields=tristate_set,
        collapse_tristate=True,
    )

    per_field_headlines = _write_per_field_report(
        output_path=Path(args.output_per_field),
        tristate_fields=tristate_fields,
        nano_full_rows=nano_full_rows,
        gold_full_rows=gold_full_rows,
        nano_collapse_rows=nano_collapse_rows,
        gold_collapse_rows=gold_collapse_rows,
    )

    accuracy_rows: list[dict[str, Any]] = []
    for variant in ["a", "b", "c"]:
        accuracy_rows.extend(
            _paired_accuracy_rows(
                variant=variant,
                gold_parsed=merged_gold[variant],
                nano_parsed=merged_nano[variant],
                tristate_fields=tristate_set,
            )
        )

    _write_accuracy_report(output_path=Path(args.output_accuracy), accuracy_rows=accuracy_rows)

    _write_delta_plot(
        output_path=Path(args.figure_per_field),
        tri_fields=tristate_fields,
        nano_full_rows=nano_full_rows,
        gold_full_rows=gold_full_rows,
        nano_collapse_rows=nano_collapse_rows,
        gold_collapse_rows=gold_collapse_rows,
    )
    _write_variant_accuracy_plot(
        output_path=Path(args.figure_variant_accuracy),
        variant_medians=_variant_median_kappas(accuracy_rows),
        accuracy_rows=accuracy_rows,
    )

    reasoning_on = _load_results("reasoning_on_methodology_1k_a")
    reasoning_off = _load_results("methodology_1k_a")
    reasoning_gold = _load_results("paired_gold_methodology_1k_a")

    n_overlap_on_off, disagreement_rows, total_disagree_rate, total_disagree_n = (
        _reasoning_disagreement_rows(
            on_parsed=reasoning_on,
            off_parsed=reasoning_off,
            tristate_fields=tristate_set,
        )
    )
    on_off_vs_gold_rows = _reasoning_vs_gold_rows(
        on_parsed=reasoning_on,
        off_parsed=reasoning_off,
        gold_parsed=reasoning_gold,
        tristate_fields=tristate_set,
    )
    _write_reasoning_report(
        output_path=Path(args.output_reasoning),
        run_on="reasoning_on_methodology_1k_a",
        run_off="methodology_1k_a",
        run_gold="paired_gold_methodology_1k_a",
        n_overlap_on_off=n_overlap_on_off,
        tri_field_count=len(tristate_fields),
        total_disagree_rate=total_disagree_rate,
        total_disagree_n=total_disagree_n,
        disagreement_rows=disagreement_rows,
        on_off_vs_gold_rows=on_off_vs_gold_rows,
    )

    print(json.dumps({"status": "ok", "headlines": per_field_headlines}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
