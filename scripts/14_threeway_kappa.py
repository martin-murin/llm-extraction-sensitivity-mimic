"""
Runs staged pipeline step `14_threeway_kappa.py`.

Reads: codex_outputs/16_threeway_kappa_report.md.
Writes: codex_outputs/16_threeway_kappa_report.md.
Backs median kappa and cross-prompt agreement claims.
Usage: `python scripts/14_threeway_kappa.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src import config
from src.schema.fields import LLMNoteFeatures
from src.schema.vocabulary import ADMISSION_REASON_TAGS, CHAPTER_TO_PLAUSIBLE_TAGS
from src.utils.threeway_kappa import (
    cohen_kappa_safe,
    count_positive_admission_tag,
    count_positive_enum,
    count_positive_tristate,
    encode_admission_reason_tags,
    encode_tristate,
    intersect_successful_hadm_ids,
    low_base_rate_flag,
    pabak_score,
    percent_agreement,
)

PAIR_KEYS = [("A", "B"), ("A", "C"), ("B", "C")]
TRISTATE_FIELDS = {
    "shock_present",
    "infection_as_trigger",
    "aki_present",
    "lives_alone",
    "social_support_absent",
    "financial_hardship",
    "substance_use_active",
    "fall_risk_documented",
    "cognitive_impairment",
    "goals_of_care_flag",
    "palliative_care_consult",
    "dnr_dni_documented",
    "home_health_ordered",
    "cardiac_rehab_referred",
    "discharge_delayed_reason",
    "hospital_acquired_complication",
    "unresolved_diagnosis_at_discharge",
}
ENUM_FIELDS = {"functional_status", "mental_status", "discharge_condition_category"}
COUNT_FIELDS = {"new_meds_started_count", "meds_stopped_count"}
SKIP_KAPPA_FIELDS = {"primary_diagnosis_text", "reasoning", *COUNT_FIELDS}


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return f"{value:.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if value is None:
        return ""
    return str(value)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        rendered = [
            _format_number(row.get(column, "")).replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *lines])


def _load_results(run_id: str) -> dict[str, Any]:
    base_dir = config.RAW_RESPONSES_DIR / run_id
    results_path = base_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    parsed: dict[int, dict[str, Any]] = {}
    parse_ok = 0
    for payload in records:
        if bool(payload.get("parse_ok")) and isinstance(payload.get("features_json"), dict):
            hadm_id = int(payload["hadm_id"])
            parsed[hadm_id] = payload["features_json"]
            parse_ok += 1

    input_tokens = int(sum(int(item.get("input_tokens", 0) or 0) for item in records))
    output_tokens = int(sum(int(item.get("output_tokens", 0) or 0) for item in records))

    return {
        "run_id": run_id,
        "results_path": results_path,
        "mtime_utc": datetime.fromtimestamp(results_path.stat().st_mtime, tz=UTC).isoformat(),
        "n_attempted": len(records),
        "n_parsed": parse_ok,
        "parsed": parsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "records": records,
    }


def _pairwise_metrics(encoded: dict[str, list[int]]) -> dict[str, float]:
    out: dict[str, float] = {}
    kappas: list[float] = []
    pabaks: list[float] = []
    agreements: list[float] = []
    for left, right in PAIR_KEYS:
        key = f"{left}_{right}"
        kappa = cohen_kappa_safe(encoded[left], encoded[right])
        pabak = pabak_score(encoded[left], encoded[right])
        agreement = percent_agreement(encoded[left], encoded[right])
        out[f"kappa_{key}"] = kappa
        out[f"pabak_{key}"] = pabak
        out[f"agreement_{key}"] = agreement
        kappas.append(kappa)
        pabaks.append(pabak)
        agreements.append(agreement)
    out["kappa_mean"] = float(np.mean(kappas))
    out["pabak_mean"] = float(np.mean(pabaks))
    out["pct_agreement_mean"] = float(np.mean(agreements))
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute three-way kappa across variant runs.")
    parser.add_argument(
        "--run-ids",
        nargs=3,
        default=["refinement_v1_a", "refinement_v1_b", "refinement_v1_c"],
    )
    parser.add_argument("--output", default="codex_outputs/16_threeway_kappa_report.md")
    parser.add_argument(
        "--split",
        choices=[
            "refinement",
            "holdout",
            "smoke",
            "methodology_1k",
            "methodology_5k",
            "methodology_5k_audit_500",
        ],
        default=None,
        help="Optional explicit split override for ICD concordance lookup.",
    )
    return parser.parse_args()


def _dominant_concordance(
    split_frame: pd.DataFrame,
    features_by_hadm: dict[int, dict[str, Any]],
) -> tuple[int, int, float]:
    checked = 0
    concordant = 0
    chapters = split_frame.set_index("hadm_id")["chapter"].to_dict()
    for hadm_id, features in features_by_hadm.items():
        chapter = chapters.get(hadm_id)
        if chapter is None:
            continue
        plausible = CHAPTER_TO_PLAUSIBLE_TAGS.get(str(chapter), set())
        if not plausible:
            continue
        checked += 1
        dominant = str(features.get("dominant_admission_reason", ""))
        if dominant in plausible:
            concordant += 1
    rate = (concordant / checked) if checked else 0.0
    return checked, concordant, rate


def _base_rate_rows(
    field: str,
    joined_hadm_ids: list[int],
    by_variant: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    values = sorted(
        {
            str(by_variant[variant][hadm_id].get(field))
            for variant in ["A", "B", "C"]
            for hadm_id in joined_hadm_ids
        }
    )
    rows: list[dict[str, Any]] = []
    for value in values:
        rates = {}
        for variant in ["A", "B", "C"]:
            hits = sum(
                1
                for hadm_id in joined_hadm_ids
                if str(by_variant[variant][hadm_id].get(field)) == value
            )
            rates[variant] = hits / len(joined_hadm_ids) if joined_hadm_ids else 0.0
        rows.append(
            {
                "field": field,
                "value": value,
                "A_pct": _pct(rates["A"]),
                "B_pct": _pct(rates["B"]),
                "C_pct": _pct(rates["C"]),
                "max_divergence_pct": _pct(max(rates.values()) - min(rates.values())),
            }
        )
    return rows


def _choose_outlier_variant(values: dict[str, Any]) -> str | None:
    counts: dict[str, int] = {}
    for value in values.values():
        counts[str(value)] = counts.get(str(value), 0) + 1
    majority_values = {value for value, count in counts.items() if count == 2}
    if not majority_values:
        return None
    majority = next(iter(majority_values))
    for variant in ["A", "B", "C"]:
        if str(values[variant]) != majority:
            return variant
    return None


def _reasoning_excerpt(text: str | None, max_chars: int = 320) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[:max_chars]


def _band_counts(kappas: np.ndarray) -> dict[str, Any]:
    if kappas.size == 0:
        return {
            "kappa_ge_0_8": 0,
            "kappa_0_6_to_0_8": 0,
            "kappa_0_4_to_0_6": 0,
            "kappa_lt_0_4": 0,
            "overall_median_kappa": 0.0,
        }
    return {
        "kappa_ge_0_8": int(np.sum(kappas >= 0.8)),
        "kappa_0_6_to_0_8": int(np.sum((kappas >= 0.6) & (kappas < 0.8))),
        "kappa_0_4_to_0_6": int(np.sum((kappas >= 0.4) & (kappas < 0.6))),
        "kappa_lt_0_4": int(np.sum(kappas < 0.4)),
        "overall_median_kappa": float(np.median(kappas)),
    }


def _infer_split_from_run_ids(run_ids: list[str]) -> str:
    lowered = [run_id.lower() for run_id in run_ids]
    if lowered and all("methodology_5k_audit_500" in run_id for run_id in lowered):
        return "methodology_5k_audit_500"
    if lowered and all("methodology_5k" in run_id for run_id in lowered):
        return "methodology_5k"
    if lowered and all("methodology_1k" in run_id for run_id in lowered):
        return "methodology_1k"
    if lowered and all("holdout" in run_id for run_id in lowered):
        return "holdout"
    if lowered and all("smoke" in run_id for run_id in lowered):
        return "smoke"
    return "refinement"


def _resolve_split_csv(split_name: str) -> Path:
    direct = config.SPLITS_DIR / f"{split_name}.csv"
    if direct.exists():
        return direct
    candidates = sorted(config.SPLITS_DIR.glob(f"{split_name}_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find split CSV matching pattern '{split_name}_*.csv' in {config.SPLITS_DIR}"
        )
    return candidates[0]


def main() -> int:
    args = _parse_args()
    config.load_env()

    loaded = [_load_results(run_id) for run_id in args.run_ids]
    by_variant = {
        "A": loaded[0]["parsed"],
        "B": loaded[1]["parsed"],
        "C": loaded[2]["parsed"],
    }
    joined_hadm_ids = intersect_successful_hadm_ids(by_variant)

    split_name = args.split or _infer_split_from_run_ids(args.run_ids)
    split_path = _resolve_split_csv(split_name)
    split_frame = pd.read_csv(split_path)
    split_frame["hadm_id"] = pd.to_numeric(split_frame["hadm_id"], errors="coerce").astype("int64")

    field_names = [
        field for field in LLMNoteFeatures.model_fields if field not in SKIP_KAPPA_FIELDS
    ]
    kappa_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    base_rate_rows: list[dict[str, Any]] = []
    disagreement_pool: dict[str, list[dict[str, Any]]] = {}

    for field in field_names:
        if field == "admission_reason_tags":
            tag_scores = []
            for tag in ADMISSION_REASON_TAGS:
                encoded: dict[str, list[int]] = {}
                tag_lists_by_variant: dict[str, list[list[str]]] = {"A": [], "B": [], "C": []}
                for variant in ["A", "B", "C"]:
                    encoded_values: list[int] = []
                    for hadm_id in joined_hadm_ids:
                        tags = list(by_variant[variant][hadm_id].get("admission_reason_tags", []))
                        tag_lists_by_variant[variant].append(tags)
                        encoded_values.append(encode_admission_reason_tags(tags)[tag])
                    encoded[variant] = encoded_values

                metrics = _pairwise_metrics(encoded)
                n_positive_a = count_positive_admission_tag(tag_lists_by_variant["A"], tag)
                n_positive_b = count_positive_admission_tag(tag_lists_by_variant["B"], tag)
                n_positive_c = count_positive_admission_tag(tag_lists_by_variant["C"], tag)
                n_positive_total = n_positive_a + n_positive_b + n_positive_c
                low_base = low_base_rate_flag(n_positive_total)

                row_name = f"admission_reason_tags::{tag}"
                kappa_rows.append(
                    {
                        "field": row_name,
                        "field_group": "admission_reason_tags",
                        "kappa_A_B": metrics["kappa_A_B"],
                        "kappa_A_C": metrics["kappa_A_C"],
                        "kappa_B_C": metrics["kappa_B_C"],
                        "kappa_mean": metrics["kappa_mean"],
                        "pabak_mean": metrics["pabak_mean"],
                        "pct_agreement_mean": metrics["pct_agreement_mean"],
                        "n_positive_a": n_positive_a,
                        "n_positive_b": n_positive_b,
                        "n_positive_c": n_positive_c,
                        "n_positive_total": n_positive_total,
                        "low_base_rate_flag": low_base,
                        "flag_lt_0_6": "YES" if metrics["kappa_mean"] < 0.6 else "",
                    }
                )
                tag_scores.append(metrics["kappa_mean"])

                disagreements = []
                for hadm_id in joined_hadm_ids:
                    values = {
                        variant: encode_admission_reason_tags(
                            by_variant[variant][hadm_id].get("admission_reason_tags", [])
                        )[tag]
                        for variant in ["A", "B", "C"]
                    }
                    if len(set(values.values())) <= 1:
                        continue
                    outlier = _choose_outlier_variant(values)
                    outlier_key = outlier if outlier is not None else "A"
                    reasoning = _reasoning_excerpt(
                        str(by_variant[outlier_key][hadm_id].get("reasoning", ""))
                    )
                    disagreements.append(
                        {
                            "hadm_id": hadm_id,
                            "A_vote": "yes" if values["A"] else "no",
                            "B_vote": "yes" if values["B"] else "no",
                            "C_vote": "yes" if values["C"] else "no",
                            "outlier_variant": outlier or "",
                            "outlier_reasoning_excerpt": reasoning,
                        }
                    )
                disagreement_pool[row_name] = disagreements

            class_rows.append(
                {
                    "field_group": "admission_reason_tags",
                    "mean_kappa": float(np.mean(tag_scores)) if tag_scores else 0.0,
                }
            )
            continue

        encoded_map: dict[str, list[int]] = {}
        display_values: dict[str, dict[int, Any]] = {"A": {}, "B": {}, "C": {}}

        if field in TRISTATE_FIELDS:
            field_group = "tristate"
            positives = {"A": 0, "B": 0, "C": 0}
            for variant in ["A", "B", "C"]:
                tristate_values = [
                    str(by_variant[variant][hadm_id].get(field, "not_documented"))
                    for hadm_id in joined_hadm_ids
                ]
                positives[variant] = count_positive_tristate(tristate_values)
                encoded_map[variant] = [encode_tristate(value) for value in tristate_values]
                for hadm_id, value in zip(joined_hadm_ids, tristate_values, strict=True):
                    display_values[variant][hadm_id] = value
            base_rate_rows.extend(_base_rate_rows(field, joined_hadm_ids, by_variant))
        elif field in ENUM_FIELDS:
            field_group = "enum"
            positives = {"A": 0, "B": 0, "C": 0}
            observed = sorted(
                {
                    str(by_variant[variant][hadm_id].get(field))
                    for variant in ["A", "B", "C"]
                    for hadm_id in joined_hadm_ids
                }
            )
            mapping = {value: index for index, value in enumerate(observed)}
            for variant in ["A", "B", "C"]:
                enum_values = [
                    str(by_variant[variant][hadm_id].get(field))
                    for hadm_id in joined_hadm_ids
                ]
                positives[variant] = count_positive_enum(enum_values)
                encoded_map[variant] = [mapping[value] for value in enum_values]
                for hadm_id, value in zip(joined_hadm_ids, enum_values, strict=True):
                    display_values[variant][hadm_id] = value
            base_rate_rows.extend(_base_rate_rows(field, joined_hadm_ids, by_variant))
        elif field == "dominant_admission_reason":
            field_group = "dominant_admission_reason"
            positives = {
                "A": len(joined_hadm_ids),
                "B": len(joined_hadm_ids),
                "C": len(joined_hadm_ids),
            }
            mapping = {tag: index for index, tag in enumerate(ADMISSION_REASON_TAGS)}
            for variant in ["A", "B", "C"]:
                dominant_values = [
                    str(by_variant[variant][hadm_id].get(field, "other"))
                    for hadm_id in joined_hadm_ids
                ]
                encoded_map[variant] = [
                    mapping.get(value, mapping["other"])
                    for value in dominant_values
                ]
                for hadm_id, value in zip(joined_hadm_ids, dominant_values, strict=True):
                    display_values[variant][hadm_id] = value
        else:
            continue

        metrics = _pairwise_metrics(encoded_map)
        n_positive_total = positives["A"] + positives["B"] + positives["C"]
        low_base = low_base_rate_flag(n_positive_total)

        kappa_rows.append(
            {
                "field": field,
                "field_group": field_group,
                "kappa_A_B": metrics["kappa_A_B"],
                "kappa_A_C": metrics["kappa_A_C"],
                "kappa_B_C": metrics["kappa_B_C"],
                "kappa_mean": metrics["kappa_mean"],
                "pabak_mean": metrics["pabak_mean"],
                "pct_agreement_mean": metrics["pct_agreement_mean"],
                "n_positive_a": positives["A"],
                "n_positive_b": positives["B"],
                "n_positive_c": positives["C"],
                "n_positive_total": n_positive_total,
                "low_base_rate_flag": low_base,
                "flag_lt_0_6": "YES" if metrics["kappa_mean"] < 0.6 else "",
            }
        )
        class_rows.append({"field_group": field_group, "mean_kappa": metrics["kappa_mean"]})

        disagreements = []
        for hadm_id in joined_hadm_ids:
            values = {variant: display_values[variant][hadm_id] for variant in ["A", "B", "C"]}
            if len(set(str(value) for value in values.values())) <= 1:
                continue
            outlier = _choose_outlier_variant(values)
            outlier_key = outlier if outlier is not None else "A"
            reasoning = _reasoning_excerpt(
                str(by_variant[outlier_key][hadm_id].get("reasoning", ""))
            )
            disagreements.append(
                {
                    "hadm_id": hadm_id,
                    "A_vote": values["A"],
                    "B_vote": values["B"],
                    "C_vote": values["C"],
                    "outlier_variant": outlier or "",
                    "outlier_reasoning_excerpt": reasoning,
                }
            )
        disagreement_pool[field] = disagreements

    kappa_rows.sort(key=lambda row: float(row["kappa_mean"]))
    base_rate_rows.sort(
        key=lambda row: float(str(row["max_divergence_pct"]).rstrip("%")),
        reverse=True,
    )

    count_stats_rows: list[dict[str, Any]] = []
    for count_field in COUNT_FIELDS:
        abs_diffs: list[int] = []
        both_null = 0
        both_ge_1 = 0
        compared = 0
        for hadm_id in joined_hadm_ids:
            count_values = [
                by_variant[variant][hadm_id].get(count_field)
                for variant in ["A", "B", "C"]
            ]
            for left, right in combinations(count_values, 2):
                if left is None and right is None:
                    both_null += 1
                if left is not None and right is not None:
                    compared += 1
                    abs_diffs.append(abs(int(left) - int(right)))
                    if int(left) >= 1 and int(right) >= 1:
                        both_ge_1 += 1
        count_stats_rows.append(
            {
                "field": count_field,
                "median_abs_diff": float(np.median(abs_diffs)) if abs_diffs else 0.0,
                "pct_both_null": _pct(both_null / max(compared + both_null, 1)),
                "pct_both_ge_1": _pct(both_ge_1 / max(compared, 1)),
            }
        )

    concordance_rows = []
    for label in ["A", "B", "C"]:
        checked, concordant, rate = _dominant_concordance(split_frame, by_variant[label])
        concordance_rows.append(
            {
                "variant": label,
                "n_checked": checked,
                "n_concordant": concordant,
                "concordance_rate": _pct(rate),
            }
        )

    run_rows = []
    token_rows = []
    total_cost = 0.0
    for label, entry in zip(["A", "B", "C"], loaded, strict=True):
        input_cost = (entry["input_tokens"] / 1_000_000) * config.INPUT_PRICE_PER_MILLION_USD
        output_cost = (entry["output_tokens"] / 1_000_000) * config.OUTPUT_PRICE_PER_MILLION_USD
        run_cost = input_cost + output_cost
        total_cost += run_cost

        run_rows.append(
            {
                "variant": label,
                "run_id": entry["run_id"],
                "results_mtime_utc": entry["mtime_utc"],
                "n_attempted": entry["n_attempted"],
                "n_parsed": entry["n_parsed"],
            }
        )

        input_vector = np.asarray(
            [int(item.get("input_tokens", 0) or 0) for item in entry["records"]],
            dtype=np.float64,
        )
        output_vector = np.asarray(
            [int(item.get("output_tokens", 0) or 0) for item in entry["records"]],
            dtype=np.float64,
        )
        median_input = float(np.median(input_vector)) if input_vector.size else 0.0
        median_output = float(np.median(output_vector)) if output_vector.size else 0.0
        per_note_cost = (
            (median_input / 1_000_000) * config.INPUT_PRICE_PER_MILLION_USD
            + (median_output / 1_000_000) * config.OUTPUT_PRICE_PER_MILLION_USD
        )
        token_rows.append(
            {
                "variant": label,
                "median_input_tokens": median_input,
                "median_output_tokens": median_output,
                "per_note_cost_usd": per_note_cost,
                "run_cost_usd": run_cost,
            }
        )

    all_kappas = np.asarray([float(row["kappa_mean"]) for row in kappa_rows], dtype=np.float64)
    top_line_all = _band_counts(all_kappas)

    filtered_rows = [row for row in kappa_rows if not bool(row["low_base_rate_flag"])]
    filtered_kappas = np.asarray(
        [float(row["kappa_mean"]) for row in filtered_rows],
        dtype=np.float64,
    )
    top_line_filtered = _band_counts(filtered_kappas)
    top_line_filtered["n_fields_included"] = int(filtered_kappas.size)

    low_base_rows = [row for row in kappa_rows if bool(row["low_base_rate_flag"])]

    class_df = pd.DataFrame(class_rows)
    class_summary_rows = []
    if not class_df.empty:
        class_summary = class_df.groupby("field_group", as_index=False)["mean_kappa"].mean()
        class_summary = class_summary.sort_values("mean_kappa", ascending=True)
        class_summary_rows = class_summary.to_dict(orient="records")

    worst_fields = [str(row["field"]) for row in kappa_rows[:5]]
    disagreement_lines: list[str] = []
    for field in worst_fields:
        disagreement_lines.append(f"### {field}")
        disagreement_lines.append("")
        rows = disagreement_pool.get(field, [])[:5]
        disagreement_lines.append(
            _markdown_table(
                rows,
                [
                    "hadm_id",
                    "A_vote",
                    "B_vote",
                    "C_vote",
                    "outlier_variant",
                    "outlier_reasoning_excerpt",
                ],
            )
        )
        disagreement_lines.append("")

    output_lines = [
        "# Three-Way Kappa Report",
        "",
        "## Run metadata",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_ids": ", ".join(args.run_ids),
                    "split": split_name,
                    "intersection_parsed_all_three": len(joined_hadm_ids),
                    "total_cost_usd": total_cost,
                }
            ],
            [
                "timestamp_utc",
                "run_ids",
                "split",
                "intersection_parsed_all_three",
                "total_cost_usd",
            ],
        ),
        "",
        "## Notes parsed",
        _markdown_table(
            run_rows,
            ["variant", "run_id", "results_mtime_utc", "n_attempted", "n_parsed"],
        ),
        "",
        "## Token cost per variant",
        _markdown_table(
            token_rows,
            [
                "variant",
                "median_input_tokens",
                "median_output_tokens",
                "per_note_cost_usd",
                "run_cost_usd",
            ],
        ),
        "",
        "## Pairwise kappa table",
        _markdown_table(
            kappa_rows,
            [
                "field",
                "field_group",
                "kappa_A_B",
                "kappa_A_C",
                "kappa_B_C",
                "kappa_mean",
                "pabak_mean",
                "pct_agreement_mean",
                "n_positive_a",
                "n_positive_b",
                "n_positive_c",
                "n_positive_total",
                "low_base_rate_flag",
                "flag_lt_0_6",
            ],
        ),
        "",
        "## Kappa summary (all fields)",
        _markdown_table([top_line_all], list(top_line_all.keys())),
        "",
        "## Kappa summary (filtered, excluding low-base-rate fields)",
        _markdown_table([top_line_filtered], list(top_line_filtered.keys())),
        "",
        "## Low-base-rate fields (excluded from filtered summary)",
        _markdown_table(
            low_base_rows,
            [
                "field",
                "field_group",
                "n_positive_total",
                "kappa_mean",
                "pabak_mean",
                "pct_agreement_mean",
            ],
        ),
        "",
        "## Field-class summary",
        _markdown_table(class_summary_rows, ["field_group", "mean_kappa"]),
        "",
        "## Base-rate divergence table",
        _markdown_table(
            base_rate_rows,
            ["field", "value", "A_pct", "B_pct", "C_pct", "max_divergence_pct"],
        ),
        "",
        "## Count-field disagreement stats (non-kappa)",
        _markdown_table(
            count_stats_rows,
            ["field", "median_abs_diff", "pct_both_null", "pct_both_ge_1"],
        ),
        "",
        "## Five worst-kappa-field disagreement audit",
        *disagreement_lines,
        "## ICD concordance per variant",
        _markdown_table(
            concordance_rows,
            ["variant", "n_checked", "n_concordant", "concordance_rate"],
        ),
        "",
        (
            "Filtered median kappa excludes fields where "
            "n_positive_total < 10 (low_base_rate_flag=True)."
        ),
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    kappa_results = {
        str(row["field"]): {
            "field_group": str(row["field_group"]),
            "kappa_A_B": float(row["kappa_A_B"]),
            "kappa_A_C": float(row["kappa_A_C"]),
            "kappa_B_C": float(row["kappa_B_C"]),
            "kappa_mean": float(row["kappa_mean"]),
            "pabak_mean": float(row["pabak_mean"]),
            "pct_agreement_mean": float(row["pct_agreement_mean"]),
            "n_positive_a": int(row["n_positive_a"]),
            "n_positive_b": int(row["n_positive_b"]),
            "n_positive_c": int(row["n_positive_c"]),
            "n_positive_total": int(row["n_positive_total"]),
            "low_base_rate_flag": bool(row["low_base_rate_flag"]),
        }
        for row in kappa_rows
    }
    sidecar_payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_ids": args.run_ids,
        "split": split_name,
        "intersection_parsed_all_three": len(joined_hadm_ids),
        "kappa_results": kappa_results,
        "kappa_summary_all": top_line_all,
        "kappa_summary_filtered": top_line_filtered,
        "low_base_rate_fields": [str(row["field"]) for row in low_base_rows],
    }
    sidecar_path = Path(f"{output_path}.json")
    sidecar_path.write_text(
        json.dumps(sidecar_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Wrote three-way kappa report to {output_path}")
    print(f"Wrote three-way kappa sidecar JSON to {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
