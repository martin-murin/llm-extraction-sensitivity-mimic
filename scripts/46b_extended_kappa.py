"""
Runs staged pipeline step `46b_extended_kappa.py`.

Reads: data/splits/extended_5k.csv, codex_outputs/46_extended_kappa_report.md, codex_outputs/46_kappa_5way_comparison.md, data/optimization/audit_corpus_extended_5k.jsonl, codex_outputs/46_extended_audit_clusters.md, codex_outputs/46_verification.md.
Writes: data/splits/extended_5k.csv, codex_outputs/46_extended_kappa_report.md, codex_outputs/46_kappa_5way_comparison.md, data/optimization/audit_corpus_extended_5k.jsonl, codex_outputs/46_extended_audit_clusters.md, codex_outputs/46_verification.md.
Backs extended-sample kappa and stability claims.
Usage: `python scripts/46b_extended_kappa.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src import config
from src.optimization.audit_corpus import (
    classify_vote_pattern,
    has_any_disagreement,
    select_representative_examples,
    should_include_record,
    summarize_disagreement_pattern,
)
from src.optimization.pattern_clustering import cluster_corpus
from src.schema.fields import LLMNoteFeatures
from src.schema.vocabulary import ADMISSION_REASON_TAGS
from src.utils.diagnostic_plots import plot_kappa_with_bootstrap_ci
from src.utils.threeway_kappa import (
    cohen_kappa_safe,
    count_positive_admission_tag,
    count_positive_enum,
    count_positive_tristate,
    encode_admission_reason_tags,
    encode_tristate,
    file_sha256,
    low_base_rate_flag,
    pabak_score,
    percent_agreement,
)

PAIR_KEYS = [("A", "B"), ("A", "C"), ("B", "C")]
TRISTATE_FIELDS = [
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
]
ENUM_FIELDS = ["functional_status", "mental_status", "discharge_condition_category"]
COUNT_FIELDS = ["new_meds_started_count", "meds_stopped_count"]
SKIP_KAPPA_FIELDS = {"primary_diagnosis_text", "reasoning", *COUNT_FIELDS}

BASELINE_SAMPLES: dict[str, tuple[str, str, str]] = {
    "refinement_150": ("refinement_v1_a", "refinement_v1_b", "refinement_v1_c"),
    "holdout_150": ("holdout_v1_a", "holdout_v1_b", "holdout_v1_c"),
    "methodology_1k": ("methodology_1k_a", "methodology_1k_b", "methodology_1k_c"),
    "methodology_5k_audit_500": (
        "methodology_5k_a_subset500",
        "methodology_5k_audit_b",
        "methodology_5k_audit_c",
    ),
    "extended_5k": ("production_v1", "extended_5k_b", "extended_5k_c"),
}


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return f"{value:.4f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None:
        return ""
    return str(value)


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


def _load_results(
    run_id: str,
    *,
    keep_hadm_ids: set[int] | None = None,
) -> dict[str, Any]:
    results_path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    parsed: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0

    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            hadm_id = int(row["hadm_id"])
            if keep_hadm_ids is not None and hadm_id not in keep_hadm_ids:
                continue

            records.append(row)
            input_tokens += int(row.get("input_tokens", 0) or 0)
            output_tokens += int(row.get("output_tokens", 0) or 0)

            if bool(row.get("parse_ok", False)) and isinstance(row.get("features_json"), dict):
                parsed[hadm_id] = row["features_json"]

    return {
        "run_id": run_id,
        "results_path": results_path,
        "mtime_utc": datetime.fromtimestamp(results_path.stat().st_mtime, tz=UTC).isoformat(),
        "n_attempted": len(records),
        "n_parsed": len(parsed),
        "parsed": parsed,
        "records": records,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
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
    return " ".join(str(text).split())[:max_chars]


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


def _field_class(field_key: str) -> str:
    if field_key.startswith("admission_reason_tags::"):
        return "admission_tags"
    if field_key == "dominant_admission_reason":
        return "dominant_admission_reason"
    if field_key in set(TRISTATE_FIELDS):
        return "tristates"
    if field_key in set(ENUM_FIELDS):
        return "enums"
    return "other"


def _load_sidecar(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sidecar: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("kappa_results"), dict):
        raise RuntimeError(f"Invalid sidecar format: {path}")
    return payload


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _load_split_hadm_ids(split_path: Path) -> list[int]:
    frame = pd.read_csv(split_path)
    return sorted(pd.to_numeric(frame["hadm_id"], errors="coerce").astype("int64").tolist())


def _build_cases_and_examples(
    *,
    hadm_ids: list[int],
    votes_by_hadm: dict[int, dict[str, str]],
    parsed_by_variant: dict[str, dict[int, dict[str, Any]]],
    icd_info: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    disagreements: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for hadm_id in hadm_ids:
        votes = votes_by_hadm[hadm_id]
        if len({votes["a"], votes["b"], votes["c"]}) <= 1:
            continue

        pattern = classify_vote_pattern(votes)
        disagreements.append({"hadm_id": hadm_id, "votes": votes})

        outlier = pattern.outlier_variant
        outlier_reasoning = ""
        consensus_reasoning: dict[str, str] = {}
        if outlier is not None:
            outlier_reasoning = _reasoning_excerpt(
                str(parsed_by_variant[outlier][hadm_id].get("reasoning", "")),
                max_chars=600,
            )
            for variant in ["a", "b", "c"]:
                if variant == outlier:
                    continue
                consensus_reasoning[variant] = _reasoning_excerpt(
                    str(parsed_by_variant[variant][hadm_id].get("reasoning", "")),
                    max_chars=400,
                )

        info = icd_info.get(hadm_id, {})
        cases.append(
            {
                "hadm_id": hadm_id,
                "votes": votes,
                "outlier_variant": outlier,
                "primary_icd_code": str(info.get("primary_icd_code", "")),
                "outlier_reasoning_excerpt": outlier_reasoning,
                "consensus_reasoning_excerpts": consensus_reasoning,
                "pattern_key": pattern.pattern_key,
                "chapter": str(info.get("chapter", "")),
            }
        )

    summary = summarize_disagreement_pattern(disagreements)
    examples = select_representative_examples(cases, max_examples=10)
    cleaned_examples = [
        {
            "hadm_id": item["hadm_id"],
            "votes": item["votes"],
            "outlier_variant": item["outlier_variant"],
            "primary_icd_code": item["primary_icd_code"],
            "outlier_reasoning_excerpt": item["outlier_reasoning_excerpt"],
            "consensus_reasoning_excerpts": item["consensus_reasoning_excerpts"],
        }
        for item in examples
    ]
    return disagreements, cleaned_examples, summary


def _build_audit_records(
    *,
    hadm_ids: list[int],
    parsed_by_variant: dict[str, dict[int, dict[str, Any]]],
    icd_info: dict[int, dict[str, Any]],
    include_low_base_rate: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for field in TRISTATE_FIELDS:
        values = {
            variant: [
                str(parsed_by_variant[variant][hadm_id].get(field, "not_documented"))
                for hadm_id in hadm_ids
            ]
            for variant in ["a", "b", "c"]
        }
        vote_tuples = list(zip(values["a"], values["b"], values["c"], strict=True))
        if not has_any_disagreement(vote_tuples):
            continue

        encoded = {
            variant: [encode_tristate(value) for value in vals] for variant, vals in values.items()
        }
        metrics = _pairwise_metrics({"A": encoded["a"], "B": encoded["b"], "C": encoded["c"]})

        votes_by_hadm = {
            hadm_id: {"a": values["a"][idx], "b": values["b"][idx], "c": values["c"][idx]}
            for idx, hadm_id in enumerate(hadm_ids)
        }
        disagreements, examples, summary = _build_cases_and_examples(
            hadm_ids=hadm_ids,
            votes_by_hadm=votes_by_hadm,
            parsed_by_variant=parsed_by_variant,
            icd_info=icd_info,
        )

        n_positive_a = count_positive_tristate(values["a"])
        n_positive_b = count_positive_tristate(values["b"])
        n_positive_c = count_positive_tristate(values["c"])
        n_positive_total = n_positive_a + n_positive_b + n_positive_c

        records.append(
            {
                "field": field,
                "target_value": None,
                "field_type": "tristate",
                "kappa_a_b": metrics["kappa_A_B"],
                "kappa_a_c": metrics["kappa_A_C"],
                "kappa_b_c": metrics["kappa_B_C"],
                "kappa_mean": metrics["kappa_mean"],
                "pabak_mean": metrics["pabak_mean"],
                "pct_agreement_mean": metrics["pct_agreement_mean"],
                "n_positive_a": n_positive_a,
                "n_positive_b": n_positive_b,
                "n_positive_c": n_positive_c,
                "n_positive_total": n_positive_total,
                "low_base_rate_flag": n_positive_total < 10,
                "disagreement_count": len(disagreements),
                "disagreement_pattern_summary": summary,
                "examples": examples,
            }
        )

    for field in [*ENUM_FIELDS, "dominant_admission_reason"]:
        values = {
            variant: [
                str(parsed_by_variant[variant][hadm_id].get(field, "not_documented"))
                for hadm_id in hadm_ids
            ]
            for variant in ["a", "b", "c"]
        }
        vote_tuples = list(zip(values["a"], values["b"], values["c"], strict=True))
        if not has_any_disagreement(vote_tuples):
            continue

        observed = sorted({value for vals in values.values() for value in vals})
        mapping = {value: idx for idx, value in enumerate(observed)}
        encoded = {variant: [mapping[value] for value in vals] for variant, vals in values.items()}
        metrics = _pairwise_metrics({"A": encoded["a"], "B": encoded["b"], "C": encoded["c"]})

        votes_by_hadm = {
            hadm_id: {"a": values["a"][idx], "b": values["b"][idx], "c": values["c"][idx]}
            for idx, hadm_id in enumerate(hadm_ids)
        }
        disagreements, examples, summary = _build_cases_and_examples(
            hadm_ids=hadm_ids,
            votes_by_hadm=votes_by_hadm,
            parsed_by_variant=parsed_by_variant,
            icd_info=icd_info,
        )

        n_positive_a = count_positive_enum(values["a"])
        n_positive_b = count_positive_enum(values["b"])
        n_positive_c = count_positive_enum(values["c"])
        n_positive_total = n_positive_a + n_positive_b + n_positive_c

        records.append(
            {
                "field": field,
                "target_value": None,
                "field_type": "enum"
                if field != "dominant_admission_reason"
                else "dominant_admission_reason",
                "kappa_a_b": metrics["kappa_A_B"],
                "kappa_a_c": metrics["kappa_A_C"],
                "kappa_b_c": metrics["kappa_B_C"],
                "kappa_mean": metrics["kappa_mean"],
                "pabak_mean": metrics["pabak_mean"],
                "pct_agreement_mean": metrics["pct_agreement_mean"],
                "n_positive_a": n_positive_a,
                "n_positive_b": n_positive_b,
                "n_positive_c": n_positive_c,
                "n_positive_total": n_positive_total,
                "low_base_rate_flag": n_positive_total < 10,
                "disagreement_count": len(disagreements),
                "disagreement_pattern_summary": summary,
                "examples": examples,
            }
        )

    tags_observed = set()
    for variant in ["a", "b", "c"]:
        for hadm_id in hadm_ids:
            tags_observed.update(
                parsed_by_variant[variant][hadm_id].get("admission_reason_tags", [])
            )

    for tag in ADMISSION_REASON_TAGS:
        if tag not in tags_observed:
            continue

        tag_votes: dict[str, list[str]] = {"a": [], "b": [], "c": []}
        lists_by_variant: dict[str, list[list[str]]] = {"a": [], "b": [], "c": []}
        for variant in ["a", "b", "c"]:
            for hadm_id in hadm_ids:
                tags = list(parsed_by_variant[variant][hadm_id].get("admission_reason_tags", []))
                lists_by_variant[variant].append(tags)
                tag_votes[variant].append("present" if tag in set(tags) else "absent")

        vote_tuples = list(zip(tag_votes["a"], tag_votes["b"], tag_votes["c"], strict=True))
        if not has_any_disagreement(vote_tuples):
            continue

        encoded = {
            variant: [1 if value == "present" else 0 for value in vals]
            for variant, vals in tag_votes.items()
        }
        metrics = _pairwise_metrics({"A": encoded["a"], "B": encoded["b"], "C": encoded["c"]})

        votes_by_hadm = {
            hadm_id: {
                "a": tag_votes["a"][idx],
                "b": tag_votes["b"][idx],
                "c": tag_votes["c"][idx],
            }
            for idx, hadm_id in enumerate(hadm_ids)
        }
        disagreements, examples, summary = _build_cases_and_examples(
            hadm_ids=hadm_ids,
            votes_by_hadm=votes_by_hadm,
            parsed_by_variant=parsed_by_variant,
            icd_info=icd_info,
        )

        n_positive_a = count_positive_admission_tag(lists_by_variant["a"], tag)
        n_positive_b = count_positive_admission_tag(lists_by_variant["b"], tag)
        n_positive_c = count_positive_admission_tag(lists_by_variant["c"], tag)
        n_positive_total = n_positive_a + n_positive_b + n_positive_c

        records.append(
            {
                "field": "admission_reason_tags",
                "target_value": tag,
                "field_type": "admission_tag",
                "kappa_a_b": metrics["kappa_A_B"],
                "kappa_a_c": metrics["kappa_A_C"],
                "kappa_b_c": metrics["kappa_B_C"],
                "kappa_mean": metrics["kappa_mean"],
                "pabak_mean": metrics["pabak_mean"],
                "pct_agreement_mean": metrics["pct_agreement_mean"],
                "n_positive_a": n_positive_a,
                "n_positive_b": n_positive_b,
                "n_positive_c": n_positive_c,
                "n_positive_total": n_positive_total,
                "low_base_rate_flag": n_positive_total < 10,
                "disagreement_count": len(disagreements),
                "disagreement_pattern_summary": summary,
                "examples": examples,
            }
        )

    emitted: list[dict[str, Any]] = []
    for record in records:
        if should_include_record(
            int(record["n_positive_total"]),
            include_low_base_rate=include_low_base_rate,
        ):
            emitted.append(record)

    emitted.sort(
        key=lambda row: (
            int(row["disagreement_count"]),
            -float(row["kappa_mean"]),
            str(row["field"]),
            str(row.get("target_value") or ""),
        ),
        reverse=True,
    )
    return emitted


def _extract_sample_votes(
    *,
    field_key: str,
    sample_maps: dict[str, dict[str, dict[int, dict[str, Any]]]],
    sample_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maps = sample_maps[sample_name]
    shared = sorted(set(maps["A"]).intersection(maps["B"]).intersection(maps["C"]))
    if not shared:
        return np.asarray([]), np.asarray([]), np.asarray([])

    def encode(features: dict[str, Any]) -> int | str:
        if field_key.startswith("admission_reason_tags::"):
            tag = field_key.split("::", maxsplit=1)[1]
            tags = features.get("admission_reason_tags", [])
            return int(tag in set(tags) if isinstance(tags, list) else False)
        if field_key in set(TRISTATE_FIELDS):
            return encode_tristate(str(features.get(field_key, "not_documented")))
        if field_key == "dominant_admission_reason":
            return str(features.get(field_key, "other"))
        return str(features.get(field_key, "not_documented"))

    a_vals = [encode(maps["A"][hadm_id]) for hadm_id in shared]
    b_vals = [encode(maps["B"][hadm_id]) for hadm_id in shared]
    c_vals = [encode(maps["C"][hadm_id]) for hadm_id in shared]

    if any(not isinstance(v, (int, np.integer)) for v in a_vals + b_vals + c_vals):
        labels = sorted({str(v) for v in a_vals + b_vals + c_vals})
        mapping = {label: idx for idx, label in enumerate(labels)}
        return (
            np.asarray([mapping[str(v)] for v in a_vals], dtype=np.int64),
            np.asarray([mapping[str(v)] for v in b_vals], dtype=np.int64),
            np.asarray([mapping[str(v)] for v in c_vals], dtype=np.int64),
        )

    return (
        np.asarray(a_vals, dtype=np.int64),
        np.asarray(b_vals, dtype=np.int64),
        np.asarray(c_vals, dtype=np.int64),
    )


def _kappa_mean(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return (cohen_kappa_safe(a, b) + cohen_kappa_safe(a, c) + cohen_kappa_safe(b, c)) / 3.0


def _bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    n_iter: int,
    seed: int,
) -> tuple[float, float, float]:
    if a.size == 0:
        return 0.0, 0.0, 0.0

    base = _kappa_mean(a, b, c)
    rng = np.random.default_rng(seed)
    n = a.size
    scores = np.zeros(n_iter, dtype=np.float64)
    for idx in range(n_iter):
        sample_idx = rng.integers(0, n, size=n)
        scores[idx] = _kappa_mean(a[sample_idx], b[sample_idx], c[sample_idx])

    lo = float(np.percentile(scores, 2.5))
    hi = float(np.percentile(scores, 97.5))
    return base, lo, hi


def _load_total_cost_usd(run_id: str) -> float:
    path = config.LOGS_DIR / "runs" / f"{run_id}_cost.json"
    if not path.exists():
        return 0.0
    last_cost = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            last_cost = float(row.get("total_cost_usd", last_cost) or last_cost)
    return last_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended 5k kappa + audit + 5-way comparison.")
    parser.add_argument("--split", default="data/splits/extended_5k.csv")
    parser.add_argument("--run-id-a", default="production_v1")
    parser.add_argument("--run-id-b", default="extended_5k_b")
    parser.add_argument("--run-id-c", default="extended_5k_c")
    parser.add_argument("--output-kappa", default="codex_outputs/46_extended_kappa_report.md")
    parser.add_argument("--output-compare", default="codex_outputs/46_kappa_5way_comparison.md")
    parser.add_argument(
        "--audit-output", default="data/optimization/audit_corpus_extended_5k.jsonl"
    )
    parser.add_argument("--cluster-output", default="codex_outputs/46_extended_audit_clusters.md")
    parser.add_argument("--verification", default="codex_outputs/46_verification.md")
    parser.add_argument("--figure-output", default="docs/figures/46_kappa_with_ci_5_samples.png")
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--include-low-base-rate", action="store_true", default=False)
    parser.add_argument("--hard-cap-usd", type=float, default=30.0)
    parser.add_argument("--kappa-refinement", default="codex_outputs/16c_iter2_kappa.md.json")
    parser.add_argument("--kappa-holdout", default="codex_outputs/21_holdout_kappa_report.md.json")
    parser.add_argument(
        "--kappa-1k", default="codex_outputs/22_methodology_1k_kappa_report.md.json"
    )
    parser.add_argument(
        "--kappa-5k-audit",
        default="codex_outputs/26_methodology_5k_audit_kappa_report.md.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    split_path = Path(args.split)
    if not split_path.exists():
        raise FileNotFoundError(f"Missing extended split: {split_path}")

    split_frame = pd.read_csv(split_path)
    split_frame["hadm_id"] = pd.to_numeric(split_frame["hadm_id"], errors="coerce").astype("int64")
    split_hadm_ids = sorted(split_frame["hadm_id"].astype(int).tolist())
    split_hadm_id_set = set(split_hadm_ids)

    loaded_a = _load_results(str(args.run_id_a), keep_hadm_ids=split_hadm_id_set)
    loaded_b = _load_results(str(args.run_id_b))
    loaded_c = _load_results(str(args.run_id_c))

    by_variant = {
        "A": loaded_a["parsed"],
        "B": loaded_b["parsed"],
        "C": loaded_c["parsed"],
    }
    joined_hadm_ids = sorted(
        set(by_variant["A"]).intersection(by_variant["B"]).intersection(by_variant["C"])
    )

    kappa_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    base_rate_rows: list[dict[str, Any]] = []
    disagreement_pool: dict[str, list[dict[str, Any]]] = {}

    field_names = [
        field for field in LLMNoteFeatures.model_fields if field not in SKIP_KAPPA_FIELDS
    ]
    for field in field_names:
        if field == "admission_reason_tags":
            tag_scores: list[float] = []
            for tag in ADMISSION_REASON_TAGS:
                encoded: dict[str, list[int]] = {}
                tag_lists_by_variant: dict[str, list[list[str]]] = {"A": [], "B": [], "C": []}
                for variant in ["A", "B", "C"]:
                    vals: list[int] = []
                    for hadm_id in joined_hadm_ids:
                        tags = list(by_variant[variant][hadm_id].get("admission_reason_tags", []))
                        tag_lists_by_variant[variant].append(tags)
                        vals.append(encode_admission_reason_tags(tags)[tag])
                    encoded[variant] = vals

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

                tag_disagreements: list[dict[str, Any]] = []
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
                    tag_disagreements.append(
                        {
                            "hadm_id": hadm_id,
                            "A_vote": "yes" if values["A"] else "no",
                            "B_vote": "yes" if values["B"] else "no",
                            "C_vote": "yes" if values["C"] else "no",
                            "outlier_variant": outlier or "",
                            "outlier_reasoning_excerpt": reasoning,
                        }
                    )
                disagreement_pool[row_name] = tag_disagreements

            class_rows.append(
                {
                    "field_group": "admission_reason_tags",
                    "mean_kappa": float(np.mean(tag_scores)) if tag_scores else 0.0,
                }
            )
            continue

        encoded_map: dict[str, list[int]] = {}
        display_values: dict[str, dict[int, Any]] = {"A": {}, "B": {}, "C": {}}

        if field in set(TRISTATE_FIELDS):
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

            observed_values = sorted(
                {
                    str(by_variant[variant][hadm_id].get(field, "not_documented"))
                    for variant in ["A", "B", "C"]
                    for hadm_id in joined_hadm_ids
                }
            )
            for value in observed_values:
                rates = {}
                for variant in ["A", "B", "C"]:
                    hits = sum(
                        1
                        for hadm_id in joined_hadm_ids
                        if str(by_variant[variant][hadm_id].get(field, "not_documented")) == value
                    )
                    rates[variant] = hits / len(joined_hadm_ids) if joined_hadm_ids else 0.0
                base_rate_rows.append(
                    {
                        "field": field,
                        "value": value,
                        "A_pct": f"{rates['A'] * 100:.2f}%",
                        "B_pct": f"{rates['B'] * 100:.2f}%",
                        "C_pct": f"{rates['C'] * 100:.2f}%",
                        "max_divergence_pct": (
                            f"{(max(rates.values()) - min(rates.values())) * 100:.2f}%"
                        ),
                    }
                )

        elif field in set(ENUM_FIELDS):
            field_group = "enum"
            positives = {"A": 0, "B": 0, "C": 0}
            observed = sorted(
                {
                    str(by_variant[variant][hadm_id].get(field, "not_documented"))
                    for variant in ["A", "B", "C"]
                    for hadm_id in joined_hadm_ids
                }
            )
            mapping = {value: idx for idx, value in enumerate(observed)}
            for variant in ["A", "B", "C"]:
                enum_values = [
                    str(by_variant[variant][hadm_id].get(field, "not_documented"))
                    for hadm_id in joined_hadm_ids
                ]
                positives[variant] = count_positive_enum(enum_values)
                encoded_map[variant] = [mapping[value] for value in enum_values]
                for hadm_id, value in zip(joined_hadm_ids, enum_values, strict=True):
                    display_values[variant][hadm_id] = value

        elif field == "dominant_admission_reason":
            field_group = "dominant_admission_reason"
            positives = {
                "A": len(joined_hadm_ids),
                "B": len(joined_hadm_ids),
                "C": len(joined_hadm_ids),
            }
            mapping = {tag: idx for idx, tag in enumerate(ADMISSION_REASON_TAGS)}
            for variant in ["A", "B", "C"]:
                dominant_values = [
                    str(by_variant[variant][hadm_id].get(field, "other"))
                    for hadm_id in joined_hadm_ids
                ]
                encoded_map[variant] = [
                    mapping.get(value, mapping["other"]) for value in dominant_values
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

        field_disagreements: list[dict[str, Any]] = []
        for hadm_id in joined_hadm_ids:
            values = {variant: display_values[variant][hadm_id] for variant in ["A", "B", "C"]}
            if len(set(str(v) for v in values.values())) <= 1:
                continue
            outlier = _choose_outlier_variant(values)
            outlier_key = outlier if outlier is not None else "A"
            reasoning = _reasoning_excerpt(
                str(by_variant[outlier_key][hadm_id].get("reasoning", ""))
            )
            field_disagreements.append(
                {
                    "hadm_id": hadm_id,
                    "A_vote": values["A"],
                    "B_vote": values["B"],
                    "C_vote": values["C"],
                    "outlier_variant": outlier or "",
                    "outlier_reasoning_excerpt": reasoning,
                }
            )
        disagreement_pool[field] = field_disagreements

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
                by_variant[variant][hadm_id].get(count_field) for variant in ["A", "B", "C"]
            ]
            pairs = [
                (count_values[0], count_values[1]),
                (count_values[0], count_values[2]),
                (count_values[1], count_values[2]),
            ]
            for left, right in pairs:
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
                "pct_both_null": f"{(both_null / max(compared + both_null, 1)) * 100:.2f}%",
                "pct_both_ge_1": f"{(both_ge_1 / max(compared, 1)) * 100:.2f}%",
            }
        )

    run_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    total_cost = 0.0
    for label, entry in zip(["A", "B", "C"], [loaded_a, loaded_b, loaded_c], strict=True):
        input_cost = (entry["input_tokens"] / 1_000_000.0) * config.INPUT_PRICE_PER_MILLION_USD
        output_cost = (entry["output_tokens"] / 1_000_000.0) * config.OUTPUT_PRICE_PER_MILLION_USD
        run_cost = input_cost + output_cost
        total_cost += run_cost

        run_rows.append(
            {
                "variant": label,
                "run_id": entry["run_id"],
                "results_mtime_utc": entry["mtime_utc"],
                "n_attempted": int(entry["n_attempted"]),
                "n_parsed": int(entry["n_parsed"]),
            }
        )

        input_vector = np.asarray(
            [int(item.get("input_tokens", 0) or 0) for item in entry["records"]], dtype=np.float64
        )
        output_vector = np.asarray(
            [int(item.get("output_tokens", 0) or 0) for item in entry["records"]], dtype=np.float64
        )
        median_input = float(np.median(input_vector)) if input_vector.size else 0.0
        median_output = float(np.median(output_vector)) if output_vector.size else 0.0
        per_note_cost = (median_input / 1_000_000.0) * config.INPUT_PRICE_PER_MILLION_USD + (
            median_output / 1_000_000.0
        ) * config.OUTPUT_PRICE_PER_MILLION_USD
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
        [float(row["kappa_mean"]) for row in filtered_rows], dtype=np.float64
    )
    top_line_filtered = _band_counts(filtered_kappas)
    top_line_filtered["n_fields_included"] = int(filtered_kappas.size)

    low_base_rows = [row for row in kappa_rows if bool(row["low_base_rate_flag"])]

    class_df = pd.DataFrame(class_rows)
    class_summary_rows: list[dict[str, Any]] = []
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

    output_kappa_path = Path(args.output_kappa)
    output_kappa_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines = [
        "# Extended Three-Way Kappa Report (5k)",
        "",
        "## Run metadata",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_ids": f"{args.run_id_a}, {args.run_id_b}, {args.run_id_c}",
                    "split": str(split_path),
                    "split_n": len(split_hadm_ids),
                    "intersection_parsed_all_three": len(joined_hadm_ids),
                    "total_cost_usd": total_cost,
                }
            ],
            [
                "timestamp_utc",
                "run_ids",
                "split",
                "split_n",
                "intersection_parsed_all_three",
                "total_cost_usd",
            ],
        ),
        "",
        "## Notes parsed",
        _markdown_table(
            run_rows, ["variant", "run_id", "results_mtime_utc", "n_attempted", "n_parsed"]
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
            base_rate_rows, ["field", "value", "A_pct", "B_pct", "C_pct", "max_divergence_pct"]
        ),
        "",
        "## Count-field disagreement stats (non-kappa)",
        _markdown_table(
            count_stats_rows, ["field", "median_abs_diff", "pct_both_null", "pct_both_ge_1"]
        ),
        "",
        "## Five worst-kappa-field disagreement audit",
        *disagreement_lines,
        (
            "Filtered median kappa excludes fields where "
            "n_positive_total < 10 (low_base_rate_flag=True)."
        ),
        "",
    ]
    output_kappa_path.write_text("\n".join(output_lines), encoding="utf-8")

    sidecar_payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_ids": [args.run_id_a, args.run_id_b, args.run_id_c],
        "split": str(split_path),
        "intersection_parsed_all_three": len(joined_hadm_ids),
        "kappa_results": {
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
        },
        "kappa_summary_all": top_line_all,
        "kappa_summary_filtered": top_line_filtered,
        "low_base_rate_fields": [str(row["field"]) for row in low_base_rows],
    }
    extended_sidecar_path = Path(f"{output_kappa_path}.json")
    extended_sidecar_path.write_text(
        json.dumps(sidecar_payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    # Build extended audit corpus + cluster report.
    parsed_by_variant_lower = {
        "a": by_variant["A"],
        "b": by_variant["B"],
        "c": by_variant["C"],
    }
    icd_info = {
        int(row["hadm_id"]): {
            "primary_icd_code": str(row.get("primary_icd_code", "")),
            "chapter": str(row.get("chapter", "")),
        }
        for _, row in split_frame.iterrows()
    }

    audit_records = _build_audit_records(
        hadm_ids=joined_hadm_ids,
        parsed_by_variant=parsed_by_variant_lower,
        icd_info=icd_info,
        include_low_base_rate=bool(args.include_low_base_rate),
    )

    audit_output_path = Path(args.audit_output)
    audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_output_path.open("w", encoding="utf-8") as handle:
        for record in audit_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    clusters = cluster_corpus(audit_records)
    cluster_rows = [
        {
            "cluster_id": c.cluster_id,
            "cluster_label": c.cluster_label,
            "affected_variant": c.affected_variant or "",
            "total_disagreements": c.total_disagreement_count,
            "n_member_fields": len(c.member_fields),
        }
        for c in clusters
    ]
    top_cluster_member_rows: list[dict[str, Any]] = []
    if clusters:
        for member in clusters[0].member_fields[:12]:
            top_cluster_member_rows.append(
                {
                    "field": member.get("field", ""),
                    "target_value": member.get("target_value", ""),
                    "kappa_mean": member.get("kappa_mean", 0.0),
                    "disagreement_count": member.get("disagreement_count", 0),
                    "n_positive_total": member.get("n_positive_total", 0),
                }
            )

    cluster_output_path = Path(args.cluster_output)
    cluster_output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_lines = [
        "# Extended Audit Cluster Report",
        "",
        "## Metadata",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "audit_path": str(audit_output_path),
                    "records_emitted": len(audit_records),
                    "clusters": len(clusters),
                }
            ],
            ["timestamp_utc", "audit_path", "records_emitted", "clusters"],
        ),
        "",
        "## Cluster summary",
        _markdown_table(
            cluster_rows,
            [
                "cluster_id",
                "cluster_label",
                "affected_variant",
                "total_disagreements",
                "n_member_fields",
            ],
        ),
        "",
        "## Top cluster members",
        _markdown_table(
            top_cluster_member_rows,
            ["field", "target_value", "kappa_mean", "disagreement_count", "n_positive_total"],
        ),
        "",
    ]
    cluster_output_path.write_text("\n".join(cluster_lines), encoding="utf-8")

    # 5-way comparison report.
    refinement = _load_sidecar(Path(args.kappa_refinement))
    holdout = _load_sidecar(Path(args.kappa_holdout))
    meth_1k = _load_sidecar(Path(args.kappa_1k))
    meth_5k_audit = _load_sidecar(Path(args.kappa_5k_audit))
    extended = _load_sidecar(extended_sidecar_path)

    ref_results = refinement["kappa_results"]
    hold_results = holdout["kappa_results"]
    onek_results = meth_1k["kappa_results"]
    fivek_audit_results = meth_5k_audit["kappa_results"]
    ext_results = extended["kappa_results"]

    shared_keys = sorted(
        set(ref_results)
        & set(hold_results)
        & set(onek_results)
        & set(fivek_audit_results)
        & set(ext_results)
    )
    filtered_shared = [
        key
        for key in shared_keys
        if (
            not bool(ref_results[key].get("low_base_rate_flag", False))
            and not bool(hold_results[key].get("low_base_rate_flag", False))
            and not bool(onek_results[key].get("low_base_rate_flag", False))
            and not bool(fivek_audit_results[key].get("low_base_rate_flag", False))
            and not bool(ext_results[key].get("low_base_rate_flag", False))
        )
    ]

    refinement_median = float(
        refinement.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
    )
    holdout_median = float(
        holdout.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
    )
    meth_1k_median = float(
        meth_1k.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
    )
    meth_5k_audit_median = float(
        meth_5k_audit.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
    )
    extended_median = float(
        extended.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
    )
    top_rows: list[dict[str, Any]] = [
        {
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "refinement_filtered_median": refinement_median,
            "holdout_filtered_median": holdout_median,
            "methodology_1k_filtered_median": meth_1k_median,
            "methodology_5k_audit_filtered_median": meth_5k_audit_median,
            "extended_5k_filtered_median": extended_median,
            "delta_extended_vs_5k_audit_pp": (extended_median - meth_5k_audit_median) * 100.0,
            "delta_extended_vs_1k_pp": (extended_median - meth_1k_median) * 100.0,
        }
    ]

    class_names = sorted({_field_class(key) for key in filtered_shared})
    class_rows_5way: list[dict[str, Any]] = []
    for class_name in class_names:
        keys = [key for key in filtered_shared if _field_class(key) == class_name]
        ref_med = _median([float(ref_results[key]["kappa_mean"]) for key in keys])
        hold_med = _median([float(hold_results[key]["kappa_mean"]) for key in keys])
        onek_med = _median([float(onek_results[key]["kappa_mean"]) for key in keys])
        fivek_audit_med = _median([float(fivek_audit_results[key]["kappa_mean"]) for key in keys])
        ext_med = _median([float(ext_results[key]["kappa_mean"]) for key in keys])
        class_rows_5way.append(
            {
                "field_class": class_name,
                "n_fields": len(keys),
                "refinement_median": f"{ref_med:.4f}",
                "holdout_median": f"{hold_med:.4f}",
                "methodology_1k_median": f"{onek_med:.4f}",
                "methodology_5k_audit_median": f"{fivek_audit_med:.4f}",
                "extended_5k_median": f"{ext_med:.4f}",
                "delta_ext_vs_5k_audit_pp": f"{(ext_med - fivek_audit_med) * 100.0:.4f}",
            }
        )

    regression_rows: list[dict[str, Any]] = []
    for key in filtered_shared:
        ext_k = float(ext_results[key]["kappa_mean"])
        audit_k = float(fivek_audit_results[key]["kappa_mean"])
        delta = (ext_k - audit_k) * 100.0
        if delta < -5.0:
            regression_rows.append(
                {
                    "field": key,
                    "field_class": _field_class(key),
                    "kappa_5k_audit": f"{audit_k:.4f}",
                    "kappa_extended_5k": f"{ext_k:.4f}",
                    "delta_ext_vs_5k_audit_pp": f"{delta:.4f}",
                }
            )
    regression_rows.sort(key=lambda row: float(row["delta_ext_vs_5k_audit_pp"]))

    compare_path = Path(args.output_compare)
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    compare_lines = [
        "# Kappa 5-Way Comparison",
        "",
        "## Top-line filtered medians",
        _markdown_table(
            top_rows,
            [
                "timestamp_utc",
                "refinement_filtered_median",
                "holdout_filtered_median",
                "methodology_1k_filtered_median",
                "methodology_5k_audit_filtered_median",
                "extended_5k_filtered_median",
                "delta_extended_vs_5k_audit_pp",
                "delta_extended_vs_1k_pp",
            ],
        ),
        "",
        "## Per-field-class summary (filtered shared fields)",
        _markdown_table(
            class_rows_5way,
            [
                "field_class",
                "n_fields",
                "refinement_median",
                "holdout_median",
                "methodology_1k_median",
                "methodology_5k_audit_median",
                "extended_5k_median",
                "delta_ext_vs_5k_audit_pp",
            ],
        ),
        "",
        "## Fields where extended 5k kappa is >5 pp below 5k-audit",
        _markdown_table(
            regression_rows,
            [
                "field",
                "field_class",
                "kappa_5k_audit",
                "kappa_extended_5k",
                "delta_ext_vs_5k_audit_pp",
            ],
        ),
        "",
        "## Coverage",
        _markdown_table(
            [
                {
                    "shared_fields_total": len(shared_keys),
                    "filtered_shared_fields": len(filtered_shared),
                }
            ],
            ["shared_fields_total", "filtered_shared_fields"],
        ),
        "",
    ]
    compare_path.write_text("\n".join(compare_lines), encoding="utf-8")

    # 5-sample CI plot.
    sample_maps: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for sample_name, (run_a, run_b, run_c) in BASELINE_SAMPLES.items():
        keep = split_hadm_id_set if sample_name == "extended_5k" else None
        loaded_sample_a = _load_results(run_a, keep_hadm_ids=keep)
        loaded_sample_b = _load_results(run_b)
        loaded_sample_c = _load_results(run_c)
        sample_maps[sample_name] = {
            "A": loaded_sample_a["parsed"],
            "B": loaded_sample_b["parsed"],
            "C": loaded_sample_c["parsed"],
        }

    sample_order = [
        "refinement_150",
        "holdout_150",
        "methodology_1k",
        "methodology_5k_audit_500",
        "extended_5k",
    ]
    sample_labels = ["refinement 150", "holdout 150", "1k", "5k-audit 500", "extended 5k"]

    field_series: dict[str, dict[str, dict[str, float]]] = {}
    for field_idx, field_key in enumerate(filtered_shared):
        field_series[field_key] = {}
        for sample_idx, sample_name in enumerate(sample_order):
            a_arr, b_arr, c_arr = _extract_sample_votes(
                field_key=field_key,
                sample_maps=sample_maps,
                sample_name=sample_name,
            )
            mean, ci_low, ci_high = _bootstrap_ci(
                a_arr,
                b_arr,
                c_arr,
                n_iter=max(10, int(args.n_bootstrap)),
                seed=int(args.seed) + (sample_idx * 1000) + field_idx,
            )
            field_series[field_key][sample_name] = {
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

    figure_output_path = Path(args.figure_output)
    figure_output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_kappa_with_bootstrap_ci(
        field_series=field_series,
        sample_order=sample_order,
        sample_labels=sample_labels,
        output_path=figure_output_path,
    )

    # Final verification report.
    overlap_files = [
        "refinement_150.csv",
        "holdout_150.csv",
        "smoke_200.csv",
        "methodology_1k.csv",
        "methodology_5k.csv",
        "gold_1k.csv",
    ]
    overlap_rows: list[dict[str, Any]] = []
    for name in overlap_files:
        path = config.SPLITS_DIR / name
        if not path.exists():
            overlap_rows.append({"split": name, "exists": "no", "overlap_count": "NA"})
            continue
        ids = set(_load_split_hadm_ids(path))
        overlap_rows.append(
            {
                "split": name,
                "exists": "yes",
                "overlap_count": len(split_hadm_id_set.intersection(ids)),
            }
        )

    run_status_rows = []
    for entry in [loaded_a, loaded_b, loaded_c]:
        run_status_rows.append(
            {
                "run_id": entry["run_id"],
                "n_attempted": int(entry["n_attempted"]),
                "n_parsed": int(entry["n_parsed"]),
            }
        )

    cost_b = _load_total_cost_usd(str(args.run_id_b))
    cost_c = _load_total_cost_usd(str(args.run_id_c))
    combined_cost = cost_b + cost_c
    cap_ok = combined_cost <= float(args.hard_cap_usd)

    expected_outputs = [
        output_kappa_path,
        compare_path,
        figure_output_path,
        audit_output_path,
        cluster_output_path,
        Path(args.verification),
    ]
    output_rows = [
        {
            "path": str(path),
            "exists": path.exists(),
            "sha256": file_sha256(path) if path.exists() else "",
        }
        for path in expected_outputs[:-1]
    ]

    verification_path = Path(args.verification)
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    verification_lines = [
        "# Prompt 26 Agent 3 Verification",
        "",
        "## Summary",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "split": str(split_path),
                    "split_n": len(split_hadm_ids),
                    "intersection_abc": len(joined_hadm_ids),
                    "cost_b_usd": f"{cost_b:.4f}",
                    "cost_c_usd": f"{cost_c:.4f}",
                    "combined_bc_cost_usd": f"{combined_cost:.4f}",
                    "hard_cap_usd": f"{float(args.hard_cap_usd):.2f}",
                    "cap_ok": cap_ok,
                }
            ],
            [
                "timestamp_utc",
                "split",
                "split_n",
                "intersection_abc",
                "cost_b_usd",
                "cost_c_usd",
                "combined_bc_cost_usd",
                "hard_cap_usd",
                "cap_ok",
            ],
        ),
        "",
        "## Overlap checks",
        _markdown_table(overlap_rows, ["split", "exists", "overlap_count"]),
        "",
        "## Run status",
        _markdown_table(run_status_rows, ["run_id", "n_attempted", "n_parsed"]),
        "",
        "## Deliverables",
        _markdown_table(output_rows, ["path", "exists", "sha256"]),
        "",
        "## Notes",
        (
            "- `data/splits/SPLITS_MANIFEST.json` is not modified by this agent "
            "due ownership constraints."
        ),
        "",
    ]
    verification_path.write_text("\n".join(verification_lines), encoding="utf-8")

    print(f"Wrote extended kappa report: {output_kappa_path}")
    print(f"Wrote sidecar JSON: {extended_sidecar_path}")
    print(f"Wrote 5-way comparison: {compare_path}")
    print(f"Wrote extended audit corpus: {audit_output_path}")
    print(f"Wrote extended audit clusters report: {cluster_output_path}")
    print(f"Wrote CI plot: {figure_output_path}")
    print(f"Wrote verification: {verification_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
