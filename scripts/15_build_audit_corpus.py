"""
Builds disagreement audit corpora for optimization.

Reads: data/optimization/audit_corpus_v1.jsonl.
Writes: data/optimization/audit_corpus_v1.jsonl.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/15_build_audit_corpus.py` unless the script's argparse help says otherwise.
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
from src.schema.vocabulary import ADMISSION_REASON_TAGS
from src.utils.threeway_kappa import (
    cohen_kappa_safe,
    count_positive_admission_tag,
    count_positive_enum,
    count_positive_tristate,
    encode_tristate,
    pabak_score,
    percent_agreement,
)

PAIR_KEYS = [("a", "b"), ("a", "c"), ("b", "c")]
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


def _load_results(run_id: str) -> dict[str, Any]:
    results_path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    parsed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if bool(row.get("parse_ok")) and isinstance(row.get("features_json"), dict):
            parsed[int(row["hadm_id"])] = row["features_json"]
    return {"run_id": run_id, "parsed": parsed, "rows": rows}


def _pairwise_metrics(encoded: dict[str, list[int]]) -> dict[str, float]:
    kappas = {}
    pabaks = {}
    agreements = {}
    for left, right in PAIR_KEYS:
        key = f"{left}_{right}"
        kappas[key] = cohen_kappa_safe(encoded[left], encoded[right])
        pabaks[key] = pabak_score(encoded[left], encoded[right])
        agreements[key] = percent_agreement(encoded[left], encoded[right])

    return {
        "kappa_a_b": kappas["a_b"],
        "kappa_a_c": kappas["a_c"],
        "kappa_b_c": kappas["b_c"],
        "kappa_mean": float(np.mean(list(kappas.values()))),
        "pabak_mean": float(np.mean(list(pabaks.values()))),
        "pct_agreement_mean": float(np.mean(list(agreements.values()))),
    }


def _votes_disagree(votes: dict[str, str]) -> bool:
    return len({votes["a"], votes["b"], votes["c"]}) > 1


def _excerpt(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    compact = " ".join(str(text).split())
    return compact[:max_chars]


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
        if not _votes_disagree(votes):
            continue
        pattern = classify_vote_pattern(votes)
        disagreements.append({"hadm_id": hadm_id, "votes": votes})

        outlier = pattern.outlier_variant
        outlier_reasoning = ""
        consensus_reasoning: dict[str, str] = {}
        if outlier is not None:
            outlier_reasoning = _excerpt(
                str(parsed_by_variant[outlier][hadm_id].get("reasoning", "")),
                600,
            )
            for variant in ["a", "b", "c"]:
                if variant == outlier:
                    continue
                consensus_reasoning[variant] = _excerpt(
                    str(parsed_by_variant[variant][hadm_id].get("reasoning", "")),
                    400,
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


def _build_tristate_record(
    *,
    field: str,
    hadm_ids: list[int],
    parsed_by_variant: dict[str, dict[int, dict[str, Any]]],
    icd_info: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    values = {
        variant: [
            str(parsed_by_variant[variant][hadm_id].get(field, "not_documented"))
            for hadm_id in hadm_ids
        ]
        for variant in ["a", "b", "c"]
    }
    vote_tuples = list(zip(values["a"], values["b"], values["c"], strict=True))
    if not has_any_disagreement(vote_tuples):
        return None

    encoded = {
        variant: [encode_tristate(value) for value in vals]
        for variant, vals in values.items()
    }
    metrics = _pairwise_metrics(encoded)

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

    return {
        "field": field,
        "target_value": None,
        "field_type": "tristate",
        **metrics,
        "n_positive_a": n_positive_a,
        "n_positive_b": n_positive_b,
        "n_positive_c": n_positive_c,
        "n_positive_total": n_positive_total,
        "low_base_rate_flag": n_positive_total < 10,
        "disagreement_count": len(disagreements),
        "disagreement_pattern_summary": summary,
        "examples": examples,
    }


def _build_enum_record(
    *,
    field: str,
    hadm_ids: list[int],
    parsed_by_variant: dict[str, dict[int, dict[str, Any]]],
    icd_info: dict[int, dict[str, Any]],
    field_type: str,
) -> dict[str, Any] | None:
    values = {
        variant: [
            str(parsed_by_variant[variant][hadm_id].get(field, "not_documented"))
            for hadm_id in hadm_ids
        ]
        for variant in ["a", "b", "c"]
    }
    vote_tuples = list(zip(values["a"], values["b"], values["c"], strict=True))
    if not has_any_disagreement(vote_tuples):
        return None

    observed = sorted({value for values_list in values.values() for value in values_list})
    mapping = {value: idx for idx, value in enumerate(observed)}
    encoded = {variant: [mapping[value] for value in vals] for variant, vals in values.items()}
    metrics = _pairwise_metrics(encoded)

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

    return {
        "field": field,
        "target_value": None,
        "field_type": field_type,
        **metrics,
        "n_positive_a": n_positive_a,
        "n_positive_b": n_positive_b,
        "n_positive_c": n_positive_c,
        "n_positive_total": n_positive_total,
        "low_base_rate_flag": n_positive_total < 10,
        "disagreement_count": len(disagreements),
        "disagreement_pattern_summary": summary,
        "examples": examples,
    }


def _build_admission_tag_record(
    *,
    tag: str,
    hadm_ids: list[int],
    parsed_by_variant: dict[str, dict[int, dict[str, Any]]],
    icd_info: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    values: dict[str, list[str]] = {"a": [], "b": [], "c": []}
    lists_by_variant: dict[str, list[list[str]]] = {"a": [], "b": [], "c": []}

    for variant in ["a", "b", "c"]:
        for hadm_id in hadm_ids:
            tags = list(parsed_by_variant[variant][hadm_id].get("admission_reason_tags", []))
            lists_by_variant[variant].append(tags)
            values[variant].append("present" if tag in set(tags) else "absent")

    vote_tuples = list(zip(values["a"], values["b"], values["c"], strict=True))
    if not has_any_disagreement(vote_tuples):
        return None

    encoded = {
        variant: [1 if value == "present" else 0 for value in vals]
        for variant, vals in values.items()
    }
    metrics = _pairwise_metrics(encoded)

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

    n_positive_a = count_positive_admission_tag(lists_by_variant["a"], tag)
    n_positive_b = count_positive_admission_tag(lists_by_variant["b"], tag)
    n_positive_c = count_positive_admission_tag(lists_by_variant["c"], tag)
    n_positive_total = n_positive_a + n_positive_b + n_positive_c

    return {
        "field": "admission_reason_tags",
        "target_value": tag,
        "field_type": "admission_tag",
        **metrics,
        "n_positive_a": n_positive_a,
        "n_positive_b": n_positive_b,
        "n_positive_c": n_positive_c,
        "n_positive_total": n_positive_total,
        "low_base_rate_flag": n_positive_total < 10,
        "disagreement_count": len(disagreements),
        "disagreement_pattern_summary": summary,
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 3f audit corpus from refinement runs."
    )
    parser.add_argument(
        "--run-ids",
        nargs=3,
        default=["refinement_v1_a", "refinement_v1_b", "refinement_v1_c"],
    )
    parser.add_argument("--output", default="data/optimization/audit_corpus_v1.jsonl")
    parser.add_argument("--include-low-base-rate", action="store_true", default=False)
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
        help="Optional split override used to load ICD context.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(config.CODEX_OUTPUTS_DIR / "17_audit_corpus_summary.md"),
    )
    return parser.parse_args()


def _infer_split_from_run_ids(run_ids: list[str]) -> str:
    lowered = [run_id.lower() for run_id in run_ids]
    if lowered and all(
        "methodology_5k_audit" in run_id or "subset500" in run_id for run_id in lowered
    ):
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
    args = parse_args()
    config.load_env()

    loaded = [_load_results(run_id) for run_id in args.run_ids]
    parsed_by_variant = {
        "a": loaded[0]["parsed"],
        "b": loaded[1]["parsed"],
        "c": loaded[2]["parsed"],
    }
    hadm_ids = sorted(
        set(parsed_by_variant["a"])
        .intersection(parsed_by_variant["b"])
        .intersection(parsed_by_variant["c"])
    )

    split_name = args.split or _infer_split_from_run_ids(list(args.run_ids))
    split_path = _resolve_split_csv(split_name)
    split_frame = pd.read_csv(split_path)
    split_frame["hadm_id"] = pd.to_numeric(split_frame["hadm_id"], errors="coerce").astype("int64")
    icd_info = {
        int(row["hadm_id"]): {
            "primary_icd_code": str(row.get("primary_icd_code", "")),
            "chapter": str(row.get("chapter", "")),
        }
        for _, row in split_frame.iterrows()
    }

    all_candidate_records: list[dict[str, Any]] = []

    for field in TRISTATE_FIELDS:
        record = _build_tristate_record(
            field=field,
            hadm_ids=hadm_ids,
            parsed_by_variant=parsed_by_variant,
            icd_info=icd_info,
        )
        if record is not None:
            all_candidate_records.append(record)

    for field in ENUM_FIELDS:
        record = _build_enum_record(
            field=field,
            hadm_ids=hadm_ids,
            parsed_by_variant=parsed_by_variant,
            icd_info=icd_info,
            field_type="enum",
        )
        if record is not None:
            all_candidate_records.append(record)

    dominant_record = _build_enum_record(
        field="dominant_admission_reason",
        hadm_ids=hadm_ids,
        parsed_by_variant=parsed_by_variant,
        icd_info=icd_info,
        field_type="dominant_admission_reason",
    )
    if dominant_record is not None:
        all_candidate_records.append(dominant_record)

    tags_observed = set()
    for variant in ["a", "b", "c"]:
        for hadm_id in hadm_ids:
            tags_observed.update(
                parsed_by_variant[variant][hadm_id].get("admission_reason_tags", [])
            )

    for tag in ADMISSION_REASON_TAGS:
        if tag not in tags_observed:
            continue
        record = _build_admission_tag_record(
            tag=tag,
            hadm_ids=hadm_ids,
            parsed_by_variant=parsed_by_variant,
            icd_info=icd_info,
        )
        if record is not None:
            all_candidate_records.append(record)

    total_with_disagreements = len(all_candidate_records)
    excluded_low_base = 0
    emitted_records: list[dict[str, Any]] = []
    for record in all_candidate_records:
        if should_include_record(
            int(record["n_positive_total"]),
            include_low_base_rate=args.include_low_base_rate,
        ):
            emitted_records.append(record)
        else:
            excluded_low_base += 1

    emitted_records.sort(
        key=lambda row: (
            int(row["disagreement_count"]),
            -float(row["kappa_mean"]),
            str(row["field"]),
            str(row.get("target_value") or ""),
        ),
        reverse=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in emitted_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    total_examples_emitted = int(sum(len(record.get("examples", [])) for record in emitted_records))
    low_base_in_emitted = int(sum(1 for record in emitted_records if record["low_base_rate_flag"]))

    top10 = emitted_records[:10]
    summary_rows = [
        {
            "field": row["field"],
            "target_value": row.get("target_value") or "",
            "disagreement_count": row["disagreement_count"],
            "kappa_mean": row["kappa_mean"],
            "n_positive_total": row["n_positive_total"],
            "dominant_pattern": row["disagreement_pattern_summary"],
        }
        for row in top10
    ]

    summary_lines = [
        "# Audit Corpus Summary",
        "",
        "## Run metadata",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_ids": ", ".join(args.run_ids),
                    "split": split_name,
                    "output": str(output_path),
                    "include_low_base_rate": args.include_low_base_rate,
                }
            ],
            ["timestamp_utc", "run_ids", "split", "output", "include_low_base_rate"],
        ),
        "",
        "## Corpus counts",
        _markdown_table(
            [
                {
                    "fields_with_disagreements": total_with_disagreements,
                    "fields_excluded_low_base_rate": excluded_low_base,
                    "records_emitted": len(emitted_records),
                    "records_low_base_rate_in_emitted": low_base_in_emitted,
                    "total_examples_emitted": total_examples_emitted,
                    "jsonl_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
                }
            ],
            [
                "fields_with_disagreements",
                "fields_excluded_low_base_rate",
                "records_emitted",
                "records_low_base_rate_in_emitted",
                "total_examples_emitted",
                "jsonl_size_bytes",
            ],
        ),
        "",
        "## Top 10 fields by disagreement frequency",
        _markdown_table(
            summary_rows,
            [
                "field",
                "target_value",
                "disagreement_count",
                "kappa_mean",
                "n_positive_total",
                "dominant_pattern",
            ],
        ),
        "",
    ]

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote audit corpus to {output_path}")
    print(f"Wrote audit summary to {summary_path}")
    print(
        "Summary: "
        f"fields_with_disagreements={total_with_disagreements}, "
        f"excluded_low_base_rate={excluded_low_base}, "
        f"records_emitted={len(emitted_records)}, "
        f"total_examples_emitted={total_examples_emitted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
