"""
Fits/evaluates Snorkel label model on holdout artifacts.

Reads: configs/optimization.yaml, data/holdout_snorkel/holdout_v1_predictions.parquet, codex_outputs/21_holdout_snorkel_report.md.
Writes: data/holdout_snorkel/holdout_v1_predictions.parquet, codex_outputs/21_holdout_snorkel_report.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/18_run_snorkel_holdout.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src import config
from src.db.connection import get_engine
from src.db.queries import (
    fetch_icd_codes_by_hadm_ids,
    fetch_notes_by_hadm_ids,
    fetch_primary_icd_by_hadm_ids,
)
from src.labeling_functions.base import LFInput, LabelingFunction, Vote
from src.labeling_functions.icd_lf import build_all_icd_lfs
from src.labeling_functions.llm_lf import (
    FieldType,
    SNORKEL_TARGET_FIELD_VALUE_PAIRS,
    build_all_llm_lfs,
)
from src.labeling_functions.regex_lf import build_all_regex_lfs
from src.labeling_functions.section_parser import get_section, parse_sections
from src.schema.fields import LLMNoteFeatures
from src.schema.section_map import FIELD_SECTION_MAP
from src.snorkel_fit.label_model import aggregate_predictions, build_lf_vote_matrix

logger = logging.getLogger("scripts.18_run_snorkel_holdout")

ACTIVE_REGEX_TARGET_FIELDS: set[str] = {
    "aki_present",
    "dnr_dni_documented",
    "palliative_care_consult",
    "home_health_ordered",
    "substance_use_active",
    "fall_risk_documented",
    "cognitive_impairment",
    "goals_of_care_flag",
}


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *lines])


def _load_features_from_results(results_path: Path) -> dict[int, LLMNoteFeatures]:
    if not results_path.exists():
        return {}
    output: dict[int, LLMNoteFeatures] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not bool(payload.get("parse_ok", False)):
            continue
        features_json = payload.get("features_json")
        if not isinstance(features_json, dict):
            continue
        hadm_id = int(payload["hadm_id"])
        output[hadm_id] = LLMNoteFeatures.model_validate(features_json)
    return output


def _llm_consensus_probability(
    *,
    features_by_variant: dict[str, LLMNoteFeatures],
    target_field: str,
    target_value: str,
    field_type: FieldType,
) -> float:
    available_variants = sorted(features_by_variant.keys())
    if not available_variants:
        return 0.5

    if field_type == FieldType.ADMISSION_TAG_MEMBERSHIP:
        votes = []
        for variant in available_variants:
            feature = features_by_variant.get(variant)
            if feature is None:
                continue
            tags = list(feature.admission_reason_tags)
            votes.append(1 if target_value in tags else 0)
        if not votes:
            return 0.5
        return 1.0 if sum(votes) >= (len(votes) / 2.0) else 0.0

    if field_type == FieldType.TRISTATE:
        pos = 0
        neg = 0
        for variant in available_variants:
            feature = features_by_variant.get(variant)
            if feature is None:
                continue
            value = str(getattr(feature, target_field, "not_documented"))
            if target_value == "yes":
                if value == "yes":
                    pos += 1
                elif value == "no":
                    neg += 1
            elif target_value == "no":
                if value == "no":
                    pos += 1
                elif value == "yes":
                    neg += 1
        if pos == 0 and neg == 0:
            return 0.5
        if pos > neg:
            return 1.0
        if neg > pos:
            return 0.0
        return 0.5

    return 0.5


def _excerpt_for_field(field_name: str, sections: dict[str, str], fallback_note: str) -> str:
    section_names = FIELD_SECTION_MAP.get(field_name, [])
    chunks: list[str] = []
    for canonical_name in section_names:
        section_text = get_section(sections, canonical_name)
        if section_text:
            chunks.append(" ".join(section_text.split()))
    text = " | ".join(chunks) if chunks else " ".join(fallback_note.split())
    return text[:260]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Snorkel aggregation for a split.")
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=["holdout_v1_a", "holdout_v1_b", "holdout_v1_c"],
        metavar="RUN_ID",
    )
    parser.add_argument(
        "--split",
        default="holdout",
        choices=[
            "refinement",
            "holdout",
            "smoke",
            "methodology_1k",
            "methodology_5k",
            "methodology_5k_audit_500",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/optimization.yaml"))
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/holdout_snorkel/holdout_v1_predictions.parquet"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("codex_outputs/21_holdout_snorkel_report.md"),
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config.load_env()

    settings = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    split_sizes = {
        "refinement": int(settings.get("refinement_split_size", 150)),
        "holdout": int(settings.get("holdout_split_size", 150)),
        "smoke": int(settings.get("smoke_split_size", 200)),
        "methodology_1k": int(settings.get("methodology_1k_split_size", 1000)),
        "methodology_5k": int(settings.get("methodology_5k_split_size", 5000)),
        "methodology_5k_audit_500": int(
            settings.get("methodology_5k_audit_500_split_size", 500)
        ),
    }
    split_size = split_sizes[args.split]
    split_path = config.SPLITS_DIR / f"{args.split}_{split_size}.csv"
    if not split_path.exists():
        fallback_path = config.SPLITS_DIR / f"{args.split}.csv"
        if fallback_path.exists():
            split_path = fallback_path
    split_frame = pd.read_csv(split_path)
    hadm_ids = sorted(int(value) for value in split_frame["hadm_id"].tolist())

    engine = get_engine()
    notes = fetch_notes_by_hadm_ids(engine, hadm_ids)
    icd_codes_by_hadm = fetch_icd_codes_by_hadm_ids(engine, hadm_ids)
    primary_icd_by_hadm = fetch_primary_icd_by_hadm_ids(engine, hadm_ids)
    sections_by_hadm = {hadm_id: parse_sections(notes.get(hadm_id, "")) for hadm_id in hadm_ids}
    primary_icd_code_map = {
        hadm_id: (value[0] if value is not None else "")
        for hadm_id, value in primary_icd_by_hadm.items()
    }

    if not args.run_ids:
        raise ValueError("--run-ids must include at least one run id.")
    if len(args.run_ids) > 3:
        raise ValueError("At most 3 run ids are supported.")

    variant_labels = ["a", "b", "c"][: len(args.run_ids)]
    run_ids_by_variant = dict(zip(variant_labels, args.run_ids, strict=True))
    features_by_variant = {
        variant: _load_features_from_results(config.RAW_RESPONSES_DIR / run_id / "results.jsonl")
        for variant, run_id in run_ids_by_variant.items()
    }

    llm_maps_by_hadm: dict[int, dict[str, LLMNoteFeatures]] = {}
    for hadm_id in hadm_ids:
        variant_map: dict[str, LLMNoteFeatures] = {}
        for variant in variant_labels:
            feature_map = features_by_variant[variant]
            if hadm_id in feature_map:
                variant_map[variant] = feature_map[hadm_id]
        llm_maps_by_hadm[hadm_id] = variant_map

    inputs: list[LFInput] = []
    for hadm_id in hadm_ids:
        primary = primary_icd_by_hadm.get(hadm_id)
        inputs.append(
            LFInput(
                hadm_id=hadm_id,
                note_text=notes.get(hadm_id, ""),
                icd_codes=icd_codes_by_hadm.get(hadm_id),
                primary_icd_code=primary[0] if primary is not None else None,
                primary_icd_version=primary[1] if primary is not None else None,
                sections=sections_by_hadm.get(hadm_id),
                section_embeddings=None,
                llm_extraction_by_variant=llm_maps_by_hadm.get(hadm_id),
            )
        )

    icd_lfs = build_all_icd_lfs()
    regex_lfs_all = build_all_regex_lfs(
        config.REPO_ROOT / "src" / "labeling_functions" / "patterns"
    )
    regex_lfs = [lf for lf in regex_lfs_all if str(lf.target_field) in ACTIVE_REGEX_TARGET_FIELDS]
    llm_lfs = build_all_llm_lfs(
        variants=variant_labels,
        target_field_value_pairs=SNORKEL_TARGET_FIELD_VALUE_PAIRS,
    )
    embedding_lfs: list[LabelingFunction] = []

    all_lfs: list[LabelingFunction] = [*icd_lfs, *regex_lfs, *llm_lfs, *embedding_lfs]
    L_all, lf_names_all = build_lf_vote_matrix(all_lfs, inputs)

    lf_type_by_name: dict[str, str] = {}
    for lf in icd_lfs:
        lf_type_by_name[str(lf.name)] = "icd"
    for lf in regex_lfs:
        lf_type_by_name[str(lf.name)] = "regex"
    for lf in llm_lfs:
        lf_type_by_name[str(lf.name)] = "llm"

    coverage_rows: list[dict[str, Any]] = []
    for lf_index, lf_name in enumerate(lf_names_all):
        col = L_all[:, lf_index]
        n_firing = int(np.sum(col != Vote.ABSTAIN))
        coverage_rows.append(
            {
                "lf_name": lf_name,
                "lf_type": lf_type_by_name.get(lf_name, "unknown"),
                "target_field": str(getattr(all_lfs[lf_index], "target_field", "")),
                "target_value": str(getattr(all_lfs[lf_index], "target_value", "")),
                "pct_firing": f"{(n_firing / max(len(inputs), 1)) * 100.0:.2f}%",
                "n_positive": int(np.sum(col == Vote.POSITIVE)),
                "n_negative": int(np.sum(col == Vote.NEGATIVE)),
                "n_abstain": int(np.sum(col == Vote.ABSTAIN)),
            }
        )

    lf_counts_rows = [
        {"lf_type": "icd", "n_lfs": len(icd_lfs)},
        {"lf_type": "regex", "n_lfs": len(regex_lfs)},
        {"lf_type": "llm", "n_lfs": len(llm_lfs)},
        {"lf_type": "embedding", "n_lfs": len(embedding_lfs)},
        {"lf_type": "total_active", "n_lfs": len(all_lfs)},
    ]

    prediction_rows: list[dict[str, Any]] = []
    target_distribution_rows: list[dict[str, Any]] = []
    fit_status_counts: dict[str, int] = {}
    disagreement_rows: list[dict[str, Any]] = []

    for target_field, target_value, field_type in SNORKEL_TARGET_FIELD_VALUE_PAIRS:
        probs, diagnostics = aggregate_predictions(
            lfs=all_lfs,
            inputs=inputs,
            target_field=target_field,
            target_value=target_value,
        )

        fit_status = str(diagnostics.get("fit_status", "unknown"))
        fit_status_counts[fit_status] = fit_status_counts.get(fit_status, 0) + 1

        matched_lfs = [
            lf
            for lf in all_lfs
            if str(getattr(lf, "target_field", "")) == target_field
            and str(getattr(lf, "target_value", "")) == target_value
        ]
        if matched_lfs:
            L_target, _ = build_lf_vote_matrix(matched_lfs, inputs)
        else:
            L_target = np.empty((len(inputs), 0), dtype=np.int8)

        prob_values = probs[:, 1] if probs.size else np.asarray([], dtype=np.float64)
        target_distribution_rows.append(
            {
                "target_field": target_field,
                "target_value": target_value,
                "n_lfs_contributing": len(matched_lfs),
                "fit_status": fit_status,
                "median_prob_positive": float(np.median(prob_values)) if prob_values.size else 0.0,
                "p10_prob_positive": (
                    float(np.percentile(prob_values, 10)) if prob_values.size else 0.0
                ),
                "p90_prob_positive": (
                    float(np.percentile(prob_values, 90)) if prob_values.size else 0.0
                ),
            }
        )

        disagreement_candidates: list[dict[str, Any]] = []
        for row_index, hadm_id in enumerate(hadm_ids):
            prob_positive = float(probs[row_index, 1]) if probs.size else 0.5
            n_lfs_contrib = int(np.sum(L_target[row_index] != Vote.ABSTAIN)) if L_target.size else 0
            prediction_rows.append(
                {
                    "hadm_id": hadm_id,
                    "target_field": target_field,
                    "target_value": target_value,
                    "snorkel_prob_positive": prob_positive,
                    "fit_status": fit_status,
                    "n_lfs_total": len(matched_lfs),
                    "n_lfs_contributing": n_lfs_contrib,
                }
            )

            llm_consensus = _llm_consensus_probability(
                features_by_variant=llm_maps_by_hadm.get(hadm_id, {}),
                target_field=target_field,
                target_value=target_value,
                field_type=field_type,
            )
            diff = abs(prob_positive - llm_consensus)
            if diff > 0.3:
                excerpt = _excerpt_for_field(
                    field_name=target_field,
                    sections=sections_by_hadm.get(hadm_id, {}),
                    fallback_note=notes.get(hadm_id, ""),
                )
                disagreement_candidates.append(
                    {
                        "target_field": target_field,
                        "target_value": target_value,
                        "hadm_id": hadm_id,
                        "snorkel_prob_positive": prob_positive,
                        "llm_consensus_prob_positive": llm_consensus,
                        "abs_diff": diff,
                        "primary_icd_code": primary_icd_code_map.get(hadm_id, ""),
                        "excerpt": excerpt,
                    }
                )

        disagreement_candidates.sort(
            key=lambda row: float(row["abs_diff"]),
            reverse=True,
        )
        disagreement_rows.extend(disagreement_candidates[:3])

    prediction_df = pd.DataFrame(prediction_rows)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_parquet(args.output_parquet, index=False)

    fit_status_rows = [
        {"fit_status": status, "n_targets": count}
        for status, count in sorted(fit_status_counts.items())
    ]
    target_distribution_rows.sort(
        key=lambda row: (str(row["target_field"]), str(row["target_value"]))
    )
    coverage_rows.sort(key=lambda row: float(str(row["pct_firing"]).rstrip("%")), reverse=True)
    disagreement_rows.sort(key=lambda row: float(row["abs_diff"]), reverse=True)

    lines = [
        "# Snorkel Aggregation Report",
        "",
        "## Run metadata",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "split": args.split,
                    "n_hadm_ids": len(hadm_ids),
                    "run_id_a": run_ids_by_variant.get("a", ""),
                    "run_id_b": run_ids_by_variant.get("b", ""),
                    "run_id_c": run_ids_by_variant.get("c", ""),
                    "run_ids": ", ".join(args.run_ids),
                    "output_parquet": str(args.output_parquet),
                }
            ],
            [
                "timestamp_utc",
                "split",
                "n_hadm_ids",
                "run_id_a",
                "run_id_b",
                "run_id_c",
                "run_ids",
                "output_parquet",
            ],
        ),
        "",
        "## LF counts and coverage per type",
        _markdown_table(lf_counts_rows, ["lf_type", "n_lfs"]),
        "",
        "## Fit-status distribution",
        _markdown_table(fit_status_rows, ["fit_status", "n_targets"]),
        "",
        "## Per-target probability distribution",
        _markdown_table(
            target_distribution_rows,
            [
                "target_field",
                "target_value",
                "n_lfs_contributing",
                "fit_status",
                "median_prob_positive",
                "p10_prob_positive",
                "p90_prob_positive",
            ],
        ),
        "",
        "## LF coverage matrix",
        _markdown_table(
            coverage_rows,
            [
                "lf_name",
                "lf_type",
                "target_field",
                "target_value",
                "pct_firing",
                "n_positive",
                "n_negative",
                "n_abstain",
            ],
        ),
        "",
        (
            "## Disagreement audit (Snorkel probability vs LLM consensus differs by > 0.3; "
            "up to 3 examples per target)"
        ),
        _markdown_table(
            disagreement_rows,
            [
                "target_field",
                "target_value",
                "hadm_id",
                "primary_icd_code",
                "snorkel_prob_positive",
                "llm_consensus_prob_positive",
                "abs_diff",
                "excerpt",
            ],
        ),
        "",
        "Embedding LFs are parked and excluded from active aggregation.",
        "",
    ]

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote Snorkel parquet to {args.output_parquet}")
    print(f"Wrote Snorkel report to {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
