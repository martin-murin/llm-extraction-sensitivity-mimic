"""
Runs ensemble weak-supervision smoke checks.

Reads: configs/optimization.yaml.
Writes: local reports/artifacts determined by CLI defaults.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/13_ensemble_smoke.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import logging
import re
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
from src.labeling_functions.embedding_backend import EmbeddingCache, OpenAIEmbeddingBackend
from src.labeling_functions.embedding_lf import build_all_embedding_lfs
from src.labeling_functions.icd_lf import build_all_icd_lfs
from src.labeling_functions.llm_lf import (
    SNORKEL_TARGET_FIELD_VALUE_PAIRS,
    build_all_llm_lfs,
)
from src.labeling_functions.regex_lf import build_all_regex_lfs
from src.labeling_functions.section_parser import parse_sections
from src.llm.batch_runner import BatchSummary, run_batch
from src.llm.client import LLMClient
from src.schema.fields import LLMNoteFeatures
from src.snorkel_fit.label_model import aggregate_predictions, build_lf_vote_matrix

logger = logging.getLogger("scripts.13_ensemble_smoke")

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
ENSEMBLE_BUDGET_USD = 1.0


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = []
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *lines])


def _vote_to_label(vote: Vote) -> str:
    if vote == Vote.POSITIVE:
        return "POSITIVE"
    if vote == Vote.NEGATIVE:
        return "NEGATIVE"
    return "ABSTAIN"


def _combine_votes(votes: list[Vote]) -> Vote:
    if any(vote == Vote.POSITIVE for vote in votes):
        return Vote.POSITIVE
    if any(vote == Vote.NEGATIVE for vote in votes):
        return Vote.NEGATIVE
    return Vote.ABSTAIN


def _split_csv_path(split: str, run_config: dict[str, Any]) -> Path:
    size_key = f"{split}_split_size"
    split_size = int(run_config.get(size_key, 0))
    if split_size <= 0:
        raise RuntimeError(f"Invalid split size in config for key '{size_key}': {split_size}")
    return config.SPLITS_DIR / f"{split}_{split_size}.csv"


def _load_features_from_results(results_path: Path) -> dict[int, LLMNoteFeatures]:
    if not results_path.exists():
        return {}
    output: dict[int, LLMNoteFeatures] = {}
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not bool(payload.get("parse_ok", False)):
                continue
            features_json = payload.get("features_json")
            if not isinstance(features_json, dict):
                continue
            hadm_id = int(payload["hadm_id"])
            output[hadm_id] = LLMNoteFeatures.model_validate(features_json)
    return output


def _stub_variant_from_a(
    *,
    source_dir: Path,
    target_dir: Path,
    variant: str,
    hadm_ids: list[int],
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    source_jsonl = source_dir / "results.jsonl"
    target_jsonl = target_dir / "results.jsonl"

    jsonl_lines: list[str] = []
    if source_jsonl.exists():
        with source_jsonl.open("r", encoding="utf-8") as handle:
            jsonl_lines = [line.rstrip("\n") for line in handle if line.strip()]

    for hadm_id in hadm_ids:
        source_path = source_dir / f"{hadm_id}.json"
        if not source_path.exists():
            continue
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        payload["variant"] = variant
        payload["processed_at_utc"] = datetime.now(tz=UTC).isoformat()
        target_path = target_dir / f"{hadm_id}.json"
        target_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if jsonl_lines:
        target_jsonl.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
    else:
        target_jsonl.write_text("", encoding="utf-8")


def _normalize_excerpt(text: str, limit: int = 420) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3d ensemble smoke on refinement notes.")
    parser.add_argument("--run-id", type=str, default="ensemble_smoke_v1")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--split", type=str, default="refinement")
    parser.add_argument("--config", type=Path, default=Path("configs/optimization.yaml"))
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    config.load_env()
    with args.config.open("r", encoding="utf-8") as handle:
        run_config = yaml.safe_load(handle) or {}

    split_path = _split_csv_path(args.split, run_config)
    split_frame = pd.read_csv(split_path)
    hadm_ids = sorted(int(value) for value in split_frame["hadm_id"].tolist())[: args.n_samples]

    engine = get_engine()
    notes = fetch_notes_by_hadm_ids(engine, hadm_ids)
    icd_codes_by_hadm = fetch_icd_codes_by_hadm_ids(engine, hadm_ids)
    primary_icd_by_hadm = fetch_primary_icd_by_hadm_ids(engine, hadm_ids)
    sections_by_hadm = {hadm_id: parse_sections(notes.get(hadm_id, "")) for hadm_id in hadm_ids}

    max_concurrency = int(
        run_config.get("max_concurrent_requests", config.MAX_CONCURRENT_REQUESTS)
    )
    client = LLMClient(
        semaphore_limit=max_concurrency,
        run_id=args.run_id,
        max_budget_usd=ENSEMBLE_BUDGET_USD,
    )
    output_root = config.RAW_RESPONSES_DIR / args.run_id
    variant_a_dir = output_root / "variant_a"
    variant_b_dir = output_root / "variant_b"
    variant_c_dir = output_root / "variant_c"

    logger.info("Running live extraction for variant A on %s notes.", len(hadm_ids))
    batch_summary: BatchSummary = asyncio.run(
        run_batch(
            notes={hadm_id: notes.get(hadm_id, "") for hadm_id in hadm_ids},
            client=client,
            run_id=args.run_id,
            output_dir=variant_a_dir,
            variant="a",
            include_reasoning=True,
            max_concurrency=max_concurrency,
            checkpoint_every=10,
            resume=True,
        )
    )

    logger.warning(
        "Phase 3d smoke: variants B and C are stubbed as copies of variant A. "
        "Phase 3e will make B/C distinct."
    )
    _stub_variant_from_a(
        source_dir=variant_a_dir,
        target_dir=variant_b_dir,
        variant="b",
        hadm_ids=hadm_ids,
    )
    _stub_variant_from_a(
        source_dir=variant_a_dir,
        target_dir=variant_c_dir,
        variant="c",
        hadm_ids=hadm_ids,
    )

    features_a = _load_features_from_results(variant_a_dir / "results.jsonl")
    features_b = _load_features_from_results(variant_b_dir / "results.jsonl")
    features_c = _load_features_from_results(variant_c_dir / "results.jsonl")

    llm_maps_by_hadm: dict[int, dict[str, LLMNoteFeatures]] = {}
    for hadm_id in hadm_ids:
        llm_variant_map: dict[str, LLMNoteFeatures] = {}
        if hadm_id in features_a:
            llm_variant_map["a"] = features_a[hadm_id]
        if hadm_id in features_b:
            llm_variant_map["b"] = features_b[hadm_id]
        if hadm_id in features_c:
            llm_variant_map["c"] = features_c[hadm_id]
        llm_maps_by_hadm[hadm_id] = llm_variant_map

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
        variants=["a", "b", "c"],
        target_field_value_pairs=SNORKEL_TARGET_FIELD_VALUE_PAIRS,
    )
    embedding_backend = OpenAIEmbeddingBackend(client=client)
    embedding_cache = EmbeddingCache(
        cache_dir=config.DATA_DIR / "cache" / "embeddings" / embedding_backend.model_id,
        backend=embedding_backend,
    )
    embedding_lfs = build_all_embedding_lfs(
        patterns_dir=config.REPO_ROOT / "src" / "labeling_functions" / "patterns",
        backend=embedding_backend,
        cache=embedding_cache,
    )

    all_lfs: list[LabelingFunction] = [*icd_lfs, *regex_lfs, *llm_lfs, *embedding_lfs]
    L_all, lf_names_all = build_lf_vote_matrix(all_lfs, inputs)

    lf_type_by_name: dict[str, str] = {}
    for lf in icd_lfs:
        lf_type_by_name[str(lf.name)] = "icd"
    for lf in regex_lfs:
        lf_type_by_name[str(lf.name)] = "regex"
    for lf in llm_lfs:
        lf_type_by_name[str(lf.name)] = "llm"
    for lf in embedding_lfs:
        lf_type_by_name[str(lf.name)] = "embedding"

    coverage_rows: list[dict[str, Any]] = []
    n_rows = max(len(inputs), 1)
    for lf_index, lf_name in enumerate(lf_names_all):
        column = L_all[:, lf_index]
        n_pos = int(np.sum(column == Vote.POSITIVE))
        n_neg = int(np.sum(column == Vote.NEGATIVE))
        n_abs = int(np.sum(column == Vote.ABSTAIN))
        coverage_rows.append(
            {
                "lf_name": lf_name,
                "lf_type": lf_type_by_name.get(lf_name, "unknown"),
                "target_field": str(getattr(all_lfs[lf_index], "target_field", "")),
                "target_value": str(getattr(all_lfs[lf_index], "target_value", "")),
                "pct_firing": f"{((n_pos + n_neg) / n_rows) * 100.0:.2f}%",
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_abstain": n_abs,
            }
        )

    target_sections: list[str] = []
    disagreements: list[dict[str, Any]] = []

    for target_field, target_value, _field_type in SNORKEL_TARGET_FIELD_VALUE_PAIRS:
        probs, diagnostics = aggregate_predictions(
            lfs=all_lfs,
            inputs=inputs,
            target_field=target_field,
            target_value=target_value,
        )
        target_lfs = [
            lf
            for lf in all_lfs
            if str(getattr(lf, "target_field", "")) == target_field
            and str(getattr(lf, "target_value", "")) == target_value
        ]

        llm_a_lfs = [
            lf for lf in target_lfs if str(getattr(lf, "name", "")).startswith("llm_a_")
        ]
        target_icd_lfs = [
            lf for lf in target_lfs if str(getattr(lf, "name", "")).startswith("icd_")
        ]
        target_regex_lfs = [
            lf for lf in target_lfs if str(getattr(lf, "name", "")).startswith("regex_")
        ]

        rows: list[dict[str, Any]] = []
        for row_index, lf_input in enumerate(inputs):
            hadm_id = lf_input.hadm_id
            llm_vote = _combine_votes([lf(lf_input).vote for lf in llm_a_lfs])
            icd_vote = _combine_votes([lf(lf_input).vote for lf in target_icd_lfs])
            regex_vote = _combine_votes([lf(lf_input).vote for lf in target_regex_lfs])
            prob_positive = float(probs[row_index, 1]) if probs.size else 0.5

            llm_baseline = 0.5
            if llm_vote == Vote.POSITIVE:
                llm_baseline = 1.0
            elif llm_vote == Vote.NEGATIVE:
                llm_baseline = 0.0
            divergence = abs(prob_positive - llm_baseline)
            if divergence > 0.3:
                primary = primary_icd_by_hadm.get(hadm_id)
                lf_vote_pattern = {
                    str(lf.name): _vote_to_label(lf(lf_input).vote)
                    for lf in target_lfs
                }
                disagreements.append(
                    {
                        "hadm_id": hadm_id,
                        "target_field": target_field,
                        "target_value": target_value,
                        "llm_a_vote": _vote_to_label(llm_vote),
                        "snorkel_prob_positive": prob_positive,
                        "divergence": divergence,
                        "primary_icd_code": primary[0] if primary else "",
                        "note_context": _normalize_excerpt(notes.get(hadm_id, "")),
                        "lf_vote_pattern": lf_vote_pattern,
                    }
                )

            rows.append(
                {
                    "hadm_id": hadm_id,
                    "llm_variant_a_vote": _vote_to_label(llm_vote),
                    "icd_vote": _vote_to_label(icd_vote),
                    "regex_vote": _vote_to_label(regex_vote),
                    "snorkel_prob_positive": f"{prob_positive:.3f}",
                    "divergence_from_llm_a": f"{divergence:.3f}",
                }
            )

        target_sections.append(f"### {target_field} = {target_value}")
        target_sections.append("")
        target_sections.append(
            f"- n_LFs_contributing: {len(diagnostics.get('lf_names_used', []))}"
        )
        target_sections.append(f"- fit_status: {diagnostics.get('fit_status', 'unknown')}")
        target_sections.append("")
        target_sections.append(
            _markdown_table(
                rows,
                [
                    "hadm_id",
                    "llm_variant_a_vote",
                    "icd_vote",
                    "regex_vote",
                    "snorkel_prob_positive",
                    "divergence_from_llm_a",
                ],
            )
        )
        target_sections.append("")

    disagreements_sorted = sorted(
        disagreements, key=lambda row: float(row["divergence"]), reverse=True
    )
    top_three = disagreements_sorted[:3]

    cost_summary = client.cost_tracker.summary()

    report_lines: list[str] = []
    report_lines.append("# 15 Ensemble Smoke Report")
    report_lines.append("")
    report_lines.append("## Run metadata")
    report_lines.append("")
    report_lines.append(
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_id": args.run_id,
                    "split": args.split,
                    "n_samples": len(hadm_ids),
                    "variant_a_parse_ok": batch_summary.n_successful_parse,
                    "variant_a_parse_failed": batch_summary.n_failed_parse,
                    "variant_a_api_error": batch_summary.n_api_error,
                    "variant_stubbed": "b,c <- a",
                    "total_cost_usd": f"{float(cost_summary['total_cost_usd']):.6f}",
                    "budget_cap_usd": f"{ENSEMBLE_BUDGET_USD:.2f}",
                }
            ],
            [
                "timestamp_utc",
                "run_id",
                "split",
                "n_samples",
                "variant_a_parse_ok",
                "variant_a_parse_failed",
                "variant_a_api_error",
                "variant_stubbed",
                "total_cost_usd",
                "budget_cap_usd",
            ],
        )
    )
    report_lines.append("")

    report_lines.append("## LF counts")
    report_lines.append("")
    report_lines.append(
        _markdown_table(
            [
                {"lf_type": "icd", "count": len(icd_lfs)},
                {"lf_type": "regex_active", "count": len(regex_lfs)},
                {"lf_type": "llm", "count": len(llm_lfs)},
                {"lf_type": "embedding", "count": len(embedding_lfs)},
                {"lf_type": "total_active", "count": len(all_lfs)},
            ],
            ["lf_type", "count"],
        )
    )
    report_lines.append("")

    report_lines.append("## LF coverage matrix")
    report_lines.append("")
    report_lines.append(
        _markdown_table(
            sorted(coverage_rows, key=lambda row: (row["lf_type"], row["lf_name"])),
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
        )
    )
    report_lines.append("")

    report_lines.append("## Per-target aggregation results")
    report_lines.append("")
    report_lines.extend(target_sections)

    report_lines.append("## Three cases worth inspection")
    report_lines.append("")
    if not top_three:
        report_lines.append("_No high-divergence cases found (threshold > 0.3)._")
        report_lines.append("")
    else:
        for case in top_three:
            report_lines.append(
                f"### hadm_id={case['hadm_id']} | {case['target_field']}={case['target_value']}"
            )
            report_lines.append("")
            report_lines.append(
                _markdown_table(
                    [
                        {
                            "llm_a_vote": case["llm_a_vote"],
                            "snorkel_prob_positive": f"{float(case['snorkel_prob_positive']):.3f}",
                            "divergence": f"{float(case['divergence']):.3f}",
                            "primary_icd_code": case["primary_icd_code"],
                        }
                    ],
                    ["llm_a_vote", "snorkel_prob_positive", "divergence", "primary_icd_code"],
                )
            )
            report_lines.append("")
            report_lines.append("LF vote pattern:")
            pattern_rows = [
                {"lf_name": lf_name, "vote": vote}
                for lf_name, vote in sorted(case["lf_vote_pattern"].items())
            ]
            report_lines.append(_markdown_table(pattern_rows, ["lf_name", "vote"]))
            report_lines.append("")
            report_lines.append(f"Note context: {case['note_context']}")
            report_lines.append("")

    output_path = config.CODEX_OUTPUTS_DIR / "15_ensemble_smoke_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")
    logger.info("Wrote ensemble smoke report to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
