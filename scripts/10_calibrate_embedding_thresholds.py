"""
Calibrates embedding labeling-function thresholds.

Reads: configs/optimization.yaml, src/labeling_functions/patterns, data/raw_responses/coverage_v2/results.jsonl, codex_outputs/10_embedding_threshold_calibration.md.
Writes: data/raw_responses/coverage_v2/results.jsonl, codex_outputs/10_embedding_threshold_calibration.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/10_calibrate_embedding_thresholds.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import yaml

from src import config
from src.db.connection import get_engine
from src.db.queries import fetch_notes_by_hadm_ids
from src.labeling_functions.embedding_backend import EmbeddingCache, OpenAIEmbeddingBackend
from src.labeling_functions.pattern_bootstrap import load_coverage_v2_results
from src.labeling_functions.section_parser import get_section, parse_sections
from src.llm.client import LLMClient
from src.schema.section_map import FIELD_SECTION_MAP

logger = logging.getLogger("scripts.10_calibrate_embedding_thresholds")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate per-field embedding LF thresholds.")
    parser.add_argument("--run-id", type=str, default="calibrate_thresholds_v1")
    parser.add_argument("--config", type=Path, default=Path("configs/optimization.yaml"))
    parser.add_argument(
        "--patterns-dir",
        type=Path,
        default=Path("src/labeling_functions/patterns"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/raw_responses/coverage_v2/results.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("codex_outputs/10_embedding_threshold_calibration.md"),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model id for threshold calibration.",
    )
    return parser.parse_args()


def _distribution_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _load_field_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Pattern YAML must be mapping: {path}")
    return payload


def _collect_concatenated_sections(note_text: str, field_name: str) -> str:
    sections = parse_sections(note_text)
    chunks: list[str] = []
    for canonical_name in FIELD_SECTION_MAP[field_name]:
        section_text = get_section(sections, canonical_name)
        if section_text is None:
            continue
        stripped = section_text.strip()
        if stripped:
            chunks.append(stripped)
    return "\n---\n".join(chunks)


def calibrated_threshold_from_p10(raw_p10: float) -> tuple[float, str]:
    threshold = max(0.65, min(0.85, raw_p10))
    if threshold == 0.65 and raw_p10 < 0.65:
        return threshold, "floor_0.65"
    if threshold == 0.85 and raw_p10 > 0.85:
        return threshold, "cap_0.85"
    return threshold, "raw_p10"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    config.load_env()
    with args.config.open("r", encoding="utf-8") as handle:
        run_config = yaml.safe_load(handle) or {}

    budget_cap = float(run_config.get("max_budget_usd_coverage", 5.0))
    concurrency = int(run_config.get("max_concurrent_requests", config.MAX_CONCURRENT_REQUESTS))

    client = LLMClient(
        semaphore_limit=concurrency,
        run_id=args.run_id,
        max_budget_usd=budget_cap,
    )
    backend = OpenAIEmbeddingBackend(client=client, model_id=args.embedding_model)
    cache_dir = config.DATA_DIR / "cache" / "embeddings" / backend.model_id
    cache = EmbeddingCache(cache_dir=cache_dir, backend=backend)

    results = load_coverage_v2_results(args.results)
    features_by_hadm: dict[int, dict[str, Any]] = {
        int(row["hadm_id"]): row["features"]
        for row in results
        if isinstance(row.get("features"), dict)
    }

    engine = get_engine()

    summary_rows: list[dict[str, Any]] = []
    detail_lines: list[str] = []

    pattern_files = sorted(args.patterns_dir.glob("*__*.yaml"))
    for yaml_path in pattern_files:
        payload = _load_field_payload(yaml_path)
        field_name = str(payload["field_name"])
        target_value = str(payload["target_value"])

        seed_phrases_raw = payload.get("embedding_seed_phrases", [])
        if not isinstance(seed_phrases_raw, list):
            raise ValueError(f"embedding_seed_phrases must be list in {yaml_path}")
        seed_phrases = [str(item) for item in seed_phrases_raw if str(item).strip()]

        if not seed_phrases:
            logger.info("Skipping %s: no seed phrases", field_name)
            summary_rows.append(
                {
                    "field": field_name,
                    "n_source_notes": 0,
                    "n_similarity_pairs": 0,
                    "threshold": "skipped",
                    "reason": "no_seed_phrases",
                }
            )
            detail_lines.append(f"### {field_name}\n\n- Skipped: no seed phrases.\n")
            continue

        source_hadm_ids = [
            hadm_id
            for hadm_id, features in features_by_hadm.items()
            if features.get(field_name) == target_value
        ]
        source_hadm_ids = sorted(source_hadm_ids)

        notes = fetch_notes_by_hadm_ids(engine, source_hadm_ids)
        section_texts: list[str] = []
        for hadm_id in source_hadm_ids:
            note_text = notes.get(hadm_id, "")
            concatenated = _collect_concatenated_sections(note_text, field_name)
            if len(concatenated) < 20:
                continue
            section_texts.append(concatenated)

        if not section_texts:
            payload.pop("embedding_threshold", None)
            with yaml_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)

            reason = "no_source_notes" if not source_hadm_ids else "no_section_text"
            summary_rows.append(
                {
                    "field": field_name,
                    "n_source_notes": len(source_hadm_ids),
                    "n_similarity_pairs": 0,
                    "threshold": "skipped",
                    "reason": reason,
                }
            )
            detail_lines.append(
                f"### {field_name}\n\n"
                f"- Source notes with target value: {len(source_hadm_ids)}\n"
                "- No calibratable section text >=20 chars; embedding_threshold not set.\n"
            )
            continue

        seed_embeddings = asyncio.run(cache.embed_cached(seed_phrases))
        section_embeddings = asyncio.run(cache.embed_cached(section_texts))

        similarity_scores: list[float] = []
        for section_vector in section_embeddings:
            similarities = seed_embeddings @ section_vector
            similarity_scores.extend(float(value) for value in similarities.tolist())

        stats = _distribution_stats(similarity_scores)
        raw_p10 = stats["p10"]
        chosen_threshold, reason = calibrated_threshold_from_p10(raw_p10)

        payload["embedding_threshold"] = float(round(chosen_threshold, 4))
        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)

        summary_rows.append(
            {
                "field": field_name,
                "n_source_notes": len(source_hadm_ids),
                "n_similarity_pairs": len(similarity_scores),
                "threshold": f"{chosen_threshold:.4f}",
                "reason": reason,
            }
        )

        detail_lines.append(f"### {field_name}")
        detail_lines.append("")
        detail_lines.append(f"- Source notes with target value: {len(source_hadm_ids)}")
        detail_lines.append(f"- Similarity pairs computed: {len(similarity_scores)}")
        detail_lines.append(
            f"- Distribution: min={stats['min']:.4f}, p10={stats['p10']:.4f}, "
            f"p25={stats['p25']:.4f}, median={stats['median']:.4f}, "
            f"p75={stats['p75']:.4f}, p90={stats['p90']:.4f}, max={stats['max']:.4f}"
        )
        detail_lines.append(
            f"- Chosen threshold: {chosen_threshold:.4f} "
            f"(raw p10={raw_p10:.4f}, rationale={reason})"
        )
        if reason in {"floor_0.65", "cap_0.85"}:
            detail_lines.append(f"- Boundary applied due to {reason}.")
        detail_lines.append("")

    cost_summary = client.cost_tracker.summary()

    report_lines: list[str] = []
    report_lines.append("# 10 Embedding Threshold Calibration")
    report_lines.append("")
    report_lines.append("## Run metadata")
    report_lines.append("")
    report_lines.append(
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_id": args.run_id,
                    "results_path": str(args.results),
                    "patterns_dir": str(args.patterns_dir),
                    "backend_model": backend.model_id,
                    "cache_dir": str(cache_dir),
                    "cache_hits": cache.cache_hits,
                    "cache_misses": cache.cache_misses,
                    "backend_calls": cache.backend_calls,
                    "total_input_tokens": int(cost_summary["total_input_tokens"]),
                    "total_output_tokens": int(cost_summary["total_output_tokens"]),
                    "total_cost_usd": f"{float(cost_summary['total_cost_usd']):.6f}",
                    "budget_cap_usd": f"{budget_cap:.2f}",
                }
            ],
            [
                "timestamp_utc",
                "run_id",
                "results_path",
                "patterns_dir",
                "backend_model",
                "cache_dir",
                "cache_hits",
                "cache_misses",
                "backend_calls",
                "total_input_tokens",
                "total_output_tokens",
                "total_cost_usd",
                "budget_cap_usd",
            ],
        )
    )
    report_lines.append("")

    report_lines.append("## Per-field threshold summary")
    report_lines.append("")
    report_lines.append(
        _markdown_table(
            summary_rows,
            ["field", "n_source_notes", "n_similarity_pairs", "threshold", "reason"],
        )
    )
    report_lines.append("")

    report_lines.append("## Calibration details")
    report_lines.append("")
    report_lines.extend(detail_lines)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    logger.info("Wrote calibration report to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
