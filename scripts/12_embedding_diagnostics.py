"""
Produces embedding labeling-function diagnostics.

Reads: configs/optimization.yaml, codex_outputs/12_embedding_diagnostics.md, codex_outputs/11_embedding_smoke_coverage.md, data/raw_responses/coverage_v2/results.jsonl, src/labeling_functions/patterns.
Writes: codex_outputs/12_embedding_diagnostics.md, codex_outputs/11_embedding_smoke_coverage.md, data/raw_responses/coverage_v2/results.jsonl.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/12_embedding_diagnostics.py` unless the script's argparse help says otherwise.
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
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src import config
from src.db.connection import get_engine
from src.db.queries import fetch_notes_by_hadm_ids
from src.labeling_functions.embedding_backend import EmbeddingCache, OpenAIEmbeddingBackend
from src.labeling_functions.pattern_bootstrap import REGEX_PILOT_FIELDS
from src.labeling_functions.section_parser import get_section, parse_sections
from src.llm.client import LLMClient
from src.schema.section_map import FIELD_SECTION_MAP

logger = logging.getLogger("scripts.12_embedding_diagnostics")

DIAGNOSTIC_BUDGET_USD = 3.0
SMALL_MODEL = "text-embedding-3-small"
LARGE_MODEL = "text-embedding-3-large"
PRODUCTION_SECTION_TOKEN_ESTIMATE = 660_000_000


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


def _parse_markdown_table_for_section(report_text: str, section_title: str) -> list[dict[str, str]]:
    marker = f"## {section_title}"
    section_start = report_text.find(marker)
    if section_start < 0:
        return []
    remainder = report_text[section_start + len(marker):]
    lines = [line.rstrip("\n") for line in remainder.splitlines()]
    table_lines: list[str] = []
    saw_table = False
    for line in lines:
        if line.startswith("## "):
            break
        if line.strip().startswith("|"):
            table_lines.append(line.strip())
            saw_table = True
            continue
        if saw_table and not line.strip():
            break
    if len(table_lines) < 2:
        return []
    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for row_line in table_lines[2:]:
        cells = [cell.strip() for cell in row_line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells, strict=False)))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run embedding diagnostics for Phase 3b/3c extension."
    )
    parser.add_argument("--run-id", type=str, default="embed_diag_v1")
    parser.add_argument("--config", type=Path, default=Path("configs/optimization.yaml"))
    parser.add_argument(
        "--compare-large",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include text-embedding-3-large comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("codex_outputs/12_embedding_diagnostics.md"),
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("codex_outputs/11_embedding_smoke_coverage.md"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/raw_responses/coverage_v2/results.jsonl"),
    )
    parser.add_argument(
        "--patterns-dir",
        type=Path,
        default=Path("src/labeling_functions/patterns"),
    )
    return parser.parse_args()


def _load_coverage_features(path: Path) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not bool(payload.get("parse_ok", False)):
                continue
            features = payload.get("features_json")
            if not isinstance(features, dict):
                continue
            output[int(payload["hadm_id"])] = features
    return output


def _collect_field_text(note_text: str, field_name: str) -> str:
    sections = parse_sections(note_text)
    chunks: list[str] = []
    for section_name in FIELD_SECTION_MAP[field_name]:
        section_text = get_section(sections, section_name)
        if section_text:
            chunks.append(section_text.strip())
    return "\n---\n".join(chunk for chunk in chunks if chunk)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1:
        a = a.reshape(-1)
    if b.ndim != 1:
        b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _highlight_dnr(text: str, max_chars: int = 300) -> str:
    excerpt = text[:max_chars]
    pattern = re.compile(r"(dnr|dni|do not resuscitate|code status|comfort)", re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", excerpt)


def _load_seed_phrases(patterns_dir: Path, field_name: str) -> list[str]:
    yaml_path = patterns_dir / f"{field_name}__yes.yaml"
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    seeds = payload.get("embedding_seed_phrases", [])
    if not isinstance(seeds, list):
        return []
    return [str(seed) for seed in seeds if str(seed).strip()]


def _small_cost_from_tokens(tokens: int) -> float:
    return (tokens / 1_000_000.0) * 0.02


def _large_cost_from_tokens(tokens: int) -> float:
    return (tokens / 1_000_000.0) * 0.13


def _diagnostic_conclusion(
    fp_scores: list[float],
    tp_scores: list[float],
    separation_gain: float | None,
) -> str:
    if fp_scores and tp_scores and (median(tp_scores) - median(fp_scores)) >= 0.05:
        return "embedding IS signal, threshold too low"
    if separation_gain is not None and separation_gain >= 0.15:
        return "embedding discriminates poorly, move model"
    return "seed phrase is general-topic detector, fix seed"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    config.load_env()
    with args.config.open("r", encoding="utf-8") as handle:
        run_config = yaml.safe_load(handle) or {}

    concurrency = int(run_config.get("max_concurrent_requests", config.MAX_CONCURRENT_REQUESTS))
    client = LLMClient(
        semaphore_limit=concurrency,
        run_id=args.run_id,
        max_budget_usd=DIAGNOSTIC_BUDGET_USD,
    )

    small_backend = OpenAIEmbeddingBackend(client=client, model_id=SMALL_MODEL)
    small_cache = EmbeddingCache(
        cache_dir=config.DATA_DIR / "cache" / "embeddings" / SMALL_MODEL,
        backend=small_backend,
    )
    large_backend: OpenAIEmbeddingBackend | None = None
    large_cache: EmbeddingCache | None = None
    if args.compare_large:
        large_backend = OpenAIEmbeddingBackend(client=client, model_id=LARGE_MODEL)
        large_cache = EmbeddingCache(
            cache_dir=config.DATA_DIR / "cache" / "embeddings" / LARGE_MODEL,
            backend=large_backend,
        )

    features_by_hadm = _load_coverage_features(args.results)
    smoke_report_text = args.baseline_report.read_text(encoding="utf-8")
    fp_table = _parse_markdown_table_for_section(
        smoke_report_text,
        "Notes where embedding LF voted POSITIVE but LLM did not",
    )
    dnr_fp_hadm_ids = [
        int(row["hadm_id"])
        for row in fp_table
        if row.get("field") == "dnr_dni_documented"
    ][:10]

    engine = get_engine()
    fp_notes = fetch_notes_by_hadm_ids(engine, dnr_fp_hadm_ids)
    dnr_seeds = _load_seed_phrases(args.patterns_dir, "dnr_dni_documented")
    seed_for_fp = dnr_seeds[4] if len(dnr_seeds) >= 5 else (dnr_seeds[0] if dnr_seeds else "")

    sanity_variants = [
        "DNR/DNI confirmed",
        "dnr/dni confirmed",
        "DNR / DNI   confirmed",
        "Code Status was confirmed as DNR/DNI.",
        "patient presented with chest pain",
    ]
    sanity_embeddings = asyncio.run(small_cache.embed_cached(sanity_variants))
    sanity_scores = [
        _cosine(sanity_embeddings[0], sanity_embeddings[0]),
        _cosine(sanity_embeddings[0], sanity_embeddings[1]),
        _cosine(sanity_embeddings[0], sanity_embeddings[2]),
        _cosine(sanity_embeddings[0], sanity_embeddings[3]),
        _cosine(sanity_embeddings[0], sanity_embeddings[4]),
    ]

    fp_rows: list[dict[str, Any]] = []
    fp_scores: list[float] = []
    if seed_for_fp:
        seed_vector = asyncio.run(small_cache.embed_cached([seed_for_fp]))[0]
        for hadm_id in dnr_fp_hadm_ids:
            note_text = fp_notes.get(hadm_id, "")
            section_text = _collect_field_text(note_text, "dnr_dni_documented")
            if not section_text:
                continue
            section_vector = asyncio.run(small_cache.embed_cached([section_text]))[0]
            sim = _cosine(seed_vector, section_vector)
            fp_scores.append(sim)
            feature = features_by_hadm.get(hadm_id, {})
            fp_rows.append(
                {
                    "hadm_id": hadm_id,
                    "similarity_small": f"{sim:.3f}",
                    "section_excerpt": _highlight_dnr(section_text),
                    "llm_value": str(feature.get("dnr_dni_documented", "")),
                    "llm_reasoning_excerpt": str(feature.get("reasoning", ""))[:200],
                }
            )

    dnr_yes_hadm_ids = sorted(
        hadm_id
        for hadm_id, features in features_by_hadm.items()
        if features.get("dnr_dni_documented") == "yes"
    )[:5]
    tp_notes = fetch_notes_by_hadm_ids(engine, dnr_yes_hadm_ids)
    tp_rows: list[dict[str, Any]] = []
    tp_scores: list[float] = []
    if seed_for_fp:
        seed_vector = asyncio.run(small_cache.embed_cached([seed_for_fp]))[0]
        for hadm_id in dnr_yes_hadm_ids:
            note_text = tp_notes.get(hadm_id, "")
            section_text = _collect_field_text(note_text, "dnr_dni_documented")
            if not section_text:
                continue
            section_vector = asyncio.run(small_cache.embed_cached([section_text]))[0]
            sim = _cosine(seed_vector, section_vector)
            tp_scores.append(sim)
            feature = features_by_hadm.get(hadm_id, {})
            tp_rows.append(
                {
                    "hadm_id": hadm_id,
                    "similarity_small": f"{sim:.3f}",
                    "section_excerpt": _highlight_dnr(section_text),
                    "llm_value": str(feature.get("dnr_dni_documented", "")),
                    "llm_reasoning_excerpt": str(feature.get("reasoning", ""))[:200],
                }
            )

    model_rows: list[dict[str, Any]] = []
    separation_small = 0.0
    separation_large: float | None = None
    separation_gain: float | None = None
    recommendation = "stay on text-embedding-3-small"

    if args.compare_large and large_cache is not None:
        split_path = config.SPLITS_DIR / "smoke_200.csv"
        split_df = pd.read_csv(split_path)
        smoke_hadm_ids = sorted(int(value) for value in split_df["hadm_id"].tolist())
        notes = fetch_notes_by_hadm_ids(engine, smoke_hadm_ids)

        pair_specs: list[tuple[str, str, str, str]] = []
        for field_name, target_value in REGEX_PILOT_FIELDS:
            seeds = _load_seed_phrases(args.patterns_dir, field_name)
            if not seeds:
                continue
            seed_phrase = seeds[0]
            positives = [
                hadm_id
                for hadm_id, features in features_by_hadm.items()
                if features.get(field_name) == target_value
            ]
            negatives = [
                hadm_id
                for hadm_id, features in features_by_hadm.items()
                if features.get(field_name) != target_value
            ]
            for hadm_id in positives[:2]:
                text = _collect_field_text(notes.get(hadm_id, ""), field_name)
                if text:
                    pair_specs.append((field_name, "self_match", seed_phrase, text))
            for hadm_id in negatives[:2]:
                text = _collect_field_text(notes.get(hadm_id, ""), field_name)
                if text:
                    pair_specs.append((field_name, "hard_negative", seed_phrase, text))

        pair_specs = pair_specs[:20]
        if pair_specs:
            small_seed_vectors = asyncio.run(
                small_cache.embed_cached([seed for _, _, seed, _ in pair_specs])
            )
            small_text_vectors = asyncio.run(
                small_cache.embed_cached([text for _, _, _, text in pair_specs])
            )
            large_seed_vectors = asyncio.run(
                large_cache.embed_cached([seed for _, _, seed, _ in pair_specs])
            )
            large_text_vectors = asyncio.run(
                large_cache.embed_cached([text for _, _, _, text in pair_specs])
            )

            small_self: list[float] = []
            small_neg: list[float] = []
            large_self: list[float] = []
            large_neg: list[float] = []

            for idx, (field_name, pair_type, seed_phrase, _text) in enumerate(pair_specs):
                small_sim = _cosine(small_seed_vectors[idx], small_text_vectors[idx])
                large_sim = _cosine(large_seed_vectors[idx], large_text_vectors[idx])
                delta = large_sim - small_sim
                model_rows.append(
                    {
                        "pair_description": f"{field_name}/{pair_type}/{seed_phrase[:30]}",
                        "small_cosine": f"{small_sim:.3f}",
                        "large_cosine": f"{large_sim:.3f}",
                        "delta": f"{delta:.3f}",
                    }
                )
                if pair_type == "self_match":
                    small_self.append(small_sim)
                    large_self.append(large_sim)
                else:
                    small_neg.append(small_sim)
                    large_neg.append(large_sim)

            small_self_median = median(small_self) if small_self else 0.0
            small_neg_median = median(small_neg) if small_neg else 0.0
            large_self_median = median(large_self) if large_self else 0.0
            large_neg_median = median(large_neg) if large_neg else 0.0
            separation_small = small_self_median - small_neg_median
            separation_large = large_self_median - large_neg_median
            separation_gain = separation_large - separation_small

            if separation_gain >= 0.15:
                recommendation = "switch to text-embedding-3-large"

    decision_line = f"DECISION: {recommendation}"

    section_b_conclusion = _diagnostic_conclusion(fp_scores, tp_scores, separation_gain)

    cost_summary = client.cost_tracker.summary()
    estimated_small_production = _small_cost_from_tokens(PRODUCTION_SECTION_TOKEN_ESTIMATE)
    estimated_large_production = _large_cost_from_tokens(PRODUCTION_SECTION_TOKEN_ESTIMATE)

    lines: list[str] = []
    lines.append("# 12 Embedding Diagnostics")
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_id": args.run_id,
                    "compare_large": args.compare_large,
                    "total_input_tokens": int(cost_summary["total_input_tokens"]),
                    "total_cost_usd_tracker": f"{float(cost_summary['total_cost_usd']):.6f}",
                    "budget_cap_usd": f"{DIAGNOSTIC_BUDGET_USD:.2f}",
                }
            ],
            [
                "timestamp_utc",
                "run_id",
                "compare_large",
                "total_input_tokens",
                "total_cost_usd_tracker",
                "budget_cap_usd",
            ],
        )
    )
    lines.append("")

    lines.append("## Section A: backend sanity check")
    lines.append("")
    lines.append(
        _markdown_table(
            [
                {"pair": "self vs self", "cosine_small": f"{sanity_scores[0]:.3f}"},
                {"pair": "self vs lowercase", "cosine_small": f"{sanity_scores[1]:.3f}"},
                {
                    "pair": "self vs whitespace-normalized",
                    "cosine_small": f"{sanity_scores[2]:.3f}",
                },
                {"pair": "self vs expanded", "cosine_small": f"{sanity_scores[3]:.3f}"},
                {"pair": "self vs unrelated", "cosine_small": f"{sanity_scores[4]:.3f}"},
            ],
            ["pair", "cosine_small"],
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: self-similarity is expected near 1.0, lexical variants stay high, "
        "and unrelated text should be materially lower."
    )
    lines.append("")

    lines.append("## Section B: case-by-case false-positive and true-positive review")
    lines.append("")
    lines.append("### DNR false-positive sample (10)")
    lines.append("")
    lines.append(
        _markdown_table(
            fp_rows,
            [
                "hadm_id",
                "similarity_small",
                "llm_value",
                "section_excerpt",
                "llm_reasoning_excerpt",
            ],
        )
    )
    lines.append("")
    lines.append("### DNR true-positive sample (5)")
    lines.append("")
    lines.append(
        _markdown_table(
            tp_rows,
            [
                "hadm_id",
                "similarity_small",
                "llm_value",
                "section_excerpt",
                "llm_reasoning_excerpt",
            ],
        )
    )
    lines.append("")
    median_true = median(tp_scores) if tp_scores else 0.0
    median_false = median(fp_scores) if fp_scores else 0.0
    lines.append(
        f"Section B conclusion: **{section_b_conclusion}** "
        f"(median_true={median_true:.3f}, median_false={median_false:.3f})."
    )
    lines.append("")

    lines.append("## Section C: small-vs-large model comparison")
    lines.append("")
    if args.compare_large:
        lines.append(
            _markdown_table(
                model_rows,
                ["pair_description", "small_cosine", "large_cosine", "delta"],
            )
        )
        lines.append("")
        lines.append(
            _markdown_table(
                [
                    {
                        "small_separation_median": f"{separation_small:.3f}",
                        "large_separation_median": f"{(separation_large or 0.0):.3f}",
                        "separation_gain_large_minus_small": f"{(separation_gain or 0.0):.3f}",
                        "production_cost_small_usd_est": f"{estimated_small_production:.2f}",
                        "production_cost_large_usd_est": f"{estimated_large_production:.2f}",
                    }
                ],
                [
                    "small_separation_median",
                    "large_separation_median",
                    "separation_gain_large_minus_small",
                    "production_cost_small_usd_est",
                    "production_cost_large_usd_est",
                ],
            )
        )
    else:
        lines.append("_Model comparison skipped (`--no-compare-large`)._")
    lines.append("")

    lines.append("## Decision")
    lines.append("")
    lines.append(decision_line)
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(decision_line)
    logger.info("Wrote embedding diagnostics report to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
