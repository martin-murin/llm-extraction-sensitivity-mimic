"""
Evaluates embedding labeling-function smoke coverage.

Reads: codex_outputs/11_embedding_smoke_coverage.md, configs/optimization.yaml, src/labeling_functions/patterns, data/raw_responses/coverage_v2/results.jsonl.
Writes: codex_outputs/11_embedding_smoke_coverage.md, data/raw_responses/coverage_v2/results.jsonl.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/11_smoke_embedding_coverage.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import yaml

from src import config
from src.db.connection import get_engine
from src.db.queries import fetch_notes_by_hadm_ids
from src.labeling_functions.base import LFInput, LFOutput, Vote
from src.labeling_functions.embedding_backend import EmbeddingCache, OpenAIEmbeddingBackend
from src.labeling_functions.embedding_lf import build_all_embedding_lfs
from src.labeling_functions.regex_lf import build_all_regex_lfs
from src.labeling_functions.section_embed import embed_notes_sections
from src.labeling_functions.section_parser import get_section, parse_sections
from src.llm.client import LLMClient
from src.schema.section_map import FIELD_SECTION_MAP

logger = logging.getLogger("scripts.11_smoke_embedding_coverage")

ORIGINAL_V1_FIELDS: tuple[str, ...] = (
    "dnr_dni_documented",
    "home_health_ordered",
    "palliative_care_consult",
    "substance_use_active",
)


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
    parser = argparse.ArgumentParser(description="Run smoke embedding LF coverage diagnostics.")
    parser.add_argument("--run-id", type=str, default="embed_smoke_v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("codex_outputs/11_embedding_smoke_coverage.md"),
    )
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
        "--baseline-report",
        type=Path,
        default=Path("codex_outputs/11_embedding_smoke_coverage.md"),
        help="Baseline embedding report for v1-v2 comparison.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
    )
    return parser.parse_args()


def _load_coverage_features(path: Path) -> dict[int, dict[str, Any]]:
    features_by_hadm: dict[int, dict[str, Any]] = {}
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
            features_by_hadm[int(payload["hadm_id"])] = features
    return features_by_hadm


def _extract_similarity(output: LFOutput) -> str:
    if output.confidence is None:
        return ""
    return f"{float(output.confidence):.3f}"


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


def _evaluate_regex_metrics(
    *,
    regex_lfs: list[Any],
    hadm_ids: list[int],
    fields: list[str],
    notes: dict[int, str],
    parsed_sections: dict[int, dict[str, str]],
    features_by_hadm: dict[int, dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    field_to_positive: dict[str, set[int]] = {}

    for lf in regex_lfs:
        field_name = str(lf.target_field)
        positives: set[int] = set()
        for hadm_id in hadm_ids:
            output = lf(
                LFInput(
                    hadm_id=hadm_id,
                    note_text=notes.get(hadm_id, ""),
                    sections=parsed_sections.get(hadm_id),
                )
            )
            if output.vote == Vote.POSITIVE:
                positives.add(hadm_id)
        field_to_positive[field_name] = positives

    metrics: dict[str, dict[str, float | int]] = {}
    for field_name in fields:
        regex_positive = field_to_positive.get(field_name, set())
        llm_yes = {
            hadm_id
            for hadm_id, features in features_by_hadm.items()
            if features.get(field_name) == "yes"
        }
        overlap = regex_positive.intersection(llm_yes)

        agreement = (len(overlap) / len(regex_positive) * 100.0) if regex_positive else 0.0
        recall = (len(overlap) / len(llm_yes) * 100.0) if llm_yes else 0.0

        metrics[field_name] = {
            "positive_n": len(regex_positive),
            "agreement_pct": agreement,
            "recall_pct": recall,
        }

    return metrics


def _sort_fields(fields: list[str]) -> list[str]:
    return sorted(fields, key=lambda value: (value not in ORIGINAL_V1_FIELDS, value))


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

    split_path = config.SPLITS_DIR / "smoke_200.csv"
    split_frame = pd.read_csv(split_path)
    hadm_ids = sorted(int(value) for value in split_frame["hadm_id"].tolist())

    engine = get_engine()
    notes = fetch_notes_by_hadm_ids(engine, hadm_ids)
    parsed_sections = {hadm_id: parse_sections(notes.get(hadm_id, "")) for hadm_id in hadm_ids}

    features_by_hadm = _load_coverage_features(args.results)

    embedding_lfs = build_all_embedding_lfs(
        patterns_dir=args.patterns_dir,
        backend=backend,
        cache=cache,
    )
    available_fields = _sort_fields(
        sorted({str(lf.target_field) for lf in embedding_lfs if lf.target_field is not None})
    )

    section_embeddings_by_hadm = asyncio.run(
        embed_notes_sections(
            notes=notes,
            fields=available_fields,
            backend=backend,
            cache=cache,
        )
    )

    outputs_by_lf: dict[str, dict[int, LFOutput]] = {str(lf.name): {} for lf in embedding_lfs}
    per_lf_positive_rows: dict[str, list[dict[str, Any]]] = {
        str(lf.name): [] for lf in embedding_lfs
    }

    for lf in embedding_lfs:
        lf_name = str(lf.name)
        for hadm_id in hadm_ids:
            output = lf(
                LFInput(
                    hadm_id=hadm_id,
                    note_text=notes.get(hadm_id, ""),
                    section_embeddings=section_embeddings_by_hadm.get(hadm_id),
                )
            )
            outputs_by_lf[lf_name][hadm_id] = output
            if output.vote != Vote.POSITIVE:
                continue
            per_lf_positive_rows[lf_name].append(
                {
                    "hadm_id": hadm_id,
                    "field": str(lf.target_field),
                    "target_value": str(lf.target_value),
                    "similarity": _extract_similarity(output),
                    "evidence": output.evidence or "",
                }
            )

    firing_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    fp_examples: list[dict[str, Any]] = []
    miss_examples: list[dict[str, Any]] = []
    embedding_metrics_by_field: dict[str, dict[str, float | int]] = {}

    for lf in embedding_lfs:
        lf_name = str(lf.name)
        field_name = str(lf.target_field)
        positives = per_lf_positive_rows[lf_name]
        positive_ids = {int(row["hadm_id"]) for row in positives}

        llm_yes_ids = {
            hadm_id
            for hadm_id, features in features_by_hadm.items()
            if features.get(field_name) == str(lf.target_value)
        }
        overlap = positive_ids.intersection(llm_yes_ids)

        agreement_pct = (len(overlap) / len(positive_ids) * 100.0) if positive_ids else 0.0
        recall_pct = (len(overlap) / len(llm_yes_ids) * 100.0) if llm_yes_ids else 0.0

        firing_rows.append(
            {
                "lf_name": lf_name,
                "field": field_name,
                "target_value": str(lf.target_value),
                "n_positive_votes": len(positive_ids),
                "pct_of_notes": f"{(len(positive_ids) / len(hadm_ids) * 100):.2f}%",
            }
        )

        agreement_rows.append(
            {
                "lf_name": lf_name,
                "field": field_name,
                "embedding_positive_n": len(positive_ids),
                "llm_yes_n": len(llm_yes_ids),
                "intersection_n": len(overlap),
                "agreement_pct": f"{agreement_pct:.2f}%",
                "recall_from_llm_pct": f"{recall_pct:.2f}%",
            }
        )

        embedding_metrics_by_field[field_name] = {
            "positive_n": len(positive_ids),
            "agreement_pct": agreement_pct,
            "recall_pct": recall_pct,
        }

        if len(fp_examples) < 10:
            for row in positives:
                hadm_id = int(row["hadm_id"])
                if hadm_id in llm_yes_ids:
                    continue
                fp_examples.append(
                    {
                        "hadm_id": hadm_id,
                        "field": field_name,
                        "similarity": row["similarity"],
                        "evidence": str(row["evidence"])[:300],
                    }
                )
                if len(fp_examples) >= 10:
                    break

        if len(miss_examples) < 10:
            missing_ids = sorted(llm_yes_ids.difference(positive_ids))
            for hadm_id in missing_ids:
                output = outputs_by_lf[lf_name][hadm_id]
                features = features_by_hadm.get(hadm_id, {})
                reasoning = str(features.get("reasoning") or "")
                miss_examples.append(
                    {
                        "hadm_id": hadm_id,
                        "field": field_name,
                        "max_similarity_or_reason": output.evidence or "",
                        "llm_reasoning": reasoning[:300],
                    }
                )
                if len(miss_examples) >= 10:
                    break

    section_rows: list[dict[str, Any]] = []
    for field_name in available_fields:
        for section_name in FIELD_SECTION_MAP[field_name]:
            n_present = 0
            for hadm_id in hadm_ids:
                if get_section(parsed_sections[hadm_id], section_name) is not None:
                    n_present += 1
            section_rows.append(
                {
                    "field": field_name,
                    "required_section": section_name,
                    "n_present": n_present,
                    "pct_of_notes": f"{(n_present / len(hadm_ids) * 100):.2f}%",
                }
            )

    regex_lfs = build_all_regex_lfs(args.patterns_dir)
    regex_metrics_by_field = _evaluate_regex_metrics(
        regex_lfs=regex_lfs,
        hadm_ids=hadm_ids,
        fields=available_fields,
        notes=notes,
        parsed_sections=parsed_sections,
        features_by_hadm=features_by_hadm,
    )

    comparison_rows: list[dict[str, Any]] = []
    for field_name in available_fields:
        regex_metrics = regex_metrics_by_field.get(field_name, {})
        embed_metrics = embedding_metrics_by_field.get(field_name, {})
        comparison_rows.append(
            {
                "field": field_name,
                "regex_positive_n": int(regex_metrics.get("positive_n", 0)),
                "regex_agreement_pct": f"{float(regex_metrics.get('agreement_pct', 0.0)):.2f}%",
                "regex_recall_pct": f"{float(regex_metrics.get('recall_pct', 0.0)):.2f}%",
                "embed_positive_n": int(embed_metrics.get("positive_n", 0)),
                "embed_agreement_pct": f"{float(embed_metrics.get('agreement_pct', 0.0)):.2f}%",
                "embed_recall_pct": f"{float(embed_metrics.get('recall_pct', 0.0)):.2f}%",
            }
        )

    baseline_rows: list[dict[str, str]] = []
    if args.baseline_report.exists():
        baseline_rows = _parse_markdown_table_for_section(
            args.baseline_report.read_text(encoding="utf-8"),
            "LLM agreement sanity check",
        )
    baseline_by_field: dict[str, dict[str, str]] = {
        row.get("field", ""): row for row in baseline_rows if row.get("field")
    }

    v2_vs_v1_rows: list[dict[str, Any]] = []
    for field_name in available_fields:
        current = embedding_metrics_by_field.get(field_name, {})
        baseline = baseline_by_field.get(field_name)
        if baseline is None:
            v2_vs_v1_rows.append(
                {
                    "field": field_name,
                    "v1_embed_positive_n": "n/a",
                    "v2_embed_positive_n": int(current.get("positive_n", 0)),
                    "v1_agreement_pct": "n/a",
                    "v2_agreement_pct": f"{float(current.get('agreement_pct', 0.0)):.2f}%",
                    "v1_recall_pct": "n/a",
                    "v2_recall_pct": f"{float(current.get('recall_pct', 0.0)):.2f}%",
                    "note": "first_run_in_v2",
                }
            )
            continue

        v2_vs_v1_rows.append(
            {
                "field": field_name,
                "v1_embed_positive_n": baseline.get("embedding_positive_n", "n/a"),
                "v2_embed_positive_n": int(current.get("positive_n", 0)),
                "v1_agreement_pct": baseline.get("agreement_pct", "n/a"),
                "v2_agreement_pct": f"{float(current.get('agreement_pct', 0.0)):.2f}%",
                "v1_recall_pct": baseline.get("recall_from_llm_pct", "n/a"),
                "v2_recall_pct": f"{float(current.get('recall_pct', 0.0)):.2f}%",
                "note": "original_4"
                if field_name in ORIGINAL_V1_FIELDS
                else "baseline_present",
            }
        )

    cost_summary = client.cost_tracker.summary()

    lines: list[str] = []
    lines.append("# 11 Embedding Smoke Coverage")
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_id": args.run_id,
                    "backend_model": backend.model_id,
                    "baseline_report": str(args.baseline_report),
                    "n_notes": len(hadm_ids),
                    "n_embedding_lfs": len(embedding_lfs),
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
                "backend_model",
                "baseline_report",
                "n_notes",
                "n_embedding_lfs",
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
    lines.append("")

    lines.append("## Per-LF firing rates")
    lines.append("")
    lines.append(
        _markdown_table(
            firing_rows,
            [
                "lf_name",
                "field",
                "target_value",
                "n_positive_votes",
                "pct_of_notes",
            ],
        )
    )
    lines.append("")

    lines.append("## LLM agreement sanity check")
    lines.append("")
    lines.append(
        _markdown_table(
            agreement_rows,
            [
                "lf_name",
                "field",
                "embedding_positive_n",
                "llm_yes_n",
                "intersection_n",
                "agreement_pct",
                "recall_from_llm_pct",
            ],
        )
    )
    lines.append("")

    lines.append("## Notes where embedding LF voted POSITIVE but LLM did not")
    lines.append("")
    lines.append(
        _markdown_table(fp_examples, ["hadm_id", "field", "similarity", "evidence"])
        if fp_examples
        else "_No examples._"
    )
    lines.append("")

    lines.append("## Notes where LLM voted yes but embedding LF abstained")
    lines.append("")
    lines.append(
        _markdown_table(
            miss_examples,
            ["hadm_id", "field", "max_similarity_or_reason", "llm_reasoning"],
        )
        if miss_examples
        else "_No examples._"
    )
    lines.append("")

    lines.append("## Section availability")
    lines.append("")
    lines.append(
        _markdown_table(section_rows, ["field", "required_section", "n_present", "pct_of_notes"])
    )
    lines.append("")

    lines.append("## Regex vs embedding comparison")
    lines.append("")
    lines.append(
        _markdown_table(
            comparison_rows,
            [
                "field",
                "regex_positive_n",
                "regex_agreement_pct",
                "regex_recall_pct",
                "embed_positive_n",
                "embed_agreement_pct",
                "embed_recall_pct",
            ],
        )
    )
    lines.append("")

    lines.append("## Embedding v1 vs v2 comparison")
    lines.append("")
    lines.append(
        _markdown_table(
            v2_vs_v1_rows,
            [
                "field",
                "v1_embed_positive_n",
                "v2_embed_positive_n",
                "v1_agreement_pct",
                "v2_agreement_pct",
                "v1_recall_pct",
                "v2_recall_pct",
                "note",
            ],
        )
        if v2_vs_v1_rows
        else "_No comparison rows available._"
    )
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logger.info("Wrote embedding smoke coverage report to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
