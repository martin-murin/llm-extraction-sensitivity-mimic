"""
Runs paired gold/model-size comparison extractions.

Reads: configs/production.yaml, data/splits/SPLITS_MANIFEST.json, codex_outputs/55_paired_extraction_summary.md.
Writes: data/splits/SPLITS_MANIFEST.json, codex_outputs/55_paired_extraction_summary.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/55_run_paired_gold.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from src import config
from src.db.connection import get_engine
from src.db.queries import fetch_notes_by_hadm_ids
from src.llm.batch_runner import BatchSummary, run_batch
from src.llm.client import LLMClient

logger = logging.getLogger("scripts.55_run_paired_gold")

RUN_PLAN: list[tuple[str, str, str]] = [
    ("methodology_1k", "paired_gold_methodology_1k_a", "a"),
    ("methodology_1k", "paired_gold_methodology_1k_b", "b"),
    ("methodology_1k", "paired_gold_methodology_1k_c", "c"),
    ("methodology_5k_audit_500", "paired_gold_methodology_5k_audit_a", "a"),
    ("methodology_5k_audit_500", "paired_gold_methodology_5k_audit_b", "b"),
    ("methodology_5k_audit_500", "paired_gold_methodology_5k_audit_c", "c"),
]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        vals = [str(row.get(col, "")).replace("|", "\\|").replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider, *lines])


def _set_model_id(model_id: str) -> None:
    config.SETTINGS.model_id = model_id
    config.MODEL_ID = model_id


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Manifest is not a dict: {path}")
    return payload


def _verify_split_hashes(manifest_path: Path, split_paths: dict[str, Path]) -> list[dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    checksums = manifest.get("checksums_sha256", {})
    if not isinstance(checksums, dict):
        raise RuntimeError("Manifest missing checksums_sha256 mapping.")

    rows: list[dict[str, Any]] = []
    for split_key, split_path in split_paths.items():
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split: {split_path}")
        observed = _file_sha256(split_path)
        expected = str(checksums.get(split_path.name, ""))
        rows.append(
            {
                "split_key": split_key,
                "split_file": split_path.name,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "match": "yes" if expected == observed else "no",
            }
        )
    return rows


def _load_notes(split_path: Path) -> dict[int, str]:
    split = pd.read_csv(split_path)
    split["hadm_id"] = pd.to_numeric(split["hadm_id"], errors="coerce").astype("int64")
    hadm_ids = split["hadm_id"].astype(int).tolist()
    if len(set(hadm_ids)) != len(hadm_ids):
        raise RuntimeError(f"Duplicate hadm_ids in split: {split_path}")
    engine = get_engine()
    notes = fetch_notes_by_hadm_ids(engine, hadm_ids)
    missing = [h for h in hadm_ids if h not in notes]
    if missing:
        raise RuntimeError(
            f"Missing notes for {len(missing)} hadm_ids in {split_path.name}; first={missing[:10]}"
        )
    return {h: notes[h] for h in hadm_ids}


def _run_one(
    *,
    run_id: str,
    variant: str,
    notes: dict[int, str],
    max_concurrency: int,
    budget_cap_usd: float,
    checkpoint_every: int,
    resume: bool,
) -> BatchSummary:
    output_dir = config.RAW_RESPONSES_DIR / run_id
    client = LLMClient(
        semaphore_limit=max_concurrency,
        run_id=run_id,
        max_budget_usd=budget_cap_usd,
    )
    return asyncio.run(
        run_batch(
            notes=notes,
            client=client,
            run_id=run_id,
            output_dir=output_dir,
            variant=variant,
            include_reasoning=False,
            max_concurrency=max_concurrency,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )
    )


def _build_summary(
    *,
    split_check_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    model_id: str,
    budget_cap_usd: float,
    total_cost_usd: float,
    output_path: Path,
) -> None:
    split_ok = all(str(r["match"]) == "yes" for r in split_check_rows)
    completed_all = all(str(r.get("completed")) == "yes" for r in run_rows)
    lines = [
        "# Prompt 27 Paired Gold Extraction Summary",
        "",
        "## Top-line",
        _markdown_table(
            [
                {"metric": "timestamp_utc", "value": datetime.now(tz=UTC).isoformat()},
                {"metric": "model_id", "value": model_id},
                {"metric": "reasoning", "value": "OFF"},
                {"metric": "max_concurrency", "value": 8},
                {"metric": "split_hash_checks_pass", "value": split_ok},
                {"metric": "all_runs_completed", "value": completed_all},
                {"metric": "budget_cap_usd", "value": f"{budget_cap_usd:.6f}"},
                {"metric": "total_cost_usd", "value": f"{total_cost_usd:.6f}"},
                {
                    "metric": "remaining_budget_usd",
                    "value": f"{max(0.0, budget_cap_usd - total_cost_usd):.6f}",
                },
            ],
            ["metric", "value"],
        ),
        "",
        "## Split SHA-256 verification",
        _markdown_table(
            split_check_rows,
            ["split_key", "split_file", "expected_sha256", "observed_sha256", "match"],
        ),
        "",
        "## Run results",
        _markdown_table(
            run_rows,
            [
                "source_split",
                "run_id",
                "variant",
                "attempted_total",
                "successful_parse",
                "failed_parse",
                "api_error",
                "processed_total",
                "remaining_unprocessed",
                "run_cost_usd",
                "cumulative_cost_usd",
                "completed",
                "halted_by_budget",
            ],
        ),
        "",
        "## Notes",
        (
            "- Priority ordering enforced: all methodology_1k runs execute before "
            "methodology_5k_audit_500."
        ),
        "- If budget is exhausted, remaining runs are marked as skipped.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired full-model extraction on locked splits."
    )
    parser.add_argument("--model-id", default="gpt-5.4-2026-03-05")
    parser.add_argument("--config", default="configs/production.yaml")
    parser.add_argument("--manifest", default="data/splits/SPLITS_MANIFEST.json")
    parser.add_argument("--budget-cap-usd", type=float, default=300.0)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--output", default="codex_outputs/55_paired_extraction_summary.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config.load_env()
    _set_model_id(str(args.model_id))

    if not config.SETTINGS.openai_api_key and not args.skip_run:
        raise RuntimeError("OPENAI_API_KEY missing; cannot run paired gold extraction.")

    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise RuntimeError(f"Invalid YAML: {args.config}")
    max_concurrency = int(settings.get("max_concurrent_requests", config.MAX_CONCURRENT_REQUESTS))
    if max_concurrency != 8:
        raise RuntimeError(f"Expected max_concurrent_requests=8, observed {max_concurrency}")
    checkpoint_every = (
        int(args.checkpoint_every)
        if args.checkpoint_every is not None
        else int(settings.get("checkpoint_every", 50))
    )

    split_paths = {
        "methodology_1k": config.SPLITS_DIR / "methodology_1k.csv",
        "methodology_5k_audit_500": config.SPLITS_DIR / "methodology_5k_audit_500.csv",
    }
    split_check_rows = _verify_split_hashes(Path(args.manifest), split_paths)
    if not all(str(r["match"]) == "yes" for r in split_check_rows):
        raise RuntimeError("Split SHA checks failed; aborting before extraction.")

    notes_by_split = {name: _load_notes(path) for name, path in split_paths.items()}

    run_rows: list[dict[str, Any]] = []
    total_cost_usd = 0.0
    for split_name, run_id, variant in RUN_PLAN:
        remaining_budget = float(args.budget_cap_usd) - total_cost_usd
        if remaining_budget <= 0:
            run_rows.append(
                {
                    "source_split": split_name,
                    "run_id": run_id,
                    "variant": variant,
                    "attempted_total": len(notes_by_split[split_name]),
                    "successful_parse": 0,
                    "failed_parse": 0,
                    "api_error": 0,
                    "processed_total": 0,
                    "remaining_unprocessed": len(notes_by_split[split_name]),
                    "run_cost_usd": "0.000000",
                    "cumulative_cost_usd": f"{total_cost_usd:.6f}",
                    "completed": "no",
                    "halted_by_budget": "yes",
                }
            )
            continue

        if args.skip_run:
            run_rows.append(
                {
                    "source_split": split_name,
                    "run_id": run_id,
                    "variant": variant,
                    "attempted_total": len(notes_by_split[split_name]),
                    "successful_parse": "n/a",
                    "failed_parse": "n/a",
                    "api_error": "n/a",
                    "processed_total": "n/a",
                    "remaining_unprocessed": "n/a",
                    "run_cost_usd": "n/a",
                    "cumulative_cost_usd": "n/a",
                    "completed": "n/a",
                    "halted_by_budget": "n/a",
                }
            )
            continue

        summary = _run_one(
            run_id=run_id,
            variant=variant,
            notes=notes_by_split[split_name],
            max_concurrency=max_concurrency,
            budget_cap_usd=remaining_budget,
            checkpoint_every=checkpoint_every,
            resume=not args.no_resume,
        )
        processed_total = (
            int(summary.n_successful_parse) + int(summary.n_failed_parse) + int(summary.n_api_error)
        )
        n_total = int(summary.n_total)
        unprocessed = max(0, n_total - processed_total)
        run_cost = float(summary.total_cost_usd)
        total_cost_usd += run_cost
        run_rows.append(
            {
                "source_split": split_name,
                "run_id": run_id,
                "variant": variant,
                "attempted_total": n_total,
                "successful_parse": int(summary.n_successful_parse),
                "failed_parse": int(summary.n_failed_parse),
                "api_error": int(summary.n_api_error),
                "processed_total": processed_total,
                "remaining_unprocessed": unprocessed,
                "run_cost_usd": f"{run_cost:.6f}",
                "cumulative_cost_usd": f"{total_cost_usd:.6f}",
                "completed": "yes" if unprocessed == 0 else "no",
                "halted_by_budget": "yes" if unprocessed > 0 else "no",
            }
        )
        if unprocessed > 0:
            logger.warning("Run %s halted before completion (likely budget cap).", run_id)
            break

    output_path = Path(args.output)
    _build_summary(
        split_check_rows=split_check_rows,
        run_rows=run_rows,
        model_id=str(args.model_id),
        budget_cap_usd=float(args.budget_cap_usd),
        total_cost_usd=total_cost_usd,
        output_path=output_path,
    )
    print(f"Wrote summary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
