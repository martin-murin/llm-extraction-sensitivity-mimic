"""
Runs larger-model proxy/gold extraction variants.

Reads: data/splits/gold_1k.csv, data/splits/SPLITS_MANIFEST.json, codex_outputs/45_gold_extraction_summary.md, configs/production.yaml.
Writes: data/splits/gold_1k.csv, data/splits/SPLITS_MANIFEST.json, codex_outputs/45_gold_extraction_summary.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/45_run_gold_standard.py` unless the script's argparse help says otherwise.
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

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from src import config
from src.db.connection import get_engine
from src.db.icd_utils import icd10_chapter_from_code
from src.db.queries import fetch_notes_by_hadm_ids, pull_split_candidates
from src.io.splits import build_stratified_splits
from src.llm.batch_runner import BatchSummary, run_batch
from src.llm.client import LLMClient
from src.utils.threeway_kappa import file_sha256

logger = logging.getLogger("scripts.45_run_gold_standard")

RUN_PLAN: list[tuple[str, str]] = [
    ("gold_v1_a", "a"),
    ("gold_v1_b", "b"),
    ("gold_v1_c", "c"),
]
EXCLUSION_SPLITS: list[str] = [
    "refinement_150.csv",
    "holdout_150.csv",
    "smoke_200.csv",
    "methodology_1k.csv",
    "methodology_5k.csv",
]


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


def _read_split_hadm_ids(path: Path) -> set[int]:
    frame = pd.read_csv(path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").dropna().astype("int64").tolist())


def _read_production_hadm_ids(results_path: Path) -> set[int]:
    if not results_path.exists():
        raise FileNotFoundError(f"Missing production results.jsonl: {results_path}")
    hadm_ids: set[int] = set()
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            hadm_id = payload.get("hadm_id")
            if hadm_id is None:
                continue
            hadm_ids.add(int(hadm_id))
    return hadm_ids


def _chapter_balance_rows(
    *,
    population: pd.DataFrame,
    sampled: pd.DataFrame,
) -> tuple[list[dict[str, Any]], bool]:
    pop_counts = population["chapter"].value_counts(normalize=True)
    sample_counts = sampled["chapter"].value_counts(normalize=True)
    chapters = sorted(set(pop_counts.index.tolist()) | set(sample_counts.index.tolist()))
    rows: list[dict[str, Any]] = []
    all_within = True
    for chapter in chapters:
        pop_pct = float(pop_counts.get(chapter, 0.0) * 100.0)
        sample_pct = float(sample_counts.get(chapter, 0.0) * 100.0)
        delta_pp = abs(sample_pct - pop_pct)
        if delta_pp > 2.0:
            all_within = False
        rows.append(
            {
                "chapter": chapter,
                "population_pct": f"{pop_pct:.2f}",
                "sample_pct": f"{sample_pct:.2f}",
                "abs_delta_pp": f"{delta_pp:.2f}",
                "within_2pp": "yes" if delta_pp <= 2.0 else "no",
            }
        )
    rows.sort(key=lambda row: float(row["abs_delta_pp"]), reverse=True)
    return rows, all_within


def _set_model_id(model_id: str) -> None:
    config.SETTINGS.model_id = model_id
    config.MODEL_ID = model_id


def _build_gold_split(
    *,
    seed: int,
    n_notes: int,
    output_csv: Path,
) -> dict[str, Any]:
    production_results = config.RAW_RESPONSES_DIR / "production_v1" / "results.jsonl"
    production_hadm_ids = _read_production_hadm_ids(production_results)
    if len(production_hadm_ids) < n_notes:
        found = len(production_hadm_ids)
        raise RuntimeError(
            "Production pool too small for requested sample: "
            f"need {n_notes}, got {found}"
        )

    split_dir = config.SPLITS_DIR
    excluded_by_split: dict[str, set[int]] = {}
    for split_name in EXCLUSION_SPLITS:
        split_path = split_dir / split_name
        if not split_path.exists():
            raise FileNotFoundError(f"Required exclusion split missing: {split_path}")
        excluded_by_split[split_name] = _read_split_hadm_ids(split_path)

    excluded_union: set[int] = set()
    for ids in excluded_by_split.values():
        excluded_union.update(ids)

    engine = get_engine()
    candidates = pull_split_candidates(engine).copy()
    candidates = candidates[candidates["hadm_id"].isin(production_hadm_ids)].copy()
    candidates = candidates[~candidates["hadm_id"].isin(excluded_union)].copy()

    if len(candidates) < n_notes:
        raise RuntimeError(
            f"Not enough candidates after exclusions: need {n_notes}, got {len(candidates)}."
        )

    candidates["chapter"] = candidates.apply(
        lambda row: icd10_chapter_from_code(
            str(row["primary_icd_code"]),
            int(row["primary_icd_version"]),
        ),
        axis=1,
    )

    sampled = build_stratified_splits(
        candidates,
        seed=seed,
        refinement_n=n_notes,
        holdout_n=0,
        smoke_n=0,
    )["refinement"].copy()

    ordered_cols = [
        "hadm_id",
        "subject_id",
        "primary_icd_code",
        "primary_icd_version",
        "chapter",
        "note_char_len",
        "n_diagnoses",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled[ordered_cols].to_csv(output_csv, index=False)

    hadm_ids = set(pd.to_numeric(sampled["hadm_id"], errors="coerce").astype("int64").tolist())
    if len(hadm_ids) != n_notes:
        raise RuntimeError(
            f"Gold split uniqueness check failed: expected {n_notes}, got {len(hadm_ids)}"
        )

    overlap_counts: dict[str, int] = {}
    for split_name, ids in excluded_by_split.items():
        overlap_counts[split_name] = len(hadm_ids.intersection(ids))

    chapter_rows, chapter_ok = _chapter_balance_rows(population=candidates, sampled=sampled)
    checksum = file_sha256(output_csv)

    return {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "seed": seed,
        "target_n": n_notes,
        "actual_n": len(hadm_ids),
        "source_pool_n_after_exclusions": len(candidates),
        "source_production_hadm_n": len(production_hadm_ids),
        "sha256": checksum,
        "output_csv": str(output_csv),
        "chapter_rows": chapter_rows,
        "chapter_balance_within_2pp": chapter_ok,
        "overlap_counts": overlap_counts,
    }


def _load_notes_for_split(split_csv: Path) -> dict[int, str]:
    split_frame = pd.read_csv(split_csv)
    split_frame["hadm_id"] = pd.to_numeric(split_frame["hadm_id"], errors="coerce").astype("int64")
    hadm_ids = split_frame["hadm_id"].astype(int).tolist()
    engine = get_engine()
    notes = fetch_notes_by_hadm_ids(engine, hadm_ids)
    missing = [hadm_id for hadm_id in hadm_ids if hadm_id not in notes]
    if missing:
        raise RuntimeError(
            f"Missing discharge notes for {len(missing)} hadm_ids; first={missing[:10]}"
        )
    return {int(hadm_id): notes[int(hadm_id)] for hadm_id in hadm_ids}


def _run_one_variant(
    *,
    run_id: str,
    variant: str,
    notes: dict[int, str],
    max_concurrency: int,
    budget_cap_usd: float,
    checkpoint_every: int,
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
            resume=True,
        )
    )


def _build_report(
    *,
    split_summary: dict[str, Any],
    run_rows: list[dict[str, Any]],
    total_cost_spent: float,
    budget_cap_usd: float,
    model_id: str,
    output_path: Path,
) -> None:
    overlap_rows = [
        {"split": split_name, "overlap_count": count}
        for split_name, count in split_summary["overlap_counts"].items()
    ]
    overlap_zero_ok = all(int(row["overlap_count"]) == 0 for row in overlap_rows)

    top_rows = [
        {"metric": "timestamp_utc", "value": split_summary["timestamp_utc"]},
        {"metric": "model_id", "value": model_id},
        {"metric": "seed", "value": split_summary["seed"]},
        {"metric": "target_n", "value": split_summary["target_n"]},
        {"metric": "actual_n", "value": split_summary["actual_n"]},
        {"metric": "source_production_hadm_n", "value": split_summary["source_production_hadm_n"]},
        {
            "metric": "source_pool_n_after_exclusions",
            "value": split_summary["source_pool_n_after_exclusions"],
        },
        {"metric": "gold_1k_csv", "value": split_summary["output_csv"]},
        {"metric": "gold_1k_sha256", "value": split_summary["sha256"]},
        {
            "metric": "chapter_distribution_within_2pp",
            "value": split_summary["chapter_balance_within_2pp"],
        },
        {"metric": "zero_overlap_required_splits", "value": overlap_zero_ok},
        {"metric": "budget_cap_usd", "value": f"{budget_cap_usd:.6f}"},
        {"metric": "total_cost_spent_usd", "value": f"{total_cost_spent:.6f}"},
        {
            "metric": "remaining_budget_usd",
            "value": f"{max(0.0, budget_cap_usd - total_cost_spent):.6f}",
        },
        {
            "metric": "manifest_update_status",
            "value": split_summary.get("manifest_update_status", "unknown"),
        },
    ]

    lines = [
        "# Prompt 26 Gold Extraction Summary",
        "",
        "## Top-line",
        _markdown_table(top_rows, ["metric", "value"]),
        "",
        "## Required overlap checks",
        _markdown_table(overlap_rows, ["split", "overlap_count"]),
        "",
        "## Chapter balance (post-exclusion population vs gold_1k sample)",
        _markdown_table(
            split_summary["chapter_rows"],
            ["chapter", "population_pct", "sample_pct", "abs_delta_pp", "within_2pp"],
        ),
        "",
        "## Gold extraction runs",
        _markdown_table(
            run_rows,
            [
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
                "halted_by_budget",
                "completed_1000",
            ],
        ),
        "",
        "## Notes",
        "- Reasoning set to OFF for all gold runs.",
        "- Concurrency held at 8 (production rate-limit setting).",
        "- Retry policy comes from unchanged `src/llm/client.py` / config (`max_retries=5`).",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Prompt 26 gold split and run gold extractions."
    )
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--n-notes", type=int, default=1000)
    parser.add_argument("--budget-cap-usd", type=float, default=200.0)
    parser.add_argument("--model-id", default="gpt-5.4-2026-03-05")
    parser.add_argument("--split-output", default="data/splits/gold_1k.csv")
    parser.add_argument("--manifest", default="data/splits/SPLITS_MANIFEST.json")
    parser.add_argument("--report-output", default="codex_outputs/45_gold_extraction_summary.md")
    parser.add_argument("--config", default="configs/production.yaml")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def _update_manifest(
    *,
    manifest_path: Path,
    split_output: Path,
    seed: int,
    n_notes: int,
    checksum: str,
) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = payload.get("checksums_sha256")
    if not isinstance(checksums, dict):
        raise RuntimeError("Manifest missing checksums_sha256 mapping.")
    checksums["gold_1k"] = checksum
    checksums[split_output.name] = checksum
    payload["checksums_sha256"] = checksums
    payload["gold_1k"] = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "seed": int(seed),
        "n_notes": int(n_notes),
        "source": "production_v1_excluding_refinement_holdout_smoke_methodology_1k_methodology_5k",
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config.load_env()
    _set_model_id(str(args.model_id))

    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise RuntimeError(f"Invalid YAML in {args.config}")
    max_concurrency = int(settings.get("max_concurrent_requests", 8))
    checkpoint_raw = settings.get("checkpoint_every", settings.get("checkpoint_batch_size", 50))
    checkpoint_every = int(checkpoint_raw if checkpoint_raw is not None else 50)
    if max_concurrency != 8:
        raise RuntimeError(
            f"Production concurrency mismatch: observed {max_concurrency}, expected 8"
        )
    if not config.SETTINGS.openai_api_key and not args.skip_run:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run gold extraction.")

    split_output = Path(args.split_output)
    split_summary = _build_gold_split(
        seed=int(args.seed),
        n_notes=int(args.n_notes),
        output_csv=split_output,
    )
    manifest_path = Path(args.manifest)
    _update_manifest(
        manifest_path=manifest_path,
        split_output=split_output,
        seed=int(args.seed),
        n_notes=int(args.n_notes),
        checksum=str(split_summary["sha256"]),
    )
    split_summary["manifest_update_status"] = f"updated:{manifest_path}"

    run_rows: list[dict[str, Any]] = []
    total_cost_spent = 0.0
    halted = False

    if not args.skip_run:
        notes = _load_notes_for_split(split_output)
        for run_id, variant in RUN_PLAN:
            remaining_budget = float(args.budget_cap_usd) - total_cost_spent
            if remaining_budget <= 0:
                halted = True
                run_rows.append(
                    {
                        "run_id": run_id,
                        "variant": variant,
                        "attempted_total": int(args.n_notes),
                        "successful_parse": 0,
                        "failed_parse": 0,
                        "api_error": 0,
                        "processed_total": 0,
                        "remaining_unprocessed": int(args.n_notes),
                        "run_cost_usd": "0.000000",
                        "cumulative_cost_usd": f"{total_cost_spent:.6f}",
                        "halted_by_budget": "yes",
                        "completed_1000": "no",
                    }
                )
                continue

            summary = _run_one_variant(
                run_id=run_id,
                variant=variant,
                notes=notes,
                max_concurrency=max_concurrency,
                budget_cap_usd=remaining_budget,
                checkpoint_every=checkpoint_every,
            )
            processed_total = (
                int(summary.n_successful_parse)
                + int(summary.n_failed_parse)
                + int(summary.n_api_error)
            )
            unprocessed = max(0, int(summary.n_total) - processed_total)
            halted_by_budget = unprocessed > 0
            total_cost_spent += float(summary.total_cost_usd)
            run_rows.append(
                {
                    "run_id": run_id,
                    "variant": variant,
                    "attempted_total": int(summary.n_total),
                    "successful_parse": int(summary.n_successful_parse),
                    "failed_parse": int(summary.n_failed_parse),
                    "api_error": int(summary.n_api_error),
                    "processed_total": processed_total,
                    "remaining_unprocessed": unprocessed,
                    "run_cost_usd": f"{float(summary.total_cost_usd):.6f}",
                    "cumulative_cost_usd": f"{total_cost_spent:.6f}",
                    "halted_by_budget": "yes" if halted_by_budget else "no",
                    "completed_1000": "yes" if processed_total == int(summary.n_total) else "no",
                }
            )
            if halted_by_budget:
                halted = True
                break

    if not run_rows:
        for run_id, variant in RUN_PLAN:
            run_rows.append(
                {
                    "run_id": run_id,
                    "variant": variant,
                    "attempted_total": int(args.n_notes),
                    "successful_parse": "n/a",
                    "failed_parse": "n/a",
                    "api_error": "n/a",
                    "processed_total": "n/a",
                    "remaining_unprocessed": "n/a",
                    "run_cost_usd": "n/a",
                    "cumulative_cost_usd": "n/a",
                    "halted_by_budget": "n/a",
                    "completed_1000": "n/a",
                }
            )

    report_output = Path(args.report_output)
    _build_report(
        split_summary=split_summary,
        run_rows=run_rows,
        total_cost_spent=total_cost_spent,
        budget_cap_usd=float(args.budget_cap_usd),
        model_id=str(args.model_id),
        output_path=report_output,
    )

    print(f"Wrote gold split: {split_output}")
    print(f"Wrote extraction summary: {report_output}")
    if halted:
        print("Gold extraction halted early due to budget cap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
