"""
Builds and runs the extended 5k comparison sample.

Reads: data/splits/extended_5k.csv, data/splits/SPLITS_MANIFEST.json, codex_outputs/46_verification.md.
Writes: data/splits/extended_5k.csv, data/splits/SPLITS_MANIFEST.json, codex_outputs/46_verification.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/46_run_extended_5k.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from src import config
from src.db.connection import get_engine
from src.db.icd_utils import icd10_chapter_from_code
from src.db.queries import fetch_notes_by_hadm_ids, pull_split_candidates
from src.io.splits import build_stratified_splits
from src.llm.batch_runner import BatchSummary, run_batch
from src.llm.client import LLMClient
from src.utils.threeway_kappa import file_sha256

DEFAULT_OVERLAP_FILES = [
    "refinement_150.csv",
    "holdout_150.csv",
    "smoke_200.csv",
    "methodology_1k.csv",
    "methodology_5k.csv",
    "gold_1k.csv",
]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        vals = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider, *lines])


def _read_hadm_ids(split_path: Path) -> set[int]:
    frame = pd.read_csv(split_path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").astype("int64").tolist())


def _load_overlap_sets(
    split_dir: Path, expected_files: list[str]
) -> tuple[dict[str, set[int]], list[str]]:
    overlap_sets: dict[str, set[int]] = {}
    missing: list[str] = []
    for name in expected_files:
        path = split_dir / name
        if not path.exists():
            missing.append(name)
            continue
        overlap_sets[name] = _read_hadm_ids(path)
    return overlap_sets, missing


def _load_production_parse_ok_hadm_ids(run_id: str) -> set[int]:
    results_path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing production results file: {results_path}")

    hadm_ids: set[int] = set()
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not bool(row.get("parse_ok", False)):
                continue
            if not isinstance(row.get("features_json"), dict):
                continue
            hadm_ids.add(int(row["hadm_id"]))
    if not hadm_ids:
        raise RuntimeError("No parse-ok admissions found in production results.")
    return hadm_ids


def _chapter_balance_rows(
    population: pd.DataFrame, sampled: pd.DataFrame
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
        rows.append(
            {
                "chapter": chapter,
                "population_pct": f"{pop_pct:.2f}",
                "sample_pct": f"{sample_pct:.2f}",
                "abs_delta_pp": f"{delta_pp:.2f}",
                "within_2pp": "yes" if delta_pp <= 2.0 else "no",
            }
        )
        if delta_pp > 2.0:
            all_within = False

    rows.sort(key=lambda row: float(row["abs_delta_pp"]), reverse=True)
    return rows, all_within


def _load_notes_for_split(split_path: Path, engine: Any) -> dict[int, str]:
    split_frame = pd.read_csv(split_path)
    split_frame["hadm_id"] = pd.to_numeric(split_frame["hadm_id"], errors="coerce").astype("int64")
    hadm_ids = split_frame.sort_values("hadm_id", kind="mergesort")["hadm_id"].astype(int).tolist()

    notes = fetch_notes_by_hadm_ids(engine, hadm_ids)
    missing = [hadm_id for hadm_id in hadm_ids if hadm_id not in notes]
    if missing:
        raise RuntimeError(f"Missing discharge notes for hadm_ids: {missing[:10]}")

    return {hadm_id: notes[hadm_id] for hadm_id in hadm_ids}


def _run_variant(
    *,
    run_id: str,
    variant: str,
    notes: dict[int, str],
    budget_cap_usd: float,
    max_concurrency: int,
) -> BatchSummary:
    if not config.SETTINGS.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run extraction.")

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
            checkpoint_every=50,
            resume=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build extended 5k split and run variants B/C.")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--n-notes", type=int, default=5000)
    parser.add_argument("--production-run-id", default="production_v1")
    parser.add_argument("--split-output", default="data/splits/extended_5k.csv")
    parser.add_argument("--manifest", default="data/splits/SPLITS_MANIFEST.json")
    parser.add_argument("--run-id-b", default="extended_5k_b")
    parser.add_argument("--run-id-c", default="extended_5k_c")
    parser.add_argument("--hard-cap-usd", type=float, default=30.0)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--skip-runs", action="store_true", default=False)
    parser.add_argument("--skip-split-build", action="store_true", default=False)
    parser.add_argument(
        "--variants",
        default="b,c",
        help="Comma-separated subset of variants to run (allowed: b,c).",
    )
    parser.add_argument("--report", default="codex_outputs/46_verification.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    split_path = Path(args.split_output)
    manifest_path = Path(args.manifest)
    report_path = Path(args.report)

    split_dir = config.SPLITS_DIR
    overlap_sets, missing_overlap_files = _load_overlap_sets(split_dir, DEFAULT_OVERLAP_FILES)
    excluded: set[int] = set()
    for ids in overlap_sets.values():
        excluded.update(ids)

    chapter_rows: list[dict[str, Any]] = []
    chapter_ok = True

    if args.skip_split_build:
        if not split_path.exists():
            raise FileNotFoundError(f"--skip-split-build requested but split missing: {split_path}")
        sampled = pd.read_csv(split_path)
        sampled_hadm_ids = set(
            pd.to_numeric(sampled["hadm_id"], errors="coerce").astype("int64").tolist()
        )
    else:
        engine = get_engine()
        candidates = pull_split_candidates(engine).copy()

        production_hadm_ids = _load_production_parse_ok_hadm_ids(str(args.production_run_id))
        production_candidates = candidates[candidates["hadm_id"].isin(production_hadm_ids)].copy()
        if len(production_candidates) < int(args.n_notes):
            raise RuntimeError(
                "Not enough production parse-ok candidates to draw extended split: "
                f"need {args.n_notes}, found {len(production_candidates)}"
            )

        eligible = production_candidates[~production_candidates["hadm_id"].isin(excluded)].copy()
        if len(eligible) < int(args.n_notes):
            raise RuntimeError(
                "Not enough eligible production candidates after split-exclusion filtering: "
                f"need {args.n_notes}, found {len(eligible)}"
            )

        eligible["chapter"] = eligible.apply(
            lambda row: icd10_chapter_from_code(
                str(row["primary_icd_code"]),
                int(row["primary_icd_version"]),
            ),
            axis=1,
        )

        sampled = build_stratified_splits(
            eligible,
            seed=int(args.seed),
            refinement_n=int(args.n_notes),
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
        split_path.parent.mkdir(parents=True, exist_ok=True)
        sampled[ordered_cols].to_csv(split_path, index=False)

        sampled_hadm_ids = set(
            pd.to_numeric(sampled["hadm_id"], errors="coerce").astype("int64").tolist()
        )
        chapter_rows, chapter_ok = _chapter_balance_rows(eligible, sampled)
    overlap_rows: list[dict[str, Any]] = []
    for name in DEFAULT_OVERLAP_FILES:
        ids = overlap_sets.get(name, set())
        overlap_rows.append(
            {
                "split": name,
                "exists": "yes" if name in overlap_sets else "no",
                "n_ids": len(ids),
                "overlap_count": len(sampled_hadm_ids.intersection(ids)),
            }
        )

    checksum = file_sha256(split_path)
    manifest_checksum_before = ""
    manifest_entry_exists = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksums = manifest.get("checksums_sha256", {})
        if isinstance(checksums, dict):
            manifest_checksum_before = str(checksums.get(split_path.name, ""))
            manifest_entry_exists = split_path.name in checksums

    run_rows: list[dict[str, Any]] = []
    cap_hit = False
    total_run_cost = 0.0

    if not args.skip_runs:
        engine = get_engine()
        notes = _load_notes_for_split(split_path, engine)
        requested = [v.strip().lower() for v in str(args.variants).split(",") if v.strip()]
        allowed = {"b": str(args.run_id_b), "c": str(args.run_id_c)}
        invalid = [v for v in requested if v not in allowed]
        if invalid:
            raise ValueError(f"Unsupported variants requested: {invalid}. Allowed: b,c")
        run_plan = [(allowed[v], v) for v in requested]
        if not run_plan:
            raise ValueError("No variants selected. Use --variants b,c or subset.")
        per_run_cap = float(args.hard_cap_usd) / max(1, len(run_plan))

        for run_id, variant in run_plan:
            summary = _run_variant(
                run_id=run_id,
                variant=variant,
                notes=notes,
                budget_cap_usd=per_run_cap,
                max_concurrency=int(args.max_concurrency),
            )
            attempted = int(
                summary.n_successful_parse + summary.n_failed_parse + summary.n_api_error
            )
            complete = attempted >= int(summary.n_total)
            cost = float(summary.total_cost_usd)
            total_run_cost += cost
            run_rows.append(
                {
                    "run_id": run_id,
                    "variant": variant,
                    "n_total": int(summary.n_total),
                    "n_attempted": attempted,
                    "n_successful_parse": int(summary.n_successful_parse),
                    "n_failed_parse": int(summary.n_failed_parse),
                    "n_api_error": int(summary.n_api_error),
                    "complete": complete,
                    "cost_usd": f"{cost:.4f}",
                    "reasoning": "off",
                }
            )
            if not complete:
                cap_hit = True

    now = datetime.now(tz=UTC).isoformat()
    summary_rows: list[dict[str, Any]] = [
        {"metric": "timestamp_utc", "value": now},
        {"metric": "seed", "value": int(args.seed)},
        {"metric": "n_notes_target", "value": int(args.n_notes)},
        {"metric": "n_notes_sampled", "value": len(sampled_hadm_ids)},
        {"metric": "sample_source", "value": f"{args.production_run_id} parse_ok admissions"},
        {"metric": "split_output", "value": str(split_path)},
        {"metric": "split_sha256", "value": checksum},
        {"metric": "manifest_path", "value": str(manifest_path)},
        {"metric": "manifest_entry_exists_before", "value": manifest_entry_exists},
        {"metric": "manifest_checksum_before", "value": manifest_checksum_before},
        {"metric": "manifest_checksum_target", "value": checksum},
        {"metric": "missing_overlap_files", "value": ", ".join(missing_overlap_files) or "none"},
        {"metric": "chapter_distribution_within_2pp", "value": chapter_ok},
        {"metric": "run_cost_total_usd", "value": f"{total_run_cost:.4f}"},
        {"metric": "hard_cap_usd", "value": f"{float(args.hard_cap_usd):.2f}"},
        {"metric": "cap_hit_or_incomplete", "value": cap_hit},
    ]

    report_lines = [
        "# Extended 5k Build and Run Verification",
        "",
        "## Summary",
        _markdown_table(summary_rows, ["metric", "value"]),
        "",
        "## Overlap checks",
        _markdown_table(overlap_rows, ["split", "exists", "n_ids", "overlap_count"]),
        "",
        "## Chapter distribution check (eligible production vs extended_5k)",
        _markdown_table(
            chapter_rows,
            ["chapter", "population_pct", "sample_pct", "abs_delta_pp", "within_2pp"],
        ),
        "",
        "## B/C run summaries",
        _markdown_table(
            run_rows,
            [
                "run_id",
                "variant",
                "n_total",
                "n_attempted",
                "n_successful_parse",
                "n_failed_parse",
                "n_api_error",
                "complete",
                "cost_usd",
                "reasoning",
            ],
        ),
        "",
        "## Notes",
        "- This script does not modify `data/splits/SPLITS_MANIFEST.json`.",
        "- Use `manifest_checksum_target` above to update manifest SHA for `extended_5k.csv`.",
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    if len(sampled_hadm_ids) != int(args.n_notes):
        raise RuntimeError(f"Expected {args.n_notes} sampled ids, got {len(sampled_hadm_ids)}")

    for row in overlap_rows:
        if row["exists"] == "yes" and int(row["overlap_count"]) != 0:
            raise RuntimeError(
                f"Overlap violation with {row['split']}: {row['overlap_count']} admissions overlap"
            )

    if not args.skip_runs and cap_hit:
        raise RuntimeError(
            "At least one extended run is incomplete (likely budget cap or terminal errors)."
        )

    print(f"Wrote split CSV: {split_path}")
    print(f"Wrote verification report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
