"""
Runs preflight calibration before production launch.

Reads: configs/production.yaml, codex_outputs/26_preflight_calibration.md, data/splits/phase6_preflight_100.csv.
Writes: codex_outputs/26_preflight_calibration.md, data/splits/phase6_preflight_100.csv.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/26_preflight_calibration.py` unless the script's argparse help says otherwise.
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
import yaml  # type: ignore[import-untyped]

from src import config
from src.db.connection import get_engine
from src.db.icd_utils import icd10_chapter_from_code
from src.db.queries import fetch_notes_by_hadm_ids, pull_split_candidates
from src.io.splits import build_stratified_splits
from src.llm.batch_runner import BatchSummary, run_batch
from src.llm.client import LLMClient


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


def _read_hadm_ids(path: Path) -> set[int]:
    frame = pd.read_csv(path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").astype("int64").tolist())


def _excluded_hadm_ids(split_dir: Path, include_methodology_5k: bool) -> set[int]:
    expected = [
        split_dir / "refinement_150.csv",
        split_dir / "holdout_150.csv",
        split_dir / "smoke_200.csv",
        split_dir / "methodology_1k.csv",
    ]
    if include_methodology_5k:
        expected.append(split_dir / "methodology_5k.csv")
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required split files missing for exclusion set: {missing}")
    excluded: set[int] = set()
    for path in expected:
        excluded.update(_read_hadm_ids(path))
    return excluded


def _load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 preflight 100-note calibration run.")
    parser.add_argument("--run-id", default="phase6_preflight_a")
    parser.add_argument("--variant", default="a")
    parser.add_argument("--n-notes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--config", default="configs/production.yaml")
    parser.add_argument("--budget-cap-usd", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default="codex_outputs/26_preflight_calibration.md",
    )
    parser.add_argument(
        "--split-output",
        default="data/splits/phase6_preflight_100.csv",
    )
    parser.add_argument(
        "--allow-methodology-5k-overlap",
        action="store_true",
        default=False,
        help="If set, do not exclude hadm_ids in methodology_5k.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    if not config.SETTINGS.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run preflight calibration.")

    run_conf = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    max_concurrency = int(run_conf.get("max_concurrent_requests", config.MAX_CONCURRENT_REQUESTS))
    checkpoint_every = int(run_conf.get("checkpoint_every", 25))

    excluded = _excluded_hadm_ids(
        config.SPLITS_DIR,
        include_methodology_5k=not bool(args.allow_methodology_5k_overlap),
    )

    engine = get_engine()
    candidates = pull_split_candidates(engine).copy()
    candidates = candidates[~candidates["hadm_id"].isin(excluded)].copy()
    if len(candidates) < int(args.n_notes):
        raise RuntimeError(
            f"Not enough candidates after exclusion: need {args.n_notes}, found {len(candidates)}"
        )

    candidates["chapter"] = candidates.apply(
        lambda row: icd10_chapter_from_code(
            str(row["primary_icd_code"]),
            int(row["primary_icd_version"]),
        ),
        axis=1,
    )

    selected = build_stratified_splits(
        candidates,
        seed=int(args.seed),
        refinement_n=int(args.n_notes),
        holdout_n=0,
        smoke_n=0,
    )["refinement"].copy()

    split_output_path = Path(args.split_output)
    split_output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(split_output_path, index=False)

    hadm_ids = sorted(pd.to_numeric(selected["hadm_id"], errors="coerce").astype("int64").tolist())
    notes = fetch_notes_by_hadm_ids(engine, hadm_ids)
    missing = [hadm_id for hadm_id in hadm_ids if hadm_id not in notes]
    if missing:
        raise RuntimeError(f"Missing notes for hadm_ids: {missing[:10]}")

    ordered_notes = {hadm_id: notes[hadm_id] for hadm_id in hadm_ids}
    output_dir = config.RAW_RESPONSES_DIR / args.run_id
    client = LLMClient(
        semaphore_limit=max_concurrency,
        run_id=args.run_id,
        max_budget_usd=float(args.budget_cap_usd),
    )

    summary: BatchSummary = asyncio.run(
        run_batch(
            notes=ordered_notes,
            client=client,
            run_id=args.run_id,
            output_dir=output_dir,
            variant=args.variant,
            include_reasoning=False,
            max_concurrency=max_concurrency,
            checkpoint_every=checkpoint_every,
            resume=False,
        )
    )

    result_rows = _load_results(output_dir / "results.jsonl")
    failures = [row for row in result_rows if not bool(row.get("parse_ok", False))]
    api_errors = [
        row for row in failures if str(row.get("parse_error", "")).startswith("api_error:")
    ]

    passed = (summary.n_failed_parse == 0) and (summary.n_api_error == 0)
    banner = "PASS" if passed else "FAIL"

    run_metadata_path = output_dir / "run_metadata.json"
    run_metadata = {}
    if run_metadata_path.exists():
        loaded = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            run_metadata = loaded

    lines = [
        "# Phase 6 Preflight Calibration",
        "",
        f"## Overall: {banner}",
        "",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_id": args.run_id,
                    "variant": args.variant,
                    "n_attempted": summary.n_total,
                    "n_successful_parse": summary.n_successful_parse,
                    "n_failed_parse": summary.n_failed_parse,
                    "n_api_error": summary.n_api_error,
                    "total_cost_usd": f"{summary.total_cost_usd:.6f}",
                    "elapsed_seconds": f"{summary.elapsed_seconds:.3f}",
                    "max_concurrency": run_metadata.get("client_semaphore_limit", ""),
                    "max_retries": run_metadata.get("max_retries", ""),
                }
            ],
            [
                "timestamp_utc",
                "run_id",
                "variant",
                "n_attempted",
                "n_successful_parse",
                "n_failed_parse",
                "n_api_error",
                "total_cost_usd",
                "elapsed_seconds",
                "max_concurrency",
                "max_retries",
            ],
        ),
        "",
        "## Failure details",
        _markdown_table(
            [
                {
                    "hadm_id": int(row.get("hadm_id", 0) or 0),
                    "parse_error": str(row.get("parse_error", "")),
                }
                for row in failures[:20]
            ],
            ["hadm_id", "parse_error"],
        ),
        "",
        "## Expected criteria",
        "- Target: 0 parse failures and 0 API errors before launching full 5k extraction.",
        f"- Result: parse_failures={summary.n_failed_parse}, api_errors={len(api_errors)}.",
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote preflight report to {output_path}")
    if not passed:
        print("Preflight status: FAIL")
    else:
        print("Preflight status: PASS")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
