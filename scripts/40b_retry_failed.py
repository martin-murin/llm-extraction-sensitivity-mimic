from __future__ import annotations

# Release documentation:
# Retries failed production extraction requests.
#
# Reads: configs/production.yaml, codex_outputs/40_retry_failed.md, data/raw_responses.
# Writes: codex_outputs/40_retry_failed.md, data/raw_responses.
# Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
# Usage: `python scripts/40b_retry_failed.py` unless the script's argparse help says otherwise.

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.db.connection import get_engine
from src.db.queries import fetch_notes_by_hadm_ids
from src.llm.batch_runner import run_batch
from src.llm.client import LLMClient

DEFAULT_SOURCE_RUN_ID = "production_v1"
DEFAULT_RETRY_RUN_ID = "production_v1_retry"


def _load_failed_hadm_ids(results_path: Path) -> list[int]:
    failed: list[int] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not bool(row.get("parse_ok", False)):
                failed.append(int(row.get("hadm_id", 0) or 0))
    return sorted(set(failed))


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_retry_successes(results_path: Path) -> set[int]:
    if not results_path.exists():
        return set()
    out: set[int] = set()
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if bool(row.get("parse_ok", False)):
                out.add(int(row.get("hadm_id", 0) or 0))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry failed production rows only.")
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--retry-run-id", default=DEFAULT_RETRY_RUN_ID)
    parser.add_argument("--config", default="configs/production.yaml")
    parser.add_argument("--budget-cap-usd", type=float, default=1.0)
    parser.add_argument("--report", default="codex_outputs/40_retry_failed.md")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    source_dir = Path("data/raw_responses") / args.source_run_id
    source_results = source_dir / "results.jsonl"
    if not source_results.exists():
        raise FileNotFoundError(f"Missing source results file: {source_results}")

    failed_hadm_ids = _load_failed_hadm_ids(source_results)
    source_meta = _load_run_metadata(source_dir)

    retry_dir = Path("data/raw_responses") / args.retry_run_id
    retry_results = retry_dir / "results.jsonl"
    already_success = _load_retry_successes(retry_results)
    pending_hadm_ids = [hadm_id for hadm_id in failed_hadm_ids if hadm_id not in already_success]

    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    max_concurrency = int(settings.get("max_concurrent_requests", 8))
    checkpoint_every = int(settings.get("checkpoint_every", 50))

    variant = str(source_meta.get("variant", "a"))
    include_reasoning = bool(source_meta.get("include_reasoning", False))

    started = datetime.now(tz=UTC).isoformat()

    run_status = "not_started"
    error_text = ""
    summary_payload: dict[str, Any] = {}

    if not pending_hadm_ids:
        run_status = "nothing_to_retry"
    else:
        try:
            engine = get_engine()
            notes = fetch_notes_by_hadm_ids(engine, pending_hadm_ids)
            missing = [hadm_id for hadm_id in pending_hadm_ids if hadm_id not in notes]
            if missing:
                raise RuntimeError(
                    "Missing discharge note text for failed hadm_ids; "
                    f"sample_missing={missing[:10]}"
                )

            ordered_notes = {hadm_id: notes[hadm_id] for hadm_id in pending_hadm_ids}

            if not config.SETTINGS.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")

            client = LLMClient(
                semaphore_limit=max_concurrency,
                run_id=args.retry_run_id,
                max_budget_usd=float(args.budget_cap_usd),
            )

            batch_summary = asyncio.run(
                run_batch(
                    notes=ordered_notes,
                    client=client,
                    run_id=args.retry_run_id,
                    output_dir=retry_dir,
                    variant=variant,
                    include_reasoning=include_reasoning,
                    max_concurrency=max_concurrency,
                    checkpoint_every=checkpoint_every,
                    resume=not args.no_resume,
                )
            )
            run_status = "completed"
            summary_payload = {
                "n_total": int(batch_summary.n_total),
                "n_successful_parse": int(batch_summary.n_successful_parse),
                "n_failed_parse": int(batch_summary.n_failed_parse),
                "n_api_error": int(batch_summary.n_api_error),
                "total_cost_usd": float(batch_summary.total_cost_usd),
                "elapsed_seconds": float(batch_summary.elapsed_seconds),
                "total_input_tokens": int(batch_summary.total_input_tokens),
                "total_output_tokens": int(batch_summary.total_output_tokens),
            }
        except Exception as exc:
            run_status = "failed"
            error_text = f"{type(exc).__name__}: {exc}"

    final_retry_success = _load_retry_successes(retry_results)
    recovered = len(final_retry_success.intersection(set(failed_hadm_ids)))
    remaining_failed = len(failed_hadm_ids) - recovered

    report_rows = [
        {"metric": "started_at_utc", "value": started},
        {"metric": "completed_at_utc", "value": datetime.now(tz=UTC).isoformat()},
        {"metric": "source_run_id", "value": args.source_run_id},
        {"metric": "retry_run_id", "value": args.retry_run_id},
        {"metric": "budget_cap_usd", "value": args.budget_cap_usd},
        {"metric": "variant", "value": variant},
        {"metric": "include_reasoning", "value": include_reasoning},
        {"metric": "failed_hadm_ids_source", "value": len(failed_hadm_ids)},
        {"metric": "already_successful_in_retry_dir", "value": len(already_success)},
        {"metric": "pending_hadm_ids_this_run", "value": len(pending_hadm_ids)},
        {"metric": "run_status", "value": run_status},
        {"metric": "recovered_after_retry", "value": recovered},
        {"metric": "remaining_failed_after_retry", "value": remaining_failed},
        {
            "metric": "post_retry_success_rate_over_failed_pool",
            "value": (
                f"{(recovered / len(failed_hadm_ids) * 100):.2f}%"
                if failed_hadm_ids
                else "0.00%"
            ),
        },
        {"metric": "error", "value": error_text or ""},
    ]

    for key in [
        "n_total",
        "n_successful_parse",
        "n_failed_parse",
        "n_api_error",
        "total_cost_usd",
        "elapsed_seconds",
        "total_input_tokens",
        "total_output_tokens",
    ]:
        report_rows.append({"metric": f"run_summary.{key}", "value": summary_payload.get(key, "")})

    def md_table(rows: list[dict[str, Any]]) -> str:
        header = "| metric | value |"
        divider = "|---|---|"
        body = [
            "| "
            + str(row["metric"]).replace("|", "\\|")
            + " | "
            + str(row["value"]).replace("|", "\\|")
            + " |"
            for row in rows
        ]
        return "\n".join([header, divider, *body])

    lines = [
        "# Failed-only Retry Report",
        "",
        md_table(report_rows),
        "",
        "## Notes",
        "- This script retries only hadm_ids with `parse_ok=false` from source run.",
        (
            "- No mutation is performed on source `production_v1/results.jsonl`; retry outputs "
            "are written to the retry run directory."
        ),
        "",
    ]

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"run_status={run_status}")
    print(f"failed_pool={len(failed_hadm_ids)} recovered={recovered} remaining={remaining_failed}")
    if summary_payload:
        print(f"retry_cost_usd={summary_payload.get('total_cost_usd', 0.0):.6f}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
