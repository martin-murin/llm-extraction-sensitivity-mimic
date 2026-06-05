"""
Monitors production extraction output and failure state.

Reads: data/raw_responses.
Writes: data/raw_responses.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/32_monitor_production.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import time
from datetime import UTC, datetime
from typing import Any

from src import config

TOTAL_NOTES_DEFAULT = 331_793


def _parse_iso(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _format_hms(seconds: float) -> str:
    if seconds <= 0:
        return "0m"
    seconds_int = int(seconds)
    hours, rem = divmod(seconds_int, 3600)
    mins, _ = divmod(rem, 60)
    return f"{hours}h {mins}m" if hours else f"{mins}m"


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _scan_results(results_path: Path) -> dict[str, Any]:
    n_attempted = 0
    n_success = 0
    n_parse_fail = 0
    n_api_error = 0
    total_input_tokens = 0
    total_output_tokens = 0
    last_latencies: list[float] = []

    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n_attempted += 1
            parse_ok = bool(row.get("parse_ok"))
            parse_error = row.get("parse_error")
            if parse_ok:
                n_success += 1
            elif isinstance(parse_error, str) and parse_error.startswith("api_error:"):
                n_api_error += 1
            else:
                n_parse_fail += 1

            total_input_tokens += int(row.get("input_tokens", 0) or 0)
            total_output_tokens += int(row.get("output_tokens", 0) or 0)
            latency = float(row.get("latency_seconds", 0.0) or 0.0)
            if latency > 0:
                last_latencies.append(latency)
                if len(last_latencies) > 1000:
                    last_latencies.pop(0)

    return {
        "n_attempted": n_attempted,
        "n_success": n_success,
        "n_parse_fail": n_parse_fail,
        "n_api_error": n_api_error,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "last_latencies": last_latencies,
    }


def _compute_summary(run_id: str, total_notes: int) -> dict[str, Any]:
    run_dir = Path("data/raw_responses") / run_id
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    stats = _scan_results(results_path)
    metadata = _load_run_metadata(run_dir)

    now = datetime.now(tz=UTC)
    started = _parse_iso(metadata.get("started_at_utc") if isinstance(metadata, dict) else None)
    elapsed_seconds = (now - started).total_seconds() if started else 0.0

    n_attempted = int(stats["n_attempted"])
    n_success = int(stats["n_success"])
    n_parse_fail = int(stats["n_parse_fail"])
    n_api_error = int(stats["n_api_error"])

    parse_fail_rate = (n_parse_fail / n_attempted * 100.0) if n_attempted else 0.0
    api_error_rate = (n_api_error / n_attempted * 100.0) if n_attempted else 0.0

    input_cost = (stats["total_input_tokens"] / 1_000_000) * config.INPUT_PRICE_PER_MILLION_USD
    output_cost = (stats["total_output_tokens"] / 1_000_000) * config.OUTPUT_PRICE_PER_MILLION_USD
    spent = input_cost + output_cost
    mean_cost = (spent / n_attempted) if n_attempted else 0.0
    projected_total = mean_cost * total_notes

    overall_rps = (n_attempted / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    remaining = max(total_notes - n_attempted, 0)
    eta_seconds = (remaining / overall_rps) if overall_rps > 0 else 0.0

    last_latencies = list(stats["last_latencies"])
    window_n = len(last_latencies)
    window_rps = (window_n / sum(last_latencies)) if window_n and sum(last_latencies) > 0 else 0.0

    eta_wall = now.timestamp() + eta_seconds if eta_seconds > 0 else None
    eta_dt = datetime.fromtimestamp(eta_wall, tz=UTC) if eta_wall is not None else None

    return {
        "now": now,
        "run_id": run_id,
        "total_notes": total_notes,
        "n_attempted": n_attempted,
        "n_success": n_success,
        "n_parse_fail": n_parse_fail,
        "n_api_error": n_api_error,
        "parse_fail_rate": parse_fail_rate,
        "api_error_rate": api_error_rate,
        "spent": spent,
        "projected_total": projected_total,
        "window_rps": window_rps,
        "window_n": window_n,
        "elapsed_seconds": elapsed_seconds,
        "eta_dt": eta_dt,
        "eta_seconds": eta_seconds,
        "progress_pct": (n_attempted / total_notes * 100.0) if total_notes else 0.0,
    }


def _render(summary: dict[str, Any]) -> str:
    eta_dt = summary["eta_dt"]
    eta_txt = eta_dt.strftime("%Y-%m-%d %H:%M UTC") if eta_dt else "N/A"
    rem_txt = _format_hms(float(summary["eta_seconds"]))

    success_rate = summary["n_success"] / max(summary["n_attempted"], 1) * 100.0

    lines = [
        f"=== {summary['run_id']} — {summary['now'].strftime('%Y-%m-%d %H:%M:%S UTC')} ===",
        (
            f"Progress:   {summary['n_attempted']:,} / {summary['total_notes']:,}  "
            f"({summary['progress_pct']:.2f}%)"
        ),
        f"Successes:  {summary['n_success']:,} ({success_rate:.2f}%)",
        f"Parse fail: {summary['n_parse_fail']:,} ({summary['parse_fail_rate']:.3f}%)",
        f"API error:  {summary['n_api_error']:,} ({summary['api_error_rate']:.3f}%)",
        f"Cost:       ${summary['spent']:.2f} spent / ${summary['projected_total']:.2f} projected",
        (
            f"Throughput: {summary['window_rps']:.2f} req/s "
            f"(last {summary['window_n']} notes by latency window)"
        ),
        f"Elapsed:    {_format_hms(float(summary['elapsed_seconds']))}",
        f"ETA:        {eta_txt} ({rem_txt} remaining)",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor production extraction progress.")
    parser.add_argument("--run-id", default="production_v1")
    parser.add_argument("--total-notes", type=int, default=TOTAL_NOTES_DEFAULT)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    while True:
        summary = _compute_summary(args.run_id, args.total_notes)
        rendered = _render(summary)
        if args.watch:
            os.system("clear")
        print(rendered)
        if not args.watch:
            break
        time.sleep(max(5, int(args.interval_seconds)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
