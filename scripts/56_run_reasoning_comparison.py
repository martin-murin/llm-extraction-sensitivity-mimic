"""
Runs reasoning-instruction comparison extractions.

Reads: configs/optimization.yaml, codex_outputs/56_reasoning_run.md.
Writes: codex_outputs/56_reasoning_run.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/56_run_reasoning_comparison.py` unless the script's argparse help says otherwise.
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

EXPECTED_MODEL_ID = "gpt-5.4-nano-2026-03-17"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body: list[str] = []
    for row in rows:
        vals = [str(row.get(col, "")).replace("|", "\\|").replace("\n", " ") for col in columns]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider, *body])


def _load_run_summary(results_path: Path) -> dict[str, Any]:
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.jsonl: {results_path}")

    attempted = 0
    parse_ok = 0
    parse_fail = 0
    api_error = 0
    input_tokens = 0
    output_tokens = 0

    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            attempted += 1
            payload = json.loads(line)
            input_tokens += int(payload.get("input_tokens", 0) or 0)
            output_tokens += int(payload.get("output_tokens", 0) or 0)
            if bool(payload.get("parse_ok", False)):
                parse_ok += 1
            else:
                parse_fail += 1
                if payload.get("parse_error"):
                    api_error += 1

    input_cost = (input_tokens / 1_000_000.0) * config.INPUT_PRICE_PER_MILLION_USD
    output_cost = (output_tokens / 1_000_000.0) * config.OUTPUT_PRICE_PER_MILLION_USD
    total_cost = input_cost + output_cost

    return {
        "attempted": attempted,
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
        "api_error_like": api_error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run variant A on methodology_1k with reasoning ON for Phase 9 comparison."
    )
    parser.add_argument("--run-id", default="reasoning_on_methodology_1k_a")
    parser.add_argument("--split", default="methodology_1k")
    parser.add_argument("--variant", default="a")
    parser.add_argument("--budget-cap-usd", type=float, default=10.0)
    parser.add_argument("--config", default="configs/optimization.yaml")
    parser.add_argument("--output", default="codex_outputs/56_reasoning_run.md")
    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="If set, do not fail when config.MODEL_ID differs from expected nano model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = config.REPO_ROOT

    active_model = config.MODEL_ID
    if active_model != EXPECTED_MODEL_ID and not args.allow_model_mismatch:
        raise RuntimeError(
            "Model mismatch for Phase 9 reasoning run: "
            f"expected '{EXPECTED_MODEL_ID}', got '{active_model}'."
        )

    cmd = [
        sys.executable,
        "scripts/05_run_smoke_coverage.py",
        "--run-id",
        args.run_id,
        "--split",
        args.split,
        "--variant",
        args.variant,
        "--include-reasoning",
        "--budget-cap-usd",
        f"{args.budget_cap_usd:.6f}",
        "--config",
        args.config,
    ]

    started_utc = datetime.now(tz=UTC).isoformat()
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    ended_utc = datetime.now(tz=UTC).isoformat()

    results_path = config.RAW_RESPONSES_DIR / args.run_id / "results.jsonl"
    summary: dict[str, Any]
    try:
        summary = _load_run_summary(results_path)
    except FileNotFoundError:
        summary = {
            "attempted": 0,
            "parse_ok": 0,
            "parse_fail": 0,
            "api_error_like": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        }

    report_rows = [
        {"field": "run_id", "value": args.run_id},
        {"field": "split", "value": args.split},
        {"field": "variant", "value": args.variant},
        {"field": "include_reasoning", "value": True},
        {"field": "budget_cap_usd", "value": f"{args.budget_cap_usd:.2f}"},
        {"field": "expected_model_id", "value": EXPECTED_MODEL_ID},
        {"field": "active_model_id", "value": active_model},
        {"field": "started_utc", "value": started_utc},
        {"field": "ended_utc", "value": ended_utc},
        {"field": "runner_exit_code", "value": completed.returncode},
        {"field": "attempted", "value": summary["attempted"]},
        {"field": "parse_ok", "value": summary["parse_ok"]},
        {"field": "parse_fail", "value": summary["parse_fail"]},
        {"field": "api_error_like", "value": summary["api_error_like"]},
        {"field": "input_tokens", "value": summary["input_tokens"]},
        {"field": "output_tokens", "value": summary["output_tokens"]},
        {"field": "input_cost_usd", "value": f"{summary['input_cost_usd']:.6f}"},
        {"field": "output_cost_usd", "value": f"{summary['output_cost_usd']:.6f}"},
        {"field": "total_cost_usd", "value": f"{summary['total_cost_usd']:.6f}"},
    ]

    lines = [
        "# Phase 9 Reasoning-ON Run",
        "",
        _markdown_table(report_rows, ["field", "value"]),
        "",
        "## Command",
        "```bash",
        " ".join(cmd),
        "```",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
