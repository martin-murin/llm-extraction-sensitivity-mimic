"""
Computes diagnostics for methodology pipeline outputs.

Reads: codex_outputs/22_pipeline_diagnostics.md.
Writes: codex_outputs/22_pipeline_diagnostics.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/20_pipeline_diagnostics.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src import config
from src.llm.extractor import extract_content_from_raw_response


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


def _load_rows(run_id: str) -> list[dict[str, Any]]:
    path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _cost_per_note(median_input: float, median_output: float) -> float:
    return (
        (median_input / 1_000_000.0) * config.INPUT_PRICE_PER_MILLION_USD
        + (median_output / 1_000_000.0) * config.OUTPUT_PRICE_PER_MILLION_USD
    )


def _parse_failure_excerpt(run_id: str, hadm_id: int) -> str:
    path = config.RAW_RESPONSES_DIR / run_id / f"{hadm_id}.json"
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_response = payload.get("raw_response")
    if not isinstance(raw_response, dict):
        return ""
    content = extract_content_from_raw_response(raw_response)
    compact = " ".join(content.split())
    return compact[:100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline robustness diagnostics for methodology 1k."
    )
    parser.add_argument(
        "--run-ids",
        nargs=3,
        default=["methodology_1k_a", "methodology_1k_b", "methodology_1k_c"],
        metavar=("RUN_A", "RUN_B", "RUN_C"),
    )
    parser.add_argument("--baseline-run-id", default="refinement_v1_a")
    parser.add_argument("--output", default="codex_outputs/22_pipeline_diagnostics.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    runs = [("A", args.run_ids[0]), ("B", args.run_ids[1]), ("C", args.run_ids[2])]
    baseline_rows = _load_rows(args.baseline_run_id)
    baseline_input = _median([float(row.get("input_tokens", 0) or 0) for row in baseline_rows])
    baseline_output = _median([float(row.get("output_tokens", 0) or 0) for row in baseline_rows])
    baseline_cost = _cost_per_note(baseline_input, baseline_output)

    criterion_rows: list[dict[str, Any]] = []
    per_variant_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    total_cost = 0.0
    all_pass = True

    for label, run_id in runs:
        rows = _load_rows(run_id)
        n_attempted = len(rows)
        n_parsed = int(sum(1 for row in rows if bool(row.get("parse_ok", False))))
        n_failed = n_attempted - n_parsed
        parse_failure_rate = (n_failed / n_attempted) if n_attempted else 0.0

        api_error_count = 0
        for row in rows:
            parse_error = str(row.get("parse_error", "") or "")
            if parse_error.startswith("api_error:"):
                api_error_count += 1
        api_error_rate = (api_error_count / n_attempted) if n_attempted else 0.0

        inputs = [float(row.get("input_tokens", 0) or 0) for row in rows]
        outputs = [float(row.get("output_tokens", 0) or 0) for row in rows]
        latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in rows]

        median_input = _median(inputs)
        median_output = _median(outputs)
        median_latency = _median(latencies)
        p95_latency = _percentile(latencies, 95)

        run_input_tokens = float(sum(inputs))
        run_output_tokens = float(sum(outputs))
        run_cost = (
            (run_input_tokens / 1_000_000.0) * config.INPUT_PRICE_PER_MILLION_USD
            + (run_output_tokens / 1_000_000.0) * config.OUTPUT_PRICE_PER_MILLION_USD
        )
        total_cost += run_cost
        per_note_cost = (run_cost / n_attempted) if n_attempted else 0.0
        cost_delta_pct = (
            ((per_note_cost - baseline_cost) / baseline_cost) * 100.0
            if baseline_cost > 0
            else 0.0
        )

        parse_pass = parse_failure_rate < 0.02
        api_pass = api_error_rate < 0.01
        cost_pass = abs(cost_delta_pct) <= 15.0
        latency_pass = p95_latency <= (2.0 * median_latency) if median_latency > 0 else True
        all_pass = all_pass and parse_pass and api_pass and cost_pass and latency_pass

        criterion_rows.extend(
            [
                {
                    "variant": label,
                    "criterion": "parse_failure_rate_lt_2pct",
                    "value": f"{parse_failure_rate * 100.0:.2f}%",
                    "status": "PASS" if parse_pass else "FAIL",
                },
                {
                    "variant": label,
                    "criterion": "api_error_rate_lt_1pct",
                    "value": f"{api_error_rate * 100.0:.2f}%",
                    "status": "PASS" if api_pass else "FAIL",
                },
                {
                    "variant": label,
                    "criterion": "cost_delta_within_15pct_vs_refinementA",
                    "value": f"{cost_delta_pct:.2f}%",
                    "status": "PASS" if cost_pass else "FAIL",
                },
                {
                    "variant": label,
                    "criterion": "latency_p95_le_2x_median",
                    "value": f"median={median_latency:.3f}s,p95={p95_latency:.3f}s",
                    "status": "PASS" if latency_pass else "FAIL",
                },
            ]
        )

        per_variant_rows.append(
            {
                "variant": label,
                "run_id": run_id,
                "n_attempted": n_attempted,
                "n_parsed": n_parsed,
                "parse_failure_rate_pct": f"{parse_failure_rate * 100.0:.2f}",
                "api_error_rate_pct": f"{api_error_rate * 100.0:.2f}",
                "median_input_tokens": f"{median_input:.1f}",
                "median_output_tokens": f"{median_output:.1f}",
                "per_note_cost_usd": f"{per_note_cost:.6f}",
                "cost_delta_vs_refinementA_pct": f"{cost_delta_pct:.2f}",
                "median_latency_s": f"{median_latency:.3f}",
                "p95_latency_s": f"{p95_latency:.3f}",
                "run_cost_usd": f"{run_cost:.6f}",
            }
        )

        variant_failures = [row for row in rows if not bool(row.get("parse_ok", False))]
        for row in variant_failures:
            hadm_id = int(row.get("hadm_id", 0) or 0)
            parse_error = str(row.get("parse_error", "") or "")
            error_type = parse_error.split(":", maxsplit=1)[0] if parse_error else "unknown"
            failure_rows.append(
                {
                    "variant": label,
                    "run_id": run_id,
                    "hadm_id": hadm_id,
                    "error_type": error_type,
                    "parse_error": parse_error,
                    "raw_excerpt_100": _parse_failure_excerpt(run_id, hadm_id),
                }
            )

    failure_rows = failure_rows[:50]

    lines = [
        "# Pipeline Diagnostics (Methodology 1k)",
        "",
        f"## Overall status: {'PASS' if all_pass else 'FAIL'}",
        "",
        "## Run metadata",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_ids": ", ".join(args.run_ids),
                    "baseline_run_id": args.baseline_run_id,
                    "baseline_variantA_refinement_per_note_cost_usd": f"{baseline_cost:.6f}",
                    "total_methodology_1k_cost_usd": f"{total_cost:.6f}",
                }
            ],
            [
                "timestamp_utc",
                "run_ids",
                "baseline_run_id",
                "baseline_variantA_refinement_per_note_cost_usd",
                "total_methodology_1k_cost_usd",
            ],
        ),
        "",
        "## PASS/FAIL criteria",
        _markdown_table(criterion_rows, ["variant", "criterion", "value", "status"]),
        "",
        "## Per-variant diagnostics",
        _markdown_table(
            per_variant_rows,
            [
                "variant",
                "run_id",
                "n_attempted",
                "n_parsed",
                "parse_failure_rate_pct",
                "api_error_rate_pct",
                "median_input_tokens",
                "median_output_tokens",
                "per_note_cost_usd",
                "cost_delta_vs_refinementA_pct",
                "median_latency_s",
                "p95_latency_s",
                "run_cost_usd",
            ],
        ),
        "",
        "## Parse failure mode summary",
        _markdown_table(
            failure_rows,
            [
                "variant",
                "run_id",
                "hadm_id",
                "error_type",
                "parse_error",
                "raw_excerpt_100",
            ],
        ),
        "",
        (
            "If any criterion FAILed, investigate prompt assembly, API retries, and "
            "schema-parse drift before scaling."
        ),
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote pipeline diagnostics report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
