"""
Computes diagnostics for the methodology 5k/audit run.

Reads: codex_outputs/26_methodology_5k_snorkel_report.md, codex_outputs/26_pipeline_diagnostics_5k.md.
Writes: codex_outputs/26_methodology_5k_snorkel_report.md, codex_outputs/26_pipeline_diagnostics_5k.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/26_pipeline_diagnostics_5k.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src import config

PER_VARIANT_1K_BASELINE_COST: dict[str, float] = {
    "a": 0.001901,
    "b": 0.002001,
    "c": 0.002060,
}


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


def _load_run_metadata(run_id: str) -> dict[str, Any]:
    path = config.RAW_RESPONSES_DIR / run_id / "run_metadata.json"
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _cost_per_note(rows: list[dict[str, Any]]) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    input_tokens = float(sum(float(row.get("input_tokens", 0) or 0) for row in rows))
    output_tokens = float(sum(float(row.get("output_tokens", 0) or 0) for row in rows))
    cost = (
        (input_tokens / 1_000_000.0) * config.INPUT_PRICE_PER_MILLION_USD
        + (output_tokens / 1_000_000.0) * config.OUTPUT_PRICE_PER_MILLION_USD
    )
    return cost / float(n)


def cost_within_5pct(variant: str, observed_cost_per_note: float) -> tuple[bool, str]:
    key = variant.lower()
    baseline = PER_VARIANT_1K_BASELINE_COST[key]
    delta_pct = ((observed_cost_per_note - baseline) / baseline) * 100.0
    passed = abs(delta_pct) <= 5.0
    return passed, f"{delta_pct:+.2f}% vs variant {key.upper()} 1k baseline"


def evaluate_snorkel_gate(fit_status_counts: dict[str, int]) -> tuple[bool, int, int]:
    no_votes = int(fit_status_counts.get("no_votes", 0))
    single_lf_only = int(fit_status_counts.get("single_lf_only", 0))
    return (no_votes == 0), no_votes, single_lf_only


def overall_status_from_criteria(criteria_rows: list[dict[str, str]]) -> str:
    return "PASS" if all(row.get("status") == "PASS" for row in criteria_rows) else "FAIL"


def load_snorkel_fit_status_counts(report_path: Path) -> dict[str, int]:
    if not report_path.exists():
        return {}
    lines = report_path.read_text(encoding="utf-8").splitlines()
    counts: dict[str, int] = {}
    in_section = False
    for line in lines:
        if line.strip() == "## Fit-status distribution":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section:
            continue
        # Expected markdown row: | success | 20 |
        match = re.match(r"^\|\s*([A-Za-z0-9_/-]+)\s*\|\s*([0-9]+)\s*\|$", line.strip())
        if match and match.group(1) != "fit_status":
            counts[match.group(1)] = int(match.group(2))
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline robustness diagnostics for Phase 6 (5k)."
    )
    parser.add_argument("--run-id-a", default="methodology_5k_a")
    parser.add_argument("--run-id-b", default="methodology_5k_audit_b")
    parser.add_argument("--run-id-c", default="methodology_5k_audit_c")
    parser.add_argument("--baseline-run-id", default="methodology_1k_a")
    parser.add_argument("--projected-rps", type=float, default=2.5)
    parser.add_argument(
        "--snorkel-report",
        default="codex_outputs/26_methodology_5k_snorkel_report.md",
    )
    parser.add_argument("--output", default="codex_outputs/26_pipeline_diagnostics_5k.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    run_ids = [("A", args.run_id_a), ("B", args.run_id_b), ("C", args.run_id_c)]
    baseline_rows = _load_rows(args.baseline_run_id)
    baseline_cost = _cost_per_note(baseline_rows)

    criteria_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    fail_rows: list[dict[str, Any]] = []

    overall_pass = True

    for label, run_id in run_ids:
        rows = _load_rows(run_id)
        meta = _load_run_metadata(run_id)

        n_attempted = len(rows)
        n_parsed = int(sum(1 for row in rows if bool(row.get("parse_ok", False))))
        n_failed = n_attempted - n_parsed
        n_api_error = int(
            sum(
                1
                for row in rows
                if (not bool(row.get("parse_ok", False)))
                and str(row.get("parse_error", "")).startswith("api_error:")
            )
        )

        parse_failure_rate = (n_failed / n_attempted) if n_attempted else 0.0
        api_error_rate = (n_api_error / n_attempted) if n_attempted else 0.0

        inputs = [float(row.get("input_tokens", 0) or 0) for row in rows]
        outputs = [float(row.get("output_tokens", 0) or 0) for row in rows]
        latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in rows]

        median_input = _median(inputs)
        median_output = _median(outputs)
        median_latency = _median(latencies)
        p99_latency = _percentile(latencies, 99)

        run_input_tokens = float(sum(inputs))
        run_output_tokens = float(sum(outputs))
        run_cost = (
            (run_input_tokens / 1_000_000.0) * config.INPUT_PRICE_PER_MILLION_USD
            + (run_output_tokens / 1_000_000.0) * config.OUTPUT_PRICE_PER_MILLION_USD
        )
        per_note_cost = (run_cost / n_attempted) if n_attempted else 0.0
        cost_pass, cost_msg = cost_within_5pct(label, per_note_cost)
        cost_delta_pct = float(cost_msg.split("%", maxsplit=1)[0])

        elapsed_seconds = float(meta.get("elapsed_seconds", 0.0) or 0.0)
        throughput_rps = (n_attempted / elapsed_seconds) if elapsed_seconds > 0 else 0.0

        parse_pass = parse_failure_rate < 0.005
        api_pass = api_error_rate < 0.005
        latency_pass = p99_latency < (3.0 * median_latency) if median_latency > 0 else True

        criteria_rows.extend(
            [
                {
                    "variant": label,
                    "criterion": "parse_failure_rate_lt_0_5pct",
                    "value": f"{parse_failure_rate*100:.3f}%",
                    "status": "PASS" if parse_pass else "FAIL",
                },
                {
                    "variant": label,
                    "criterion": "api_error_rate_lt_0_5pct",
                    "value": f"{api_error_rate*100:.3f}%",
                    "status": "PASS" if api_pass else "FAIL",
                },
                {
                    "variant": label,
                    "criterion": f"per_note_cost_within_5pct_vs_1k_{label}",
                    "value": cost_msg,
                    "status": "PASS" if cost_pass else "FAIL",
                },
                {
                    "variant": label,
                    "criterion": "latency_p99_lt_3x_median",
                    "value": f"median={median_latency:.3f}s,p99={p99_latency:.3f}s",
                    "status": "PASS" if latency_pass else "FAIL",
                },
            ]
        )

        throughput_pass = True
        if label == "A":
            low = float(args.projected_rps) * 0.8
            high = float(args.projected_rps) * 1.2
            throughput_pass = low <= throughput_rps <= high
            criteria_rows.append(
                {
                    "variant": label,
                    "criterion": "throughput_within_20pct_of_2_5_rps",
                    "value": f"{throughput_rps:.3f} rps (target {args.projected_rps:.3f})",
                    "status": "PASS" if throughput_pass else "FAIL",
                }
            )

        run_pass = parse_pass and api_pass and cost_pass and latency_pass and throughput_pass
        overall_pass = overall_pass and run_pass

        summary_rows.append(
            {
                "variant": label,
                "run_id": run_id,
                "n_attempted": n_attempted,
                "n_parsed": n_parsed,
                "parse_failure_rate_pct": f"{parse_failure_rate*100:.3f}",
                "api_error_rate_pct": f"{api_error_rate*100:.3f}",
                "median_input_tokens": f"{median_input:.1f}",
                "median_output_tokens": f"{median_output:.1f}",
                "per_note_cost_usd": f"{per_note_cost:.6f}",
                "cost_delta_vs_own_1k_pct": f"{cost_delta_pct:.2f}",
                "elapsed_seconds": f"{elapsed_seconds:.3f}",
                "throughput_rps": f"{throughput_rps:.3f}",
                "median_latency_s": f"{median_latency:.3f}",
                "p99_latency_s": f"{p99_latency:.3f}",
                "run_cost_usd": f"{run_cost:.6f}",
            }
        )

        for row in rows:
            if bool(row.get("parse_ok", False)):
                continue
            fail_rows.append(
                {
                    "variant": label,
                    "run_id": run_id,
                    "hadm_id": int(row.get("hadm_id", 0) or 0),
                    "parse_error": str(row.get("parse_error", "")),
                }
            )

    fail_rows = fail_rows[:60]

    fit_status_counts = load_snorkel_fit_status_counts(Path(args.snorkel_report))
    snorkel_pass, no_votes, single_lf_only = evaluate_snorkel_gate(fit_status_counts)
    overall_pass = overall_pass and snorkel_pass
    criteria_rows.append(
        {
            "variant": "SNORKEL",
            "criterion": "no_votes_eq_0",
            "value": f"no_votes={no_votes}; single_lf_only={single_lf_only} (reported, not gated)",
            "status": "PASS" if snorkel_pass else "FAIL",
        }
    )

    overall_status = overall_status_from_criteria(
        [
            {"status": row["status"]}
            for row in criteria_rows
            if isinstance(row.get("status"), str)
        ]
    )

    lines = [
        "# Pipeline Diagnostics (Methodology 5k)",
        "",
        f"## Overall status: {overall_status}",
        "",
        "## Run metadata",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_id_a": args.run_id_a,
                    "run_id_b": args.run_id_b,
                    "run_id_c": args.run_id_c,
                    "baseline_run_id": args.baseline_run_id,
                    "baseline_per_note_cost_usd_a": f"{baseline_cost:.6f}",
                    "baseline_per_note_cost_usd_b": f"{PER_VARIANT_1K_BASELINE_COST['b']:.6f}",
                    "baseline_per_note_cost_usd_c": f"{PER_VARIANT_1K_BASELINE_COST['c']:.6f}",
                    "throughput_target_rps": f"{args.projected_rps:.3f}",
                }
            ],
            [
                "timestamp_utc",
                "run_id_a",
                "run_id_b",
                "run_id_c",
                "baseline_run_id",
                "baseline_per_note_cost_usd_a",
                "baseline_per_note_cost_usd_b",
                "baseline_per_note_cost_usd_c",
                "throughput_target_rps",
            ],
        ),
        "",
        "## PASS/FAIL criteria",
        _markdown_table(criteria_rows, ["variant", "criterion", "value", "status"]),
        "",
        "## Per-run diagnostics",
        _markdown_table(
            summary_rows,
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
                "cost_delta_vs_own_1k_pct",
                "elapsed_seconds",
                "throughput_rps",
                "median_latency_s",
                "p99_latency_s",
                "run_cost_usd",
            ],
        ),
        "",
        "## Snorkel fit-status counts",
        _markdown_table(
            [
                {"fit_status": key, "n_targets": value}
                for key, value in sorted(fit_status_counts.items(), key=lambda kv: kv[0])
            ],
            ["fit_status", "n_targets"],
        ),
        "",
        (
            "Gating rule: `no_votes == 0` is production-blocking; `single_lf_only` is "
            "reported but not used as a fail condition in single-variant production."
        ),
        "",
        "## Parse/API failure samples",
        _markdown_table(fail_rows, ["variant", "run_id", "hadm_id", "parse_error"]),
        "",
    ]

    if not overall_pass:
        lines.extend(
            [
                "## Next steps",
                (
                    "- Inspect run metadata + results for bursts of "
                    "`api_error: RateLimitError` despite 8-concurrency."
                ),
                (
                    "- Compare failed-note latencies and token sizes vs median "
                    "for overload pattern detection."
                ),
                "- Validate OpenAI quota headroom and rerun a small sentinel batch if needed.",
                "",
            ]
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote pipeline diagnostics report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
