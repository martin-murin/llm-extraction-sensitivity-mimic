"""
Bootstraps regex and embedding seed patterns from local evidence.

Reads: data/raw_responses/coverage_v2/results.jsonl, src/labeling_functions/patterns, configs/optimization.yaml.
Writes: data/raw_responses/coverage_v2/results.jsonl.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/08_bootstrap_patterns.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from datetime import UTC, datetime
from typing import Any

import yaml

from src import config
from src.labeling_functions.pattern_bootstrap import (
    EXTENSION_PILOT_FIELDS,
    REGEX_PILOT_FIELDS,
    derive_embedding_seed_phrases,
    derive_regex_patterns,
    extract_anchor_phrases,
    load_coverage_v2_results,
    write_pattern_yaml,
)
from src.llm.client import LLMClient

logger = logging.getLogger("scripts.08_bootstrap_patterns")

BOOTSTRAP_BUDGET_USD = 5.0


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = []
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *lines])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap regex patterns from coverage_v2 reasoning."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/raw_responses/coverage_v2/results.jsonl"),
        help="Path to coverage_v2 results.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/labeling_functions/patterns"),
        help="Directory to write per-field pattern YAML files",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="bootstrap_v1",
        help="Run id used by CostTracker",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/optimization.yaml"),
        help="Optimization config path",
    )
    parser.add_argument(
        "--field-set",
        choices=["pilot", "extension"],
        default="pilot",
        help="Which field set to bootstrap.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    config.load_env()
    with args.config.open("r", encoding="utf-8") as handle:
        run_config = yaml.safe_load(handle) or {}

    semaphore_limit = int(run_config.get("max_concurrent_requests", config.MAX_CONCURRENT_REQUESTS))
    llm_client = LLMClient(
        semaphore_limit=semaphore_limit,
        run_id=args.run_id,
        max_budget_usd=BOOTSTRAP_BUDGET_USD,
    )

    results = load_coverage_v2_results(args.results)
    logger.info("Loaded %s parsed coverage_v2 results from %s", len(results), args.results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_fields = REGEX_PILOT_FIELDS if args.field_set == "pilot" else EXTENSION_PILOT_FIELDS
    logger.info("Using field set '%s' with %s fields.", args.field_set, len(selected_fields))

    summary_rows: list[dict[str, Any]] = []
    details_by_field: dict[str, dict[str, Any]] = {}

    for field_name, target_value in selected_fields:
        source_rows = [
            row
            for row in results
            if isinstance(row.get("features"), dict)
            and row["features"].get(field_name) == target_value
        ]
        logger.info("%s=%s source notes: %s", field_name, target_value, len(source_rows))

        anchors_by_hadm = extract_anchor_phrases(
            results=results,
            field_name=field_name,
            target_value=target_value,
            llm_client=llm_client,
        )
        flattened_anchors = [
            phrase
            for phrases in anchors_by_hadm.values()
            for phrase in phrases
            if isinstance(phrase, str) and phrase.strip()
        ]

        regex_patterns = derive_regex_patterns(flattened_anchors)
        seed_phrases = derive_embedding_seed_phrases(flattened_anchors)

        yaml_path = write_pattern_yaml(
            field_name=field_name,
            target_value=target_value,
            regex_patterns=regex_patterns,
            seed_phrases=seed_phrases,
            source_run_id="coverage_v2",
            n_source_notes=len(source_rows),
            output_dir=args.output_dir,
        )

        logger.info(
            "%s -> source_notes=%s anchors=%s regex=%s seeds=%s (%s)",
            field_name,
            len(source_rows),
            len(flattened_anchors),
            len(regex_patterns),
            len(seed_phrases),
            yaml_path,
        )

        summary_rows.append(
            {
                "field": field_name,
                "target_value": target_value,
                "n_source_notes": len(source_rows),
                "n_anchor_phrases_extracted": len(flattened_anchors),
                "n_regex_patterns": len(regex_patterns),
                "n_seed_phrases": len(seed_phrases),
            }
        )
        details_by_field[field_name] = {
            "target_value": target_value,
            "regex_patterns": regex_patterns,
            "seed_phrases": seed_phrases,
            "sample_anchors": flattened_anchors[:10],
            "yaml_path": yaml_path,
        }

    cost_summary = llm_client.cost_tracker.summary()

    report_lines: list[str] = []
    report_lines.append("# 08 Pattern Bootstrap Report")
    report_lines.append("")
    report_lines.append("## Run metadata")
    report_lines.append("")
    report_lines.append(
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "run_id": args.run_id,
                    "field_set": args.field_set,
                    "source_results": str(args.results),
                    "n_parsed_notes_available": len(results),
                    "total_input_tokens": int(cost_summary["total_input_tokens"]),
                    "total_output_tokens": int(cost_summary["total_output_tokens"]),
                    "total_cost_usd": f"{float(cost_summary['total_cost_usd']):.6f}",
                    "budget_cap_usd": f"{BOOTSTRAP_BUDGET_USD:.2f}",
                }
            ],
            [
                "timestamp_utc",
                "run_id",
                "field_set",
                "source_results",
                "n_parsed_notes_available",
                "total_input_tokens",
                "total_output_tokens",
                "total_cost_usd",
                "budget_cap_usd",
            ],
        )
    )
    report_lines.append("")

    report_lines.append("## Per-field summary")
    report_lines.append("")
    report_lines.append(
        _markdown_table(
            summary_rows,
            [
                "field",
                "target_value",
                "n_source_notes",
                "n_anchor_phrases_extracted",
                "n_regex_patterns",
                "n_seed_phrases",
            ],
        )
    )
    report_lines.append("")

    report_lines.append("## Generated patterns and seed phrases")
    report_lines.append("")
    for field_name, _ in selected_fields:
        details = details_by_field[field_name]
        report_lines.append(f"### {field_name}")
        report_lines.append("")
        report_lines.append(f"- `target_value`: `{details['target_value']}`")
        report_lines.append(f"- `yaml_path`: `{details['yaml_path']}`")
        report_lines.append("")
        report_lines.append("Regex patterns:")
        patterns = details["regex_patterns"]
        if patterns:
            for pattern in patterns:
                report_lines.append(f"- `{pattern}`")
        else:
            report_lines.append("- _None generated_")
        report_lines.append("")
        report_lines.append("Embedding seed phrases:")
        seeds = details["seed_phrases"]
        if seeds:
            for seed in seeds:
                report_lines.append(f"- {seed}")
        else:
            report_lines.append("- _None generated_")
        report_lines.append("")

    report_lines.append("## Sample raw anchor phrases")
    report_lines.append("")
    for field_name, _ in selected_fields:
        details = details_by_field[field_name]
        report_lines.append(f"### {field_name}")
        sample_anchors = details["sample_anchors"]
        if sample_anchors:
            for anchor in sample_anchors:
                report_lines.append(f"- {anchor}")
        else:
            report_lines.append("- _No anchor phrases extracted_")
        report_lines.append("")

    report_path = config.CODEX_OUTPUTS_DIR / "08_bootstrap_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    logger.info("Wrote bootstrap report to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
