"""
Runs prompt dry-run extraction checks.

Reads: data/splits/refinement_150.csv, data/splits/holdout_150.csv, data/splits/smoke_200.csv, codex_outputs/04_prompt_dryrun.md.
Writes: data/splits/refinement_150.csv, data/splits/holdout_150.csv, data/splits/smoke_200.csv, codex_outputs/04_prompt_dryrun.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/04_prompt_dryrun.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import tiktoken

from src import config
from src.db.connection import get_engine
from src.db.queries import fetch_notes_by_hadm_ids
from src.llm.extractor import (
    build_messages,
    check_prompt_vocabulary_sync,
    count_prompt_tokens,
    load_prompt_template,
)

TOTAL_NOTES_FOR_PROJECTION = 331_793

logger = logging.getLogger(__name__)


def _resolve_encoding() -> tiktoken.Encoding:
    for model_name in (config.MODEL_ID, "gpt-5.4-nano"):
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            continue

    logger.info(
        "Using tiktoken encoding `o200k_base` as fallback for model `%s`.",
        config.MODEL_ID,
    )
    return tiktoken.get_encoding("o200k_base")


def _count_text_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text))


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"

    lines = []
    for row in rows:
        rendered = []
        for column in columns:
            value = _format_number(row.get(column, ""))
            value = value.replace("\n", " ").replace("|", "\\|")
            rendered.append(value)
        lines.append("| " + " | ".join(rendered) + " |")

    return "\n".join([header, divider, *lines])


def _load_split(split: str) -> pd.DataFrame:
    split_paths = {
        "refinement": Path("data/splits/refinement_150.csv"),
        "holdout": Path("data/splits/holdout_150.csv"),
        "smoke": Path("data/splits/smoke_200.csv"),
    }
    path = split_paths[split]
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")
    frame = pd.read_csv(path)
    if "hadm_id" not in frame.columns:
        raise ValueError(f"Split CSV missing hadm_id column: {path}")
    return frame


def _truncate_note_for_display(note_text: str) -> str:
    if len(note_text) <= 3000:
        return note_text
    return note_text[:3000] + "\n[... TRUNCATED FOR DISPLAY ...]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run prompt assembly and token accounting.")
    parser.add_argument("--split", choices=["refinement", "holdout", "smoke"], default="smoke")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--variant", default="a")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include-reasoning", dest="include_reasoning", action="store_true")
    group.add_argument("--no-reasoning", dest="include_reasoning", action="store_false")
    parser.set_defaults(include_reasoning=True)

    parser.add_argument("--output", default="codex_outputs/04_prompt_dryrun.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.split == "holdout":
        logger.warning(
            "Using holdout split for dry-run. This is allowed but unusual — holdout is for "
            "final validation only."
        )

    config.load_env()
    split_frame = _load_split(args.split)
    split_frame = split_frame.sort_values("hadm_id", kind="mergesort").reset_index(drop=True)

    n_samples = min(max(args.n_samples, 1), len(split_frame))
    sample = split_frame.head(n_samples).copy()
    hadm_ids = [int(value) for value in sample["hadm_id"].tolist()]

    engine = get_engine()
    notes_by_hadm = fetch_notes_by_hadm_ids(engine, hadm_ids)

    missing = [hadm_id for hadm_id in hadm_ids if hadm_id not in notes_by_hadm]
    if missing:
        raise RuntimeError(f"Missing discharge notes for hadm_ids: {missing}")

    prompt_template = load_prompt_template(args.variant)
    try:
        check_prompt_vocabulary_sync(prompt_template)
        vocab_sync_result = "PASS"
    except Exception:
        vocab_sync_result = "FAIL"
        raise

    encoding = _resolve_encoding()
    projected_output_tokens = 600 if args.include_reasoning else 400

    note_rows: list[dict[str, Any]] = []
    messages_by_hadm: dict[int, list[dict[str, str]]] = {}
    for hadm_id in hadm_ids:
        note_text = notes_by_hadm[hadm_id]
        messages = build_messages(
            note_text=note_text,
            variant=args.variant,
            include_reasoning=args.include_reasoning,
        )
        messages_by_hadm[hadm_id] = messages

        system_tokens = _count_text_tokens(messages[0]["content"], encoding)
        user_tokens = _count_text_tokens(messages[1]["content"], encoding)
        total_input_tokens = count_prompt_tokens(messages)

        input_cost = (total_input_tokens / 1_000_000) * config.INPUT_PRICE_PER_MILLION_USD
        output_cost = (projected_output_tokens / 1_000_000) * config.OUTPUT_PRICE_PER_MILLION_USD
        total_cost = input_cost + output_cost

        note_rows.append(
            {
                "hadm_id": hadm_id,
                "note_char_len": len(note_text),
                "system_tokens": system_tokens,
                "user_tokens": user_tokens,
                "total_input_tokens": total_input_tokens,
                "projected_output_tokens": projected_output_tokens,
                "per_note_cost_usd": total_cost,
            }
        )

    total_inputs = np.asarray([row["total_input_tokens"] for row in note_rows], dtype=np.int64)
    total_costs = np.asarray([row["per_note_cost_usd"] for row in note_rows], dtype=np.float64)

    aggregate_rows = [
        {"metric": "min_total_input_tokens", "value": int(total_inputs.min())},
        {"metric": "median_total_input_tokens", "value": float(np.median(total_inputs))},
        {"metric": "max_total_input_tokens", "value": int(total_inputs.max())},
        {"metric": "min_per_note_cost_usd", "value": float(total_costs.min())},
        {"metric": "median_per_note_cost_usd", "value": float(np.median(total_costs))},
        {"metric": "max_per_note_cost_usd", "value": float(total_costs.max())},
    ]

    median_input_tokens = float(np.median(total_inputs))
    full_run_input_cost = (
        (median_input_tokens * TOTAL_NOTES_FOR_PROJECTION) / 1_000_000
    ) * config.INPUT_PRICE_PER_MILLION_USD
    full_run_output_cost = (
        (projected_output_tokens * TOTAL_NOTES_FOR_PROJECTION) / 1_000_000
    ) * config.OUTPUT_PRICE_PER_MILLION_USD
    full_run_total_cost = full_run_input_cost + full_run_output_cost
    cap_status = "within" if full_run_total_cost <= 500.0 else "above"

    first_hadm_id = hadm_ids[0]
    first_note = notes_by_hadm[first_hadm_id]
    display_messages = build_messages(
        note_text=_truncate_note_for_display(first_note),
        variant=args.variant,
        include_reasoning=args.include_reasoning,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# Prompt Dry-Run",
                "",
                "## Run metadata",
                _markdown_table(
                    [
                        {"field": "timestamp_utc", "value": datetime.now(tz=UTC).isoformat()},
                        {"field": "variant", "value": args.variant},
                        {"field": "include_reasoning", "value": args.include_reasoning},
                        {"field": "n_samples", "value": n_samples},
                        {"field": "split", "value": args.split},
                    ],
                    ["field", "value"],
                ),
                "",
                "## Vocabulary integrity check",
                f"Result: {vocab_sync_result}",
                "",
                "## Per-note token breakdown",
                _markdown_table(
                    note_rows,
                    [
                        "hadm_id",
                        "note_char_len",
                        "system_tokens",
                        "user_tokens",
                        "total_input_tokens",
                        "projected_output_tokens",
                        "per_note_cost_usd",
                    ],
                ),
                "",
                "## Aggregate statistics",
                _markdown_table(aggregate_rows, ["metric", "value"]),
                "",
                "## Prompt render — first note",
                f"hadm_id: {first_hadm_id}",
                "```json",
                json.dumps(display_messages, indent=2),
                "```",
                "",
                "## Projected full-run cost",
                _markdown_table(
                    [
                        {"metric": "median_input_tokens_sample", "value": median_input_tokens},
                        {
                            "metric": "projected_output_tokens_per_note",
                            "value": projected_output_tokens,
                        },
                        {"metric": "n_notes_projection", "value": TOTAL_NOTES_FOR_PROJECTION},
                        {"metric": "projected_input_cost_usd", "value": full_run_input_cost},
                        {"metric": "projected_output_cost_usd", "value": full_run_output_cost},
                        {"metric": "projected_total_cost_usd", "value": full_run_total_cost},
                        {"metric": "production_cap_usd", "value": 500.0},
                    ],
                    ["metric", "value"],
                ),
                "",
                (
                    f"Projected full-run total (${full_run_total_cost:,.2f}) is {cap_status} "
                    "the $500 production cap."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote prompt dry-run report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
