from __future__ import annotations

# Release documentation:
# Exports production extraction results to local parquet feature tables.
#
# Reads: data/production/parquet/production_v1_features.parquet, data/production/parquet/raw_responses.parquet, data/raw_responses.
# Writes: data/production/parquet/production_v1_features.parquet, data/production/parquet/raw_responses.parquet, data/raw_responses.
# Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
# Usage: `python scripts/41_export_parquet.py` unless the script's argparse help says otherwise.

import argparse
from collections.abc import Iterator
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema.fields import LLMNoteFeatures

DEFAULT_RUN_ID = "production_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export production JSONL outputs to parquet datasets."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--features-output",
        default="data/production/parquet/production_v1_features.parquet",
    )
    parser.add_argument("--raw-output", default="data/production/parquet/raw_responses.parquet")
    parser.add_argument("--chunk-size", type=int, default=5000)
    return parser.parse_args()


def _empty_feature_row(hadm_id: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "hadm_id": hadm_id,
        "parse_ok": False,
        "parse_error": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_seconds": 0.0,
    }
    for name in LLMNoteFeatures.model_fields:
        row[name] = None
    row["admission_reason_tags"] = []
    return row


def _load_features_frame(results_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            hadm_id = int(payload.get("hadm_id", 0) or 0)
            row = _empty_feature_row(hadm_id)
            row["parse_ok"] = bool(payload.get("parse_ok", False))
            row["parse_error"] = payload.get("parse_error")
            row["input_tokens"] = int(payload.get("input_tokens", 0) or 0)
            row["output_tokens"] = int(payload.get("output_tokens", 0) or 0)
            row["latency_seconds"] = float(payload.get("latency_seconds", 0.0) or 0.0)

            features_json = payload.get("features_json")
            if isinstance(features_json, dict):
                for name in LLMNoteFeatures.model_fields:
                    if name in features_json:
                        row[name] = features_json[name]
                if row["admission_reason_tags"] is None:
                    row["admission_reason_tags"] = []
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["hadm_id"], keep="first").sort_values("hadm_id")
    frame = frame.reset_index(drop=True)
    return frame


def _iter_raw_rows(run_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(run_dir.glob("*.json")):
        if path.name == "run_metadata.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        hadm_id = int(payload.get("hadm_id", 0) or 0)
        raw_response = payload.get("raw_response")
        raw_json_str = (
            json.dumps(raw_response, ensure_ascii=True)
            if isinstance(raw_response, dict)
            else None
        )
        yield {
            "hadm_id": hadm_id,
            "parse_ok": bool(payload.get("parse_ok", False)),
            "parse_error": str(payload.get("parse_error", "") or ""),
            "input_tokens": int(payload.get("input_tokens", 0) or 0),
            "output_tokens": int(payload.get("output_tokens", 0) or 0),
            "latency_seconds": float(payload.get("latency_seconds", 0.0) or 0.0),
            "processed_at_utc": str(payload.get("processed_at_utc", "") or ""),
            "run_id": str(payload.get("run_id", "") or ""),
            "variant": str(payload.get("variant", "") or ""),
            "include_reasoning": bool(payload.get("include_reasoning", False)),
            "raw_response_json": raw_json_str or "",
            "features_json": json.dumps(payload.get("features_json"), ensure_ascii=True)
            if isinstance(payload.get("features_json"), dict)
            else "",
        }


def _write_raw_parquet(run_dir: Path, output_path: Path, chunk_size: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []
    count = 0
    for row in _iter_raw_rows(run_dir):
        buffer.append(row)
        if len(buffer) < chunk_size:
            continue
        table = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), table.schema, compression="snappy")
        writer.write_table(table)
        count += len(buffer)
        buffer = []

    if buffer:
        table = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), table.schema, compression="snappy")
        writer.write_table(table)
        count += len(buffer)

    if writer is not None:
        writer.close()
    return count


def main() -> int:
    args = parse_args()
    run_dir = Path("data/raw_responses") / args.run_id
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    features_output = Path(args.features_output)
    raw_output = Path(args.raw_output)

    features = _load_features_frame(results_path)
    features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_output, index=False, compression="snappy")

    raw_count = _write_raw_parquet(run_dir, raw_output, chunk_size=args.chunk_size)

    print(f"Wrote {features_output} rows={len(features):,}")
    print(f"Wrote {raw_output} rows={raw_count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
