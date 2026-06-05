from __future__ import annotations

# Release documentation:
# Computes claim-registry values for production.
#
# Reads: data/raw_responses/production_v1/results.jsonl, data/raw_responses/production_v1_retry/results.jsonl, data/splits/methodology_1k.csv, data/splits/methodology_5k_audit_500.csv, data/splits/extended_5k.csv.
# Writes: data/raw_responses/production_v1/results.jsonl, data/raw_responses/production_v1_retry/results.jsonl, data/splits/methodology_1k.csv, data/splits/methodology_5k_audit_500.csv, data/splits/extended_5k.csv.
# Backs paper claim registry entries for production.

import json
import re
from datetime import datetime
from pathlib import Path

from paper.claims.scripts._common import claim_entry, require_input_files
from paper.claims.scripts._receipt import build_receipt, merge_into_claims_json, now_utc_iso

CLAIMS_PATH = Path(__file__).resolve().parent.parent / "claims.json"
SCRIPT_PATH = Path(__file__).resolve()
REPO = Path(__file__).resolve().parents[3]

INPUT_FILES = [
    "data/raw_responses/production_v1/results.jsonl",
    "data/raw_responses/production_v1_retry/results.jsonl",
    "logs/production/20260428T143846_production_v1.log",
    "logs/production/20260428T150350_production_v1.log",
    "data/splits/methodology_1k.csv",
    "data/splits/methodology_5k_audit_500.csv",
    "data/splits/extended_5k.csv",
]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _production_counts() -> tuple[int, int]:
    rows = _read_jsonl(REPO / "data/raw_responses/production_v1/results.jsonl")
    total = len(rows)
    parse_ok = sum(1 for row in rows if bool(row.get("parse_ok", False)))
    return total, parse_ok


def _parse_wall_clock_window(log_paths: list[Path]) -> tuple[float, datetime, datetime]:
    ts_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),")
    first: datetime | None = None
    last: datetime | None = None
    for path in log_paths:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = ts_pattern.match(line)
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            if first is None or t < first:
                first = t
            if last is None or t > last:
                last = t
    if first is None or last is None:
        raise ValueError("Could not parse timestamps from production logs.")
    return float((last - first).total_seconds() / 3600.0), first, last


def compute_all() -> dict:
    require_input_files(INPUT_FILES)
    n_total, n_ok = _production_counts()

    wall_hours, wall_start, wall_end = _parse_wall_clock_window(
        [
            REPO / "logs/production/20260428T143846_production_v1.log",
            REPO / "logs/production/20260428T150350_production_v1.log",
        ]
    )

    timestamp = now_utc_iso()
    receipt = build_receipt(SCRIPT_PATH, "compute_all", INPUT_FILES)
    out = {
        "production_n_admissions": claim_entry(
            value=n_total,
            format_default=",d",
            description="Total admissions submitted in production extraction run",
            sample="production_331793",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "production_n_parse_ok": claim_entry(
            value=n_ok,
            format_default=",d",
            description="Admissions with parse-OK output in production run",
            sample="production_331793",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "production_wall_clock_hours": claim_entry(
            value=wall_hours,
            format_default=".2f",
            description=(
                "Observed wall-clock runtime computed from first/last log timestamps "
                "in production_v1 log files"
            ),
            sample="production_331793",
            computed_at=timestamp,
            receipt=receipt,
        ),
    }
    out["production_wall_clock_hours"]["provenance_note"] = (
        f"log_window_start={wall_start.isoformat(sep=' ')}, "
        f"log_window_end={wall_end.isoformat(sep=' ')}"
    )
    return out

def main() -> int:
    new_claims = compute_all()
    n = merge_into_claims_json(CLAIMS_PATH, new_claims)
    print(f"Updated {n} claims in {CLAIMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
