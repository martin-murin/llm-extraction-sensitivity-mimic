from __future__ import annotations

# Release documentation:
# Provides shared helpers for claim-registry recomputation.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Supports paper claim recomputation and receipt verification.

import hashlib
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


REPO = Path(__file__).resolve().parents[3]


def require_input_files(paths: Sequence[str | Path]) -> None:
    """Raise FileNotFoundError with a clear message if any input is missing."""
    missing: list[str] = []
    for p in paths:
        abs_path = REPO / Path(p)
        if not abs_path.exists():
            missing.append(str(abs_path))
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing required input file(s):\n  - {joined}")


def read_markdown_table(path: str | Path, header_startswith: str) -> pd.DataFrame:
    """Read a markdown table by header prefix from a file."""
    abs_path = REPO / Path(path)
    text = abs_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(header_startswith):
            start = i
            break
    if start is None:
        raise ValueError(f"Could not find table '{header_startswith}' in {abs_path}")

    # Parse markdown table
    lines = lines[start:]
    if len(lines) < 2:
        raise ValueError(f"Malformed markdown table in {abs_path}")
    header = [x.strip() for x in lines[0].strip().strip("|").split("|")]
    rows: list[list[str]] = []
    for line in lines[2:]:
        if not line.startswith("|"):
            break
        rows.append([x.strip() for x in line.strip().strip("|").split("|")])
    return pd.DataFrame(rows, columns=header)


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert markdown numeric strings (with %, commas, + signs) to float."""
    cleaned = (
        series.astype(str)
        .str.replace("`", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    cleaned = cleaned.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(cleaned, errors="coerce")


def file_sha256(path: str | Path) -> str:
    p = REPO / Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def claim_entry(
    *,
    value: Any,
    description: str,
    computed_at: str,
    receipt: dict[str, Any],
    format_default: str = ".2f",
    sample: str | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "value": value,
        "format_default": format_default,
        "description": description,
        "computed_at": computed_at,
        "receipt": receipt,
    }
    if sample is not None:
        out["sample"] = sample
    if unit is not None:
        out["unit"] = unit
    return out
