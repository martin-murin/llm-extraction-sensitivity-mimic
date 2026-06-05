from __future__ import annotations

# Release documentation:
# Investigates two zero-firing ICD labeling functions from the Phase 9 paper-strengthening prompt.
#
# Reads: MIMIC `diagnoses_icd` via Postgres, production extraction outputs, and LF definitions.
# Writes: `codex_outputs/57_zero_firing_lf_investigation.md`.
# Paper role: supports LF/ICD validation claims by distinguishing true zero prevalence from LF bugs.
# Usage: `python scripts/57b_zero_firing_lf_investigation.py` unless argparse help says otherwise.

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.connection import discover_schemas, get_engine
from src.db.queries import fetch_icd_codes_by_hadm_ids, fetch_primary_icd_by_hadm_ids
from src.labeling_functions.icd_lf import ICD_LF_SPECS

ZERO_FIRING_LF_NAMES = {"icd_dka_hhs_admission", "icd_oncology_treatment_admission"}


@dataclass(frozen=True)
class ZeroLFStatus:
    name: str
    target_value: str
    match_position: str
    pattern_counts_any_rows: int
    pattern_counts_any_hadm: int
    pattern_counts_primary_rows: int
    pattern_counts_primary_hadm: int
    lf_fired_on_production: int
    diagnosis: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigate zero-firing ICD LFs.")
    parser.add_argument("--run-id", default="production_v1")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--output", default="codex_outputs/57_zero_firing_lf_investigation.md")
    return parser.parse_args()


def _md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(cols) + " |"
    divider = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for row in rows:
        body.append(
            "| " + " | ".join(str(row.get(c, "")).replace("|", "\\|") for c in cols) + " |"
        )
    return "\n".join([header, divider, *body])


def _normalize(code: str) -> str:
    return code.strip().upper()


def _matches(code: str, patterns: list[str]) -> bool:
    c = _normalize(code)
    for p in patterns:
        n = _normalize(p)
        if "." in n:
            if c == n:
                return True
            continue
        if c.startswith(n):
            return True
    return False


def _table_names() -> tuple[str, str]:
    engine = get_engine()
    tables = discover_schemas(engine)
    diagnoses = str(tables["diagnoses_icd"])
    schema, table = diagnoses.split(".", maxsplit=1)
    schema_q = schema.replace('"', '""')
    table_q = table.replace('"', '""')
    diagnoses_q = f'"{schema_q}"."{table_q}"'
    return diagnoses, diagnoses_q


def _pattern_condition(
    patterns: list[str],
    version: int,
    alias: str = "d",
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}
    for idx, pattern in enumerate(patterns):
        pname = f"p_{alias}_{version}_{idx}"
        normalized = pattern.upper()
        if "." in normalized:
            clauses.append(f"UPPER({alias}.icd_code::text) = :{pname}")
            params[pname] = normalized
        else:
            clauses.append(f"UPPER({alias}.icd_code::text) LIKE :{pname}")
            params[pname] = normalized + "%"
    version_param = f"v_{alias}_{version}"
    clauses_sql = " OR ".join(f"({c})" for c in clauses) if clauses else "FALSE"
    sql = f"(CAST({alias}.icd_version AS text) = :{version_param} AND ({clauses_sql}))"
    params[version_param] = str(version)
    return sql, params


def _count_patterns(
    diagnoses_q: str,
    icd10_patterns: list[str],
    icd9_patterns: list[str],
    primary_only: bool,
) -> tuple[int, int]:
    engine = get_engine()

    c10, p10 = (
        _pattern_condition(icd10_patterns, 10, alias="d")
        if icd10_patterns
        else ("FALSE", {})
    )
    c9, p9 = (
        _pattern_condition(icd9_patterns, 9, alias="d")
        if icd9_patterns
        else ("FALSE", {})
    )
    where_match = f"(({c10}) OR ({c9}))"
    where_primary = " AND CAST(d.seq_num AS text) = '1'" if primary_only else ""

    query = text(
        f"""
        SELECT
            COUNT(*)::bigint AS n_rows,
            COUNT(DISTINCT d.hadm_id)::bigint AS n_hadm
        FROM {diagnoses_q} AS d
        WHERE {where_match}{where_primary}
        """
    )
    params = {**p10, **p9}

    with engine.connect() as conn:
        row = conn.execute(query, params).mappings().one()
    return int(row["n_rows"]), int(row["n_hadm"])


def _load_production_parsed_hadm_ids(run_id: str) -> list[int]:
    path = Path("data/raw_responses") / run_id / "results.jsonl"
    hadm_ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if bool(row.get("parse_ok", False)):
            hadm_ids.append(int(row["hadm_id"]))
    return hadm_ids


def _iter_chunks(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _evaluate_icd_lf_firing(
    hadm_ids: list[int],
    chunk_size: int,
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    engine = get_engine()
    firing_counts: dict[str, int] = {str(spec["name"]): 0 for spec in ICD_LF_SPECS}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for chunk in _iter_chunks(hadm_ids, chunk_size):
        primary = fetch_primary_icd_by_hadm_ids(engine, chunk)
        all_codes = fetch_icd_codes_by_hadm_ids(engine, chunk)

        for hadm_id in chunk:
            hadm_primary = primary.get(hadm_id)
            hadm_codes = all_codes.get(hadm_id, [])

            for spec in ICD_LF_SPECS:
                name = str(spec["name"])
                pos = str(spec["match_position"])
                matched_code: str | None = None

                if pos == "primary_only":
                    if hadm_primary is not None:
                        code, version = hadm_primary
                        patterns = (
                            spec["icd10_prefixes"]
                            if int(version) == 10
                            else spec["icd9_prefixes"]
                        )
                        if _matches(str(code), [str(p) for p in patterns]):
                            matched_code = str(code)
                    # fallback in implementation: first any code if primary missing
                    elif hadm_codes:
                        code, version = hadm_codes[0]
                        patterns = (
                            spec["icd10_prefixes"]
                            if int(version) == 10
                            else spec["icd9_prefixes"]
                        )
                        if _matches(str(code), [str(p) for p in patterns]):
                            matched_code = str(code)
                else:  # any
                    for code, version in hadm_codes:
                        patterns = (
                            spec["icd10_prefixes"]
                            if int(version) == 10
                            else spec["icd9_prefixes"]
                        )
                        if _matches(str(code), [str(p) for p in patterns]):
                            matched_code = str(code)
                            break

                if matched_code is not None:
                    firing_counts[name] += 1
                    if len(examples[name]) < 5:
                        examples[name].append(
                            {
                                "hadm_id": hadm_id,
                                "matched_code": matched_code,
                                "target_value": str(spec["target_value"]),
                                "match_position": pos,
                            }
                        )

    return firing_counts, examples


def _spotcheck_working_families(
    firing_counts: dict[str, int],
    examples: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    family_examples: dict[str, list[dict[str, Any]]] = {}

    # Family 1: primary_only working LFs
    primary_working = [
        spec
        for spec in ICD_LF_SPECS
        if spec["match_position"] == "primary_only"
        and firing_counts[str(spec["name"])] > 0
    ]
    primary_rows: list[dict[str, Any]] = []
    for spec in primary_working:
        for ex in examples.get(str(spec["name"]), []):
            primary_rows.append({"lf": str(spec["name"]), **ex})
            if len(primary_rows) >= 5:
                break
        if len(primary_rows) >= 5:
            break
    family_examples["primary_only_working"] = primary_rows

    # Family 2: any-position working LFs
    any_working = [
        spec
        for spec in ICD_LF_SPECS
        if spec["match_position"] == "any"
        and firing_counts[str(spec["name"])] > 0
    ]
    any_rows: list[dict[str, Any]] = []
    for spec in any_working:
        for ex in examples.get(str(spec["name"]), []):
            any_rows.append({"lf": str(spec["name"]), **ex})
            if len(any_rows) >= 5:
                break
        if len(any_rows) >= 5:
            break
    family_examples["any_position_working"] = any_rows

    return family_examples


def _diagnose_zero_lf(
    name: str,
    target_value: str,
    match_position: str,
    any_rows: int,
    any_hadm: int,
    prim_rows: int,
    prim_hadm: int,
    firing: int,
) -> str:
    if any_hadm == 0:
        return (
            "No matching ICD codes present in MIMIC diagnoses_icd "
            "(data absence, not LF bug)."
        )
    if match_position == "primary_only" and prim_hadm == 0:
        return (
            "Codes exist, but never as primary diagnosis; zero firing explained by "
            "primary_only constraint."
        )
    if firing == 0 and (prim_hadm > 0 or (match_position == "any" and any_hadm > 0)):
        return (
            "Potential LF match-logic bug: candidate codes exist where LF should "
            "fire, but observed firing is zero."
        )
    return "LF appears consistent with observed code presence and match-position constraints."


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)

    run_id = str(args.run_id)
    hadm_ids = _load_production_parsed_hadm_ids(run_id)
    _, diagnoses_q = _table_names()

    firing_counts, examples = _evaluate_icd_lf_firing(hadm_ids, int(args.chunk_size))

    zero_rows: list[ZeroLFStatus] = []
    for spec in ICD_LF_SPECS:
        name = str(spec["name"])
        if name not in ZERO_FIRING_LF_NAMES:
            continue
        icd10 = [str(v) for v in spec.get("icd10_prefixes", [])]
        icd9 = [str(v) for v in spec.get("icd9_prefixes", [])]

        any_rows, any_hadm = _count_patterns(
            diagnoses_q,
            icd10_patterns=icd10,
            icd9_patterns=icd9,
            primary_only=False,
        )
        prim_rows, prim_hadm = _count_patterns(
            diagnoses_q,
            icd10_patterns=icd10,
            icd9_patterns=icd9,
            primary_only=True,
        )
        firing = int(firing_counts.get(name, 0))

        zero_rows.append(
            ZeroLFStatus(
                name=name,
                target_value=str(spec["target_value"]),
                match_position=str(spec["match_position"]),
                pattern_counts_any_rows=any_rows,
                pattern_counts_any_hadm=any_hadm,
                pattern_counts_primary_rows=prim_rows,
                pattern_counts_primary_hadm=prim_hadm,
                lf_fired_on_production=firing,
                diagnosis=_diagnose_zero_lf(
                    name=name,
                    target_value=str(spec["target_value"]),
                    match_position=str(spec["match_position"]),
                    any_rows=any_rows,
                    any_hadm=any_hadm,
                    prim_rows=prim_rows,
                    prim_hadm=prim_hadm,
                    firing=firing,
                ),
            )
        )

    family_examples = _spotcheck_working_families(firing_counts, examples)

    firing_rows = [
        {
            "lf_name": str(spec["name"]),
            "target": f"{spec['target_field']}::{spec['target_value']}",
            "match_position": str(spec["match_position"]),
            "firing_count": int(firing_counts.get(str(spec["name"]), 0)),
        }
        for spec in ICD_LF_SPECS
    ]
    firing_rows.sort(
        key=lambda r: (
            -cast(int, r["firing_count"]),
            str(r["lf_name"]),
        )
    )

    zero_table = [
        {
            "lf_name": z.name,
            "target_value": z.target_value,
            "match_position": z.match_position,
            "any_rows": z.pattern_counts_any_rows,
            "any_hadm": z.pattern_counts_any_hadm,
            "primary_rows": z.pattern_counts_primary_rows,
            "primary_hadm": z.pattern_counts_primary_hadm,
            "lf_firing_on_production": z.lf_fired_on_production,
            "diagnosis": z.diagnosis,
        }
        for z in zero_rows
    ]

    lines = [
        "# Zero-Firing ICD LF Investigation",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        f"- Production parsed admissions analyzed: **{len(hadm_ids):,}**",
        f"- Zero-firing LFs investigated: `{', '.join(sorted(ZERO_FIRING_LF_NAMES))}`",
        "",
        "## ICD LF firing counts on production parsed admissions",
        "",
        _md_table(
            firing_rows,
            ["lf_name", "target", "match_position", "firing_count"],
        ),
        "",
        "## Zero-firing diagnosis",
        "",
        _md_table(
            zero_table,
            [
                "lf_name",
                "target_value",
                "match_position",
                "any_rows",
                "any_hadm",
                "primary_rows",
                "primary_hadm",
                "lf_firing_on_production",
                "diagnosis",
            ],
        ),
        "",
        "## Spot-check: working LF families (5 hadm_ids each)",
        "",
        "### primary_only_working",
        "",
        _md_table(
            family_examples.get("primary_only_working", []),
            ["lf", "hadm_id", "matched_code", "target_value", "match_position"],
        ),
        "",
        "### any_position_working",
        "",
        _md_table(
            family_examples.get("any_position_working", []),
            ["lf", "hadm_id", "matched_code", "target_value", "match_position"],
        ),
        "",
        "## Conclusion",
        "",
        (
            "- If `any_hadm` is zero, the LF is zero-firing due to absent source "
            "codes in the database."
        ),
        (
            "- If codes exist only in non-primary positions, `primary_only` LFs can "
            "still legitimately fire zero times."
        ),
        (
            "- Only the case where eligible code presence exists but LF firing "
            "remains zero indicates a logic bug."
        ),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
