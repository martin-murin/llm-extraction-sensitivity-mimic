"""
Builds the methodology 5k and audit subset splits.

Reads: data/splits/methodology_5k.csv, data/splits/methodology_5k_audit_500.csv, data/splits/SPLITS_MANIFEST.json, codex_outputs/26_methodology_5k_split_build.md.
Writes: data/splits/methodology_5k.csv, data/splits/methodology_5k_audit_500.csv, data/splits/SPLITS_MANIFEST.json, codex_outputs/26_methodology_5k_split_build.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/25_build_5k_split.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from datetime import UTC, datetime
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from src import config
from src.db.connection import get_engine
from src.db.icd_utils import icd10_chapter_from_code
from src.db.queries import pull_split_candidates
from src.io.splits import build_stratified_splits
from src.utils.threeway_kappa import file_sha256


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


def _read_hadm_ids(split_path: Path) -> set[int]:
    frame = pd.read_csv(split_path)
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").astype("int64").tolist())


def _excluded_hadm_ids(split_dir: Path) -> set[int]:
    expected = [
        split_dir / "refinement_150.csv",
        split_dir / "holdout_150.csv",
        split_dir / "smoke_200.csv",
        split_dir / "methodology_1k.csv",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required existing split files missing: {missing}")
    excluded: set[int] = set()
    for path in expected:
        excluded.update(_read_hadm_ids(path))
    return excluded


def filter_candidates_excluding_hadm_ids(
    candidates: pd.DataFrame,
    excluded_hadm_ids: set[int],
) -> pd.DataFrame:
    return candidates[~candidates["hadm_id"].isin(excluded_hadm_ids)].copy()


def _chapter_balance_rows(
    *,
    population: pd.DataFrame,
    sampled: pd.DataFrame,
) -> tuple[list[dict[str, Any]], bool]:
    pop_counts = population["chapter"].value_counts(normalize=True)
    sample_counts = sampled["chapter"].value_counts(normalize=True)
    chapters = sorted(set(pop_counts.index.tolist()) | set(sample_counts.index.tolist()))
    rows: list[dict[str, Any]] = []
    all_within = True
    for chapter in chapters:
        pop_pct = float(pop_counts.get(chapter, 0.0) * 100.0)
        sample_pct = float(sample_counts.get(chapter, 0.0) * 100.0)
        delta_pp = abs(sample_pct - pop_pct)
        if delta_pp > 2.0:
            all_within = False
        rows.append(
            {
                "chapter": chapter,
                "population_pct": f"{pop_pct:.2f}",
                "sample_pct": f"{sample_pct:.2f}",
                "abs_delta_pp": f"{delta_pp:.2f}",
                "within_2pp": "yes" if delta_pp <= 2.0 else "no",
            }
        )
    rows.sort(key=lambda row: float(row["abs_delta_pp"]), reverse=True)
    return rows, all_within


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen methodology 5k split and audit subset."
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--n-notes", type=int, default=5000)
    parser.add_argument("--audit-n", type=int, default=500)
    parser.add_argument("--output", default="data/splits/methodology_5k.csv")
    parser.add_argument("--audit-output", default="data/splits/methodology_5k_audit_500.csv")
    parser.add_argument("--manifest", default="data/splits/SPLITS_MANIFEST.json")
    parser.add_argument(
        "--report",
        default="codex_outputs/26_methodology_5k_split_build.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    output_path = Path(args.output)
    audit_output_path = Path(args.audit_output)
    manifest_path = Path(args.manifest)
    report_path = Path(args.report)

    split_dir = config.SPLITS_DIR
    excluded = _excluded_hadm_ids(split_dir)

    engine = get_engine()
    candidates = pull_split_candidates(engine)
    candidates = candidates.copy()

    filtered = filter_candidates_excluding_hadm_ids(candidates, excluded)
    if len(filtered) < int(args.n_notes):
        raise RuntimeError(
            "Not enough eligible candidates after exclusion: "
            f"need {args.n_notes}, found {len(filtered)}"
        )

    filtered["chapter"] = filtered.apply(
        lambda row: icd10_chapter_from_code(
            str(row["primary_icd_code"]),
            int(row["primary_icd_version"]),
        ),
        axis=1,
    )

    stratified = build_stratified_splits(
        filtered,
        seed=int(args.seed),
        refinement_n=int(args.n_notes),
        holdout_n=0,
        smoke_n=0,
    )["refinement"].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_cols = [
        "hadm_id",
        "subject_id",
        "primary_icd_code",
        "primary_icd_version",
        "chapter",
        "note_char_len",
        "n_diagnoses",
    ]
    stratified[ordered_cols].to_csv(output_path, index=False)

    sampled_hadm_ids = sorted(
        pd.to_numeric(stratified["hadm_id"], errors="coerce").astype("int64").tolist()
    )
    rng = random.Random(int(args.seed))
    if len(sampled_hadm_ids) < int(args.audit_n):
        raise RuntimeError(
            f"Cannot draw audit subset of size {args.audit_n}; only {len(sampled_hadm_ids)} ids"
        )
    audit_hadm_ids = sorted(rng.sample(sampled_hadm_ids, int(args.audit_n)))
    audit_frame = stratified[stratified["hadm_id"].isin(audit_hadm_ids)].copy()
    audit_frame = audit_frame.sort_values("hadm_id", kind="mergesort").reset_index(drop=True)
    audit_frame[ordered_cols].to_csv(audit_output_path, index=False)

    hadm_ids = set(sampled_hadm_ids)
    overlap_count = len(hadm_ids.intersection(excluded))
    unique_count = len(hadm_ids)
    unique_ok = unique_count == int(args.n_notes)
    overlap_ok = overlap_count == 0

    chapter_rows, chapter_ok = _chapter_balance_rows(population=filtered, sampled=stratified)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = manifest.get("checksums_sha256")
    if not isinstance(checksums, dict):
        raise RuntimeError("Manifest missing checksums_sha256 mapping.")

    checksum_5k = file_sha256(output_path)
    checksum_audit = file_sha256(audit_output_path)

    checksums["methodology_5k"] = checksum_5k
    checksums[output_path.name] = checksum_5k
    checksums["methodology_5k_audit_500"] = checksum_audit
    checksums[audit_output_path.name] = checksum_audit
    manifest["checksums_sha256"] = checksums

    now = datetime.now(tz=UTC).isoformat()
    manifest["methodology_5k"] = {
        "generated_at_utc": now,
        "seed": int(args.seed),
        "n_notes": int(args.n_notes),
        "source": "candidates_excluding_refinement_holdout_smoke_methodology_1k",
    }
    manifest["methodology_5k_audit_500"] = {
        "generated_at_utc": now,
        "seed": int(args.seed),
        "n_notes": int(args.audit_n),
        "source": "random_sample_from_methodology_5k",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = [
        {"metric": "timestamp_utc", "value": now},
        {"metric": "seed", "value": int(args.seed)},
        {"metric": "target_n", "value": int(args.n_notes)},
        {"metric": "actual_n", "value": unique_count},
        {"metric": "target_audit_n", "value": int(args.audit_n)},
        {"metric": "actual_audit_n", "value": len(audit_frame)},
        {"metric": "zero_overlap_with_existing_1500", "value": overlap_ok},
        {"metric": "overlap_count", "value": overlap_count},
        {"metric": "chapter_distribution_within_2pp", "value": chapter_ok},
        {"metric": "output_csv", "value": str(output_path)},
        {"metric": "audit_output_csv", "value": str(audit_output_path)},
        {"metric": "manifest", "value": str(manifest_path)},
        {"metric": "sha256_methodology_5k", "value": checksum_5k},
        {"metric": "sha256_methodology_5k_audit_500", "value": checksum_audit},
    ]

    lines = [
        "# Methodology 5k Split Build",
        "",
        "## Summary",
        _markdown_table(summary_rows, ["metric", "value"]),
        "",
        "## Chapter distribution check (population vs methodology_5k)",
        _markdown_table(
            chapter_rows,
            ["chapter", "population_pct", "sample_pct", "abs_delta_pp", "within_2pp"],
        ),
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    if not unique_ok:
        raise RuntimeError(
            "Methodology 5k unique-count check failed: "
            f"expected {args.n_notes}, got {unique_count}"
        )
    if len(audit_frame) != int(args.audit_n):
        raise RuntimeError(
            "Methodology 5k audit subset size mismatch: "
            f"expected {args.audit_n}, got {len(audit_frame)}"
        )
    if not overlap_ok:
        raise RuntimeError(f"Methodology 5k split overlaps existing used-note set: {overlap_count}")
    if not chapter_ok:
        raise RuntimeError("Methodology 5k chapter-distribution tolerance failed (>2pp).")

    print(f"Wrote methodology split CSV to {output_path}")
    print(f"Wrote methodology audit subset CSV to {audit_output_path}")
    print(f"Updated manifest at {manifest_path}")
    print(f"Wrote split-build report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
