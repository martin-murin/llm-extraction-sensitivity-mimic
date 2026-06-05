"""
Builds initial refinement, holdout, and smoke note splits.

Reads: configs/optimization.yaml, data/splits, codex_outputs/02_splits_verification.md.
Writes: data/splits, codex_outputs/02_splits_verification.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/02_build_note_splits.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src import config
from src.db.connection import get_engine
from src.db.icd_utils import icd10_chapter_from_code
from src.db.queries import pull_split_candidates, top_primary_icds
from src.io.splits import build_stratified_splits, save_splits
from src.utils.logging import get_logger


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}" if not value.is_integer() else f"{int(value):,}"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"

    body = []
    for row in rows:
        values = []
        for column in columns:
            value = _format_number(row.get(column, ""))
            value = value.replace("\n", " ").replace("|", "\\|")
            values.append(value)
        body.append("| " + " | ".join(values) + " |")

    return "\n".join([header, divider, *body])


def _series_percentile(series: pd.Series, percentile: float) -> float:
    return float(np.percentile(series.to_numpy(dtype=np.int64), percentile))


def _read_saved_splits(
    output_dir: Path,
    refinement_n: int,
    holdout_n: int,
    smoke_n: int,
) -> dict[str, pd.DataFrame]:
    return {
        "refinement": pd.read_csv(output_dir / f"refinement_{refinement_n}.csv"),
        "holdout": pd.read_csv(output_dir / f"holdout_{holdout_n}.csv"),
        "smoke": pd.read_csv(output_dir / f"smoke_{smoke_n}.csv"),
    }


def _build_report(
    report_path: Path,
    output_dir: Path,
    run_id: str,
    seed: int,
    manifest: dict[str, Any],
    candidates: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    engine: Any,
    refinement_n: int,
    holdout_n: int,
    smoke_n: int,
) -> tuple[bool, bool]:
    candidate_pool = candidates.copy()
    candidate_pool["chapter"] = candidate_pool.apply(
        lambda row: icd10_chapter_from_code(
            str(row["primary_icd_code"]),
            int(row["primary_icd_version"]),
        ),
        axis=1,
    )

    candidate_chapter_counts = candidate_pool["chapter"].value_counts().sort_values(ascending=False)
    candidate_rows = [
        {
            "chapter": chapter,
            "count": int(count),
            "pct": (float(count) / len(candidate_pool)) * 100,
        }
        for chapter, count in candidate_chapter_counts.items()
    ]

    all_chapters = sorted(candidate_chapter_counts.index.tolist())
    distribution_rows = []
    for chapter in all_chapters:
        row: dict[str, Any] = {
            "chapter": chapter,
            "candidate_pool_pct": (candidate_chapter_counts[chapter] / len(candidate_pool)) * 100,
        }
        for split_name in ["refinement", "holdout", "smoke"]:
            split_frame = splits[split_name]
            split_count = int((split_frame["chapter"] == chapter).sum())
            split_pct = (split_count / len(split_frame)) * 100
            row[f"{split_name}_count"] = split_count
            row[f"{split_name}_pct"] = split_pct
        distribution_rows.append(row)

    version_rows = []
    for version in [9, 10]:
        refinement_count = int((splits["refinement"]["primary_icd_version"] == version).sum())
        holdout_count = int((splits["holdout"]["primary_icd_version"] == version).sum())
        smoke_count = int((splits["smoke"]["primary_icd_version"] == version).sum())
        version_rows.append(
            {
                "icd_version": version,
                "refinement_count": refinement_count,
                "holdout_count": holdout_count,
                "smoke_count": smoke_count,
            }
        )

    note_length_rows = []
    for split_name in ["refinement", "holdout", "smoke"]:
        series = splits[split_name]["note_char_len"].astype("int64")
        note_length_rows.append(
            {
                "split": split_name,
                "min": int(series.min()),
                "p25": _series_percentile(series, 25),
                "median": _series_percentile(series, 50),
                "p75": _series_percentile(series, 75),
                "max": int(series.max()),
            }
        )

    top20 = top_primary_icds(engine, n=20)
    top20 = top20.copy()
    top20["icd_code"] = top20["icd_code"].astype(str)
    top20["icd_version_str"] = top20["icd_version"].astype(str)

    audit_rows = []
    for _, row in top20.iterrows():
        icd_code = str(row["icd_code"])
        icd_version_str = str(row["icd_version_str"])
        audit_row: dict[str, Any] = {
            "icd_code": icd_code,
            "icd_version": icd_version_str,
            "phase0_top20_count": int(row["count"]),
            "description": str(row["description"]),
        }
        for split_name in ["refinement", "holdout", "smoke"]:
            split_df = splits[split_name]
            count = int(
                (
                    (split_df["primary_icd_code"].astype(str) == icd_code)
                    & (split_df["primary_icd_version"].astype(str) == icd_version_str)
                ).sum()
            )
            audit_row[f"{split_name}_count"] = count
        audit_rows.append(audit_row)

    refinement_ids = set(splits["refinement"]["hadm_id"].astype(int).tolist())
    holdout_ids = set(splits["holdout"]["hadm_id"].astype(int).tolist())
    smoke_ids = set(splits["smoke"]["hadm_id"].astype(int).tolist())

    inter_ref_hold = len(refinement_ids.intersection(holdout_ids)) == 0
    inter_ref_smoke = len(refinement_ids.intersection(smoke_ids)) == 0
    inter_hold_smoke = len(holdout_ids.intersection(smoke_ids)) == 0
    intersection_pass = inter_ref_hold and inter_ref_smoke and inter_hold_smoke

    rebuilt = build_stratified_splits(
        candidate_pool.drop(columns=["chapter"]),
        seed=seed,
        refinement_n=refinement_n,
        holdout_n=holdout_n,
        smoke_n=smoke_n,
    )
    saved = _read_saved_splits(
        output_dir=output_dir,
        refinement_n=refinement_n,
        holdout_n=holdout_n,
        smoke_n=smoke_n,
    )

    determinism_pass = True
    for split_name in ["refinement", "holdout", "smoke"]:
        rebuilt_ids = rebuilt[split_name]["hadm_id"].astype(int).tolist()
        saved_ids = saved[split_name]["hadm_id"].astype(int).tolist()
        if rebuilt_ids != saved_ids:
            determinism_pass = False
            break

    run_meta_rows = [
        {"field": "run_id", "value": run_id},
        {"field": "timestamp_utc", "value": datetime.now(tz=UTC).isoformat()},
        {"field": "seed", "value": seed},
        {"field": "source_table", "value": manifest["source_table"]},
        {"field": "candidate_pool_size", "value": manifest["source_row_count"]},
    ]

    checksum_rows = [
        {"filename": filename, "sha256": checksum}
        for filename, checksum in sorted(manifest["checksums_sha256"].items())
    ]

    section_lines = [
        "# Splits Verification",
        "",
        "## Run metadata",
        _markdown_table(run_meta_rows, ["field", "value"]),
        "",
        "## Manifest summary",
        _markdown_table(checksum_rows, ["filename", "sha256"]),
        "",
        "## Candidate pool characterization",
        f"Total eligible hadm_ids: {len(candidate_pool):,}",
        "",
        _markdown_table(candidate_rows, ["chapter", "count", "pct"]),
        "",
        "## Per-split chapter distribution",
        _markdown_table(
            distribution_rows,
            [
                "chapter",
                "refinement_count",
                "refinement_pct",
                "holdout_count",
                "holdout_pct",
                "smoke_count",
                "smoke_pct",
                "candidate_pool_pct",
            ],
        ),
        "",
        "## ICD version balance",
        _markdown_table(
            version_rows,
            ["icd_version", "refinement_count", "holdout_count", "smoke_count"],
        ),
        "",
        "## Note length balance",
        _markdown_table(note_length_rows, ["split", "min", "p25", "median", "p75", "max"]),
        "",
        "## Primary ICD prevalence audit",
        _markdown_table(
            audit_rows,
            [
                "icd_code",
                "icd_version",
                "phase0_top20_count",
                "refinement_count",
                "holdout_count",
                "smoke_count",
                "description",
            ],
        ),
        "",
        "## Intersection tests",
        _markdown_table(
            [
                {
                    "test": "refinement ∩ holdout == ∅",
                    "result": "PASS" if inter_ref_hold else "FAIL",
                },
                {
                    "test": "refinement ∩ smoke == ∅",
                    "result": "PASS" if inter_ref_smoke else "FAIL",
                },
                {
                    "test": "holdout ∩ smoke == ∅",
                    "result": "PASS" if inter_hold_smoke else "FAIL",
                },
            ],
            ["test", "result"],
        ),
        "",
        "## Determinism check",
        f"Result: {'PASS' if determinism_pass else 'FAIL'}",
        "",
        "## Questions and assumptions",
        (
            "- ICD chapter mapping uses the requested coarse chapter buckets and maps "
            "unknown/malformed codes to `ZZ. Unmapped`."
        ),
        (
            "- For split eligibility and deterministic primary code selection, `seq_num` is "
            "parsed numerically when possible; when `seq_num=1` is absent, the smallest "
            "parsed seq_num is used with lexical `icd_code` tie-break."
        ),
        (
            "- Split regeneration protection is enforced by `SPLITS_MANIFEST.json`; "
            "script `--force` explicitly removes the prior manifest before writing a new one."
        ),
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(section_lines), encoding="utf-8")
    return intersection_pass, determinism_pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen methodology note splits.")
    parser.add_argument("--config", default="configs/optimization.yaml")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--report", default="codex_outputs/02_splits_verification.md")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("scripts.02_build_note_splits")

    config.load_env()

    config_path = Path(args.config)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    seed = int(settings["random_seed"])
    refinement_n = int(settings["refinement_split_size"])
    holdout_n = int(settings["holdout_split_size"])
    smoke_n = int(settings["smoke_split_size"])

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "SPLITS_MANIFEST.json"

    if args.force and manifest_path.exists():
        logger.warning("--force enabled: deleting existing manifest before regeneration.")
        manifest_path.unlink()

    engine = get_engine()
    candidates = pull_split_candidates(engine)
    logger.info("Loaded split candidates", extra={"n_candidates": len(candidates)})

    splits = build_stratified_splits(
        candidates,
        seed=seed,
        refinement_n=refinement_n,
        holdout_n=holdout_n,
        smoke_n=smoke_n,
    )
    manifest = save_splits(splits, output_dir=output_dir)

    report_path = Path(args.report)
    intersection_pass, determinism_pass = _build_report(
        report_path=report_path,
        output_dir=output_dir,
        run_id=str(settings.get("run_id", "opt_v1")),
        seed=seed,
        manifest=manifest,
        candidates=candidates,
        splits=splits,
        engine=engine,
        refinement_n=refinement_n,
        holdout_n=holdout_n,
        smoke_n=smoke_n,
    )

    print(f"Wrote splits: {json.dumps(manifest['sizes'])}")
    print(f"Wrote manifest: {output_dir / 'SPLITS_MANIFEST.json'}")
    print(f"Wrote report: {report_path}")

    if not intersection_pass:
        logger.error("Intersection tests failed. See report for details.")
        return 1

    if not determinism_pass:
        logger.error("Determinism check failed. See report for details.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
