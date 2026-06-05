from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src import config
from src.db.icd_utils import icd10_chapter_from_code

logger = logging.getLogger(__name__)

STRATIFICATION_STRATEGY = "icd10_chapter_proportional_largest_remainder"


def _largest_remainder_allocation(weights: pd.Series, total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be >= 0")

    if total == 0:
        return {str(chapter): 0 for chapter in weights.index}

    positive = weights[weights > 0]
    if positive.empty:
        raise ValueError("weights must include at least one positive value")

    raw = (positive / positive.sum()) * total
    floors = np.floor(raw).astype(int)
    remainders = raw - floors

    allocation = {str(chapter): int(value) for chapter, value in floors.items()}
    remaining = int(total - sum(allocation.values()))
    if remaining > 0:
        tie_break = sorted(
            positive.index,
            key=lambda chapter: (-float(remainders[chapter]), str(chapter)),
        )
        for chapter in tie_break[:remaining]:
            allocation[str(chapter)] += 1

    for chapter in weights.index:
        allocation.setdefault(str(chapter), 0)

    return allocation


def build_stratified_splits(
    candidates: pd.DataFrame,
    seed: int,
    refinement_n: int = 150,
    holdout_n: int = 150,
    smoke_n: int = 200,
) -> dict[str, pd.DataFrame]:
    required_columns = {
        "hadm_id",
        "subject_id",
        "primary_icd_code",
        "primary_icd_version",
        "note_char_len",
        "n_diagnoses",
    }
    missing = required_columns.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing required candidate columns: {sorted(missing)}")

    total_n = refinement_n + holdout_n + smoke_n
    if len(candidates) < total_n:
        raise ValueError(
            f"Candidate pool too small: need {total_n}, found {len(candidates)}"
        )

    pool = candidates.copy()
    pool["chapter"] = pool.apply(
        lambda row: icd10_chapter_from_code(
            str(row["primary_icd_code"]),
            int(row["primary_icd_version"]),
        ),
        axis=1,
    )

    rng = np.random.default_rng(seed)
    pool["_chapter_shuffle"] = rng.random(len(pool))
    pool = pool.sort_values(
        by=["chapter", "_chapter_shuffle", "hadm_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    chapter_counts = pool["chapter"].value_counts().sort_index()
    initial_quotas = _largest_remainder_allocation(chapter_counts.astype(float), total_n)

    final_quotas = {
        chapter: min(initial_quotas[chapter], int(chapter_counts[chapter]))
        for chapter in chapter_counts.index
    }
    shortfall = total_n - sum(final_quotas.values())

    if shortfall > 0:
        logger.warning(
            "Initial chapter quota shortfall detected; redistributing proportionally.",
            extra={"shortfall": shortfall},
        )

    while shortfall > 0:
        capacity = {
            chapter: int(chapter_counts[chapter] - final_quotas[chapter])
            for chapter in chapter_counts.index
            if chapter_counts[chapter] - final_quotas[chapter] > 0
        }

        if not capacity:
            raise RuntimeError(
                "Unable to redistribute split shortfall: no remaining chapter capacity."
            )

        add = _largest_remainder_allocation(pd.Series(capacity, dtype=float), shortfall)
        granted = 0
        for chapter, requested in add.items():
            room = capacity.get(chapter, 0)
            grant = min(int(requested), int(room))
            if grant > 0:
                final_quotas[chapter] += grant
                granted += grant

        shortfall -= granted
        if granted == 0:
            raise RuntimeError("Redistribution stalled with non-zero shortfall.")

    selected_frames: list[pd.DataFrame] = []
    for chapter in sorted(final_quotas):
        quota = int(final_quotas[chapter])
        if quota <= 0:
            continue
        chapter_rows = pool[pool["chapter"] == chapter].head(quota)
        selected_frames.append(chapter_rows)

    full = pd.concat(selected_frames, ignore_index=True)
    if len(full) != total_n:
        raise RuntimeError(
            f"Stratified cohort size mismatch: expected {total_n}, got {len(full)}"
        )

    if full["hadm_id"].duplicated().any():
        raise RuntimeError("Duplicate hadm_id detected in the stratified cohort.")

    full["_global_shuffle"] = rng.random(len(full))
    full = full.sort_values(
        by=["_global_shuffle", "hadm_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    base_columns = [
        "hadm_id",
        "subject_id",
        "primary_icd_code",
        "primary_icd_version",
        "chapter",
        "note_char_len",
        "n_diagnoses",
    ]
    full = full[base_columns]

    refinement = full.iloc[:refinement_n].reset_index(drop=True)
    holdout = full.iloc[refinement_n : refinement_n + holdout_n].reset_index(drop=True)
    smoke = full.iloc[refinement_n + holdout_n :].reset_index(drop=True)

    meta: dict[str, Any] = {
        "random_seed": int(seed),
        "source_table": str(candidates.attrs.get("source_table", "public.discharge_note")),
        "source_row_count": len(candidates),
        "stratification_strategy": STRATIFICATION_STRATEGY,
        "chapter_quotas": {chapter: int(final_quotas[chapter]) for chapter in sorted(final_quotas)},
    }

    splits = {
        "refinement": refinement,
        "holdout": holdout,
        "smoke": smoke,
    }

    for frame in splits.values():
        frame.attrs.update(meta)

    return splits


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()
    return commit if commit else None


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "SPLITS_MANIFEST.json"

    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing split manifest at '{manifest_path}'. "
            "Delete it explicitly (or use script --force) to regenerate splits."
        )

    required_keys = {"refinement", "holdout", "smoke"}
    if required_keys.difference(splits):
        raise ValueError(f"Splits dict must contain keys: {sorted(required_keys)}")

    ordered_columns = [
        "hadm_id",
        "subject_id",
        "primary_icd_code",
        "primary_icd_version",
        "chapter",
        "note_char_len",
        "n_diagnoses",
    ]

    file_map = {
        "refinement": output_dir / f"refinement_{len(splits['refinement'])}.csv",
        "holdout": output_dir / f"holdout_{len(splits['holdout'])}.csv",
        "smoke": output_dir / f"smoke_{len(splits['smoke'])}.csv",
    }

    for split_name, path in file_map.items():
        frame = splits[split_name].copy()
        present = [column for column in ordered_columns if column in frame.columns]
        frame[present].to_csv(path, index=False)

    checksums = {path.name: _sha256_file(path) for path in file_map.values()}

    all_chapters: set[str] = set()
    chapter_actual_counts_per_split: dict[str, dict[str, int]] = {}
    for split_name in ["refinement", "holdout", "smoke"]:
        counts = splits[split_name]["chapter"].value_counts().to_dict()
        casted = {str(chapter): int(count) for chapter, count in counts.items()}
        chapter_actual_counts_per_split[split_name] = casted
        all_chapters.update(casted.keys())

    chapter_quotas_raw = splits["refinement"].attrs.get("chapter_quotas", {})
    chapter_quotas = {str(chapter): int(quota) for chapter, quota in chapter_quotas_raw.items()}
    all_chapters.update(chapter_quotas.keys())

    for split_name in chapter_actual_counts_per_split:
        for chapter in all_chapters:
            chapter_actual_counts_per_split[split_name].setdefault(chapter, 0)
        chapter_actual_counts_per_split[split_name] = {
            chapter: chapter_actual_counts_per_split[split_name][chapter]
            for chapter in sorted(chapter_actual_counts_per_split[split_name])
        }

    manifest = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "random_seed": int(splits["refinement"].attrs.get("random_seed", 0)),
        "source_table": str(
            splits["refinement"].attrs.get("source_table", "public.discharge_note")
        ),
        "source_row_count": int(splits["refinement"].attrs.get("source_row_count", 0)),
        "stratification_strategy": str(
            splits["refinement"].attrs.get("stratification_strategy", STRATIFICATION_STRATEGY)
        ),
        "sizes": {
            "refinement": len(splits["refinement"]),
            "holdout": len(splits["holdout"]),
            "smoke": len(splits["smoke"]),
        },
        "chapter_quotas": {
            chapter: chapter_quotas.get(chapter, 0)
            for chapter in sorted(all_chapters)
        },
        "chapter_actual_counts_per_split": chapter_actual_counts_per_split,
        "checksums_sha256": checksums,
        "git_commit": _git_commit_hash(),
        "code_version": "02",
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
