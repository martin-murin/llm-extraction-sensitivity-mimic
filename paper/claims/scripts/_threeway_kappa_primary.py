from __future__ import annotations

# Release documentation:
# Provides shared helpers for claim-registry recomputation.
#
# Reads: data/raw_responses/*/results.jsonl.
# Writes: data/raw_responses/*/results.jsonl.
# Supports paper claim recomputation and receipt verification.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.schema.fields import LLMNoteFeatures
from src.schema.vocabulary import ADMISSION_REASON_TAGS
from src.utils.threeway_kappa import (
    cohen_kappa_safe,
    count_positive_admission_tag,
    count_positive_enum,
    count_positive_tristate,
    encode_admission_reason_tags,
    encode_tristate,
    low_base_rate_flag,
)

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw_responses"
SPLITS = REPO / "data" / "splits"

PAIR_KEYS: tuple[tuple[str, str], ...] = (("A", "B"), ("A", "C"), ("B", "C"))
SKIP_KAPPA_FIELDS: set[str] = {
    "primary_diagnosis_text",
    "reasoning",
    "new_meds_started_count",
    "meds_stopped_count",
}
TRISTATE_FIELDS: set[str] = {
    "shock_present",
    "infection_as_trigger",
    "aki_present",
    "lives_alone",
    "social_support_absent",
    "financial_hardship",
    "substance_use_active",
    "fall_risk_documented",
    "cognitive_impairment",
    "goals_of_care_flag",
    "palliative_care_consult",
    "dnr_dni_documented",
    "home_health_ordered",
    "cardiac_rehab_referred",
    "discharge_delayed_reason",
    "hospital_acquired_complication",
    "unresolved_diagnosis_at_discharge",
}
ENUM_FIELDS: set[str] = {"functional_status", "mental_status", "discharge_condition_category"}


@dataclass(frozen=True)
class ThreeWaySummary:
    sample_key: str
    run_ids: tuple[str, str, str]
    n_intersection_notes: int
    n_total_fields: int
    n_fields_filtered: int
    median_kappa_filtered: float


@dataclass(frozen=True)
class ThreeWayDetails:
    sample_key: str
    run_ids: tuple[str, str, str]
    n_intersection_notes: int
    n_total_fields: int
    n_fields_filtered: int
    median_kappa_filtered: float
    filtered_kappa_values: tuple[float, ...]


SAMPLE_SPECS: dict[str, tuple[tuple[str, str, str], str]] = {
    "refinement_150": (
        ("refinement_v1_a", "refinement_v1_b", "refinement_v3_c"),
        "refinement_150.csv",
    ),
    "holdout_150": (
        ("holdout_v1_a", "holdout_v1_b", "holdout_v1_c"),
        "holdout_150.csv",
    ),
    "methodology_1k": (
        ("methodology_1k_a", "methodology_1k_b", "methodology_1k_c"),
        "methodology_1k.csv",
    ),
    "methodology_5k_audit_500": (
        (
            "methodology_5k_a_subset500",
            "methodology_5k_audit_b",
            "methodology_5k_audit_c",
        ),
        "methodology_5k_audit_500.csv",
    ),
    "extended_5k": (
        ("production_v1", "extended_5k_b", "extended_5k_c"),
        "extended_5k.csv",
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_run_parsed_features(run_id: str) -> dict[int, dict[str, Any]]:
    path = RAW / run_id / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing run results file: {path}")

    parsed: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if not bool(row.get("parse_ok", False)):
            continue
        features = row.get("features_json")
        if not isinstance(features, dict):
            continue
        parsed[int(row["hadm_id"])] = features
    return parsed


def _read_split_ids(split_csv_name: str) -> set[int]:
    split_path = SPLITS / split_csv_name
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_path}")
    frame = pd.read_csv(split_path)
    if "hadm_id" not in frame.columns:
        raise ValueError(f"Split CSV missing hadm_id column: {split_path}")
    return set(pd.to_numeric(frame["hadm_id"], errors="coerce").dropna().astype("int64").tolist())


def _field_names() -> list[str]:
    return [
        field
        for field in LLMNoteFeatures.model_fields
        if field not in SKIP_KAPPA_FIELDS
    ]


def _kappa_mean(encoded: dict[str, list[int]]) -> float:
    vals: list[float] = []
    for left, right in PAIR_KEYS:
        vals.append(cohen_kappa_safe(encoded[left], encoded[right]))
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def recompute_filtered_median(
    sample_key: str,
    run_ids: tuple[str, str, str],
    split_csv_name: str,
) -> ThreeWaySummary:
    details = recompute_sample_details(sample_key, run_ids, split_csv_name)
    return ThreeWaySummary(
        sample_key=details.sample_key,
        run_ids=details.run_ids,
        n_intersection_notes=details.n_intersection_notes,
        n_total_fields=details.n_total_fields,
        n_fields_filtered=details.n_fields_filtered,
        median_kappa_filtered=details.median_kappa_filtered,
    )


def recompute_sample_details(
    sample_key: str,
    run_ids: tuple[str, str, str],
    split_csv_name: str,
) -> ThreeWayDetails:
    allowed_ids = _read_split_ids(split_csv_name)
    parsed_by_variant = {
        "A": {
            hadm_id: feats
            for hadm_id, feats in _load_run_parsed_features(run_ids[0]).items()
            if hadm_id in allowed_ids
        },
        "B": {
            hadm_id: feats
            for hadm_id, feats in _load_run_parsed_features(run_ids[1]).items()
            if hadm_id in allowed_ids
        },
        "C": {
            hadm_id: feats
            for hadm_id, feats in _load_run_parsed_features(run_ids[2]).items()
            if hadm_id in allowed_ids
        },
    }

    hadm_ids = sorted(
        set(parsed_by_variant["A"])
        & set(parsed_by_variant["B"])
        & set(parsed_by_variant["C"])
    )
    if not hadm_ids:
        raise ValueError(f"No shared hadm_id intersection for sample '{sample_key}'")

    kappa_rows: list[dict[str, Any]] = []
    for field in _field_names():
        if field == "admission_reason_tags":
            for tag in ADMISSION_REASON_TAGS:
                encoded: dict[str, list[int]] = {}
                positives = {"A": 0, "B": 0, "C": 0}
                for variant in ("A", "B", "C"):
                    tag_lists = [
                        list(parsed_by_variant[variant][hadm_id].get("admission_reason_tags", []))
                        for hadm_id in hadm_ids
                    ]
                    encoded[variant] = [
                        encode_admission_reason_tags(tags)[tag] for tags in tag_lists
                    ]
                    positives[variant] = count_positive_admission_tag(tag_lists, tag)
                n_positive_total = positives["A"] + positives["B"] + positives["C"]
                k_ab = cohen_kappa_safe(encoded["A"], encoded["B"])
                k_ac = cohen_kappa_safe(encoded["A"], encoded["C"])
                k_bc = cohen_kappa_safe(encoded["B"], encoded["C"])
                kappa_rows.append(
                    {
                        "field": f"admission_reason_tags::{tag}",
                        "kappa_A_B": k_ab,
                        "kappa_A_C": k_ac,
                        "kappa_B_C": k_bc,
                        "kappa_mean": float(np.mean(np.asarray([k_ab, k_ac, k_bc], dtype=np.float64))),
                        "low_base_rate_flag": bool(low_base_rate_flag(n_positive_total)),
                    }
                )
            continue

        if field in TRISTATE_FIELDS:
            encoded = {}
            positives = {"A": 0, "B": 0, "C": 0}
            for variant in ("A", "B", "C"):
                values = [
                    str(parsed_by_variant[variant][hadm_id].get(field, "not_documented"))
                    for hadm_id in hadm_ids
                ]
                encoded[variant] = [encode_tristate(value) for value in values]
                positives[variant] = count_positive_tristate(values)
            n_positive_total = positives["A"] + positives["B"] + positives["C"]
            k_ab = cohen_kappa_safe(encoded["A"], encoded["B"])
            k_ac = cohen_kappa_safe(encoded["A"], encoded["C"])
            k_bc = cohen_kappa_safe(encoded["B"], encoded["C"])
            kappa_rows.append(
                {
                    "field": field,
                    "kappa_A_B": k_ab,
                    "kappa_A_C": k_ac,
                    "kappa_B_C": k_bc,
                    "kappa_mean": float(np.mean(np.asarray([k_ab, k_ac, k_bc], dtype=np.float64))),
                    "low_base_rate_flag": bool(low_base_rate_flag(n_positive_total)),
                }
            )
            continue

        if field in ENUM_FIELDS:
            encoded = {}
            positives = {"A": 0, "B": 0, "C": 0}
            observed = sorted(
                {
                    str(parsed_by_variant[variant][hadm_id].get(field))
                    for variant in ("A", "B", "C")
                    for hadm_id in hadm_ids
                }
            )
            mapping = {value: idx for idx, value in enumerate(observed)}
            for variant in ("A", "B", "C"):
                values = [
                    str(parsed_by_variant[variant][hadm_id].get(field))
                    for hadm_id in hadm_ids
                ]
                encoded[variant] = [mapping[value] for value in values]
                positives[variant] = count_positive_enum(values)
            n_positive_total = positives["A"] + positives["B"] + positives["C"]
            k_ab = cohen_kappa_safe(encoded["A"], encoded["B"])
            k_ac = cohen_kappa_safe(encoded["A"], encoded["C"])
            k_bc = cohen_kappa_safe(encoded["B"], encoded["C"])
            kappa_rows.append(
                {
                    "field": field,
                    "kappa_A_B": k_ab,
                    "kappa_A_C": k_ac,
                    "kappa_B_C": k_bc,
                    "kappa_mean": float(np.mean(np.asarray([k_ab, k_ac, k_bc], dtype=np.float64))),
                    "low_base_rate_flag": bool(low_base_rate_flag(n_positive_total)),
                }
            )
            continue

        if field == "dominant_admission_reason":
            mapping = {tag: idx for idx, tag in enumerate(ADMISSION_REASON_TAGS)}
            fallback = mapping["other"]
            encoded = {}
            for variant in ("A", "B", "C"):
                values = [
                    str(parsed_by_variant[variant][hadm_id].get(field, "other"))
                    for hadm_id in hadm_ids
                ]
                encoded[variant] = [mapping.get(value, fallback) for value in values]
            k_ab = cohen_kappa_safe(encoded["A"], encoded["B"])
            k_ac = cohen_kappa_safe(encoded["A"], encoded["C"])
            k_bc = cohen_kappa_safe(encoded["B"], encoded["C"])
            kappa_rows.append(
                {
                    "field": field,
                    "kappa_A_B": k_ab,
                    "kappa_A_C": k_ac,
                    "kappa_B_C": k_bc,
                    "kappa_mean": float(np.mean(np.asarray([k_ab, k_ac, k_bc], dtype=np.float64))),
                    "low_base_rate_flag": False,
                }
            )

    filtered = [row for row in kappa_rows if not bool(row["low_base_rate_flag"])]
    if not filtered:
        raise ValueError(f"No filtered fields remained for sample '{sample_key}'")
    filtered_kappas = np.asarray([float(row["kappa_mean"]) for row in filtered], dtype=np.float64)

    return ThreeWayDetails(
        sample_key=sample_key,
        run_ids=run_ids,
        n_intersection_notes=len(hadm_ids),
        n_total_fields=len(kappa_rows),
        n_fields_filtered=len(filtered),
        median_kappa_filtered=float(np.median(filtered_kappas)),
        filtered_kappa_values=tuple(float(v) for v in filtered_kappas.tolist()),
    )


def recompute_per_variant_medians(
    sample_key: str,
    run_ids: tuple[str, str, str],
    split_csv_name: str,
) -> dict[str, float]:
    """Compute per-variant medians from the same filtered field rows.

    Variant mapping matches historical sidecar logic:
    - A: mean(kappa_A_B, kappa_A_C)
    - B: mean(kappa_A_B, kappa_B_C)
    - C: mean(kappa_A_C, kappa_B_C)
    Then median across filtered rows.
    """
    allowed_ids = _read_split_ids(split_csv_name)
    parsed_by_variant = {
        "A": {
            hadm_id: feats
            for hadm_id, feats in _load_run_parsed_features(run_ids[0]).items()
            if hadm_id in allowed_ids
        },
        "B": {
            hadm_id: feats
            for hadm_id, feats in _load_run_parsed_features(run_ids[1]).items()
            if hadm_id in allowed_ids
        },
        "C": {
            hadm_id: feats
            for hadm_id, feats in _load_run_parsed_features(run_ids[2]).items()
            if hadm_id in allowed_ids
        },
    }
    hadm_ids = sorted(
        set(parsed_by_variant["A"])
        & set(parsed_by_variant["B"])
        & set(parsed_by_variant["C"])
    )
    if not hadm_ids:
        raise ValueError(f"No shared hadm_id intersection for sample '{sample_key}'")

    rows: list[dict[str, float | bool]] = []
    for field in _field_names():
        if field == "admission_reason_tags":
            for tag in ADMISSION_REASON_TAGS:
                encoded: dict[str, list[int]] = {}
                positives = {"A": 0, "B": 0, "C": 0}
                for variant in ("A", "B", "C"):
                    tag_lists = [
                        list(parsed_by_variant[variant][hadm_id].get("admission_reason_tags", []))
                        for hadm_id in hadm_ids
                    ]
                    encoded[variant] = [
                        encode_admission_reason_tags(tags)[tag] for tags in tag_lists
                    ]
                    positives[variant] = count_positive_admission_tag(tag_lists, tag)
                n_positive_total = positives["A"] + positives["B"] + positives["C"]
                rows.append(
                    {
                        "kappa_A_B": cohen_kappa_safe(encoded["A"], encoded["B"]),
                        "kappa_A_C": cohen_kappa_safe(encoded["A"], encoded["C"]),
                        "kappa_B_C": cohen_kappa_safe(encoded["B"], encoded["C"]),
                        "low_base_rate_flag": bool(low_base_rate_flag(n_positive_total)),
                    }
                )
            continue

        if field in TRISTATE_FIELDS:
            encoded = {}
            positives = {"A": 0, "B": 0, "C": 0}
            for variant in ("A", "B", "C"):
                values = [
                    str(parsed_by_variant[variant][hadm_id].get(field, "not_documented"))
                    for hadm_id in hadm_ids
                ]
                encoded[variant] = [encode_tristate(value) for value in values]
                positives[variant] = count_positive_tristate(values)
            n_positive_total = positives["A"] + positives["B"] + positives["C"]
            rows.append(
                {
                    "kappa_A_B": cohen_kappa_safe(encoded["A"], encoded["B"]),
                    "kappa_A_C": cohen_kappa_safe(encoded["A"], encoded["C"]),
                    "kappa_B_C": cohen_kappa_safe(encoded["B"], encoded["C"]),
                    "low_base_rate_flag": bool(low_base_rate_flag(n_positive_total)),
                }
            )
            continue

        if field in ENUM_FIELDS:
            encoded = {}
            positives = {"A": 0, "B": 0, "C": 0}
            observed = sorted(
                {
                    str(parsed_by_variant[variant][hadm_id].get(field))
                    for variant in ("A", "B", "C")
                    for hadm_id in hadm_ids
                }
            )
            mapping = {value: idx for idx, value in enumerate(observed)}
            for variant in ("A", "B", "C"):
                values = [
                    str(parsed_by_variant[variant][hadm_id].get(field))
                    for hadm_id in hadm_ids
                ]
                encoded[variant] = [mapping[value] for value in values]
                positives[variant] = count_positive_enum(values)
            n_positive_total = positives["A"] + positives["B"] + positives["C"]
            rows.append(
                {
                    "kappa_A_B": cohen_kappa_safe(encoded["A"], encoded["B"]),
                    "kappa_A_C": cohen_kappa_safe(encoded["A"], encoded["C"]),
                    "kappa_B_C": cohen_kappa_safe(encoded["B"], encoded["C"]),
                    "low_base_rate_flag": bool(low_base_rate_flag(n_positive_total)),
                }
            )
            continue

        if field == "dominant_admission_reason":
            mapping = {tag: idx for idx, tag in enumerate(ADMISSION_REASON_TAGS)}
            fallback = mapping["other"]
            encoded = {}
            for variant in ("A", "B", "C"):
                values = [
                    str(parsed_by_variant[variant][hadm_id].get(field, "other"))
                    for hadm_id in hadm_ids
                ]
                encoded[variant] = [mapping.get(value, fallback) for value in values]
            rows.append(
                {
                    "kappa_A_B": cohen_kappa_safe(encoded["A"], encoded["B"]),
                    "kappa_A_C": cohen_kappa_safe(encoded["A"], encoded["C"]),
                    "kappa_B_C": cohen_kappa_safe(encoded["B"], encoded["C"]),
                    "low_base_rate_flag": False,
                }
            )

    filtered = [row for row in rows if not bool(row["low_base_rate_flag"])]
    if not filtered:
        raise ValueError(f"No filtered fields remained for sample '{sample_key}'")

    vals_a = [0.5 * (float(row["kappa_A_B"]) + float(row["kappa_A_C"])) for row in filtered]
    vals_b = [0.5 * (float(row["kappa_A_B"]) + float(row["kappa_B_C"])) for row in filtered]
    vals_c = [0.5 * (float(row["kappa_A_C"]) + float(row["kappa_B_C"])) for row in filtered]
    return {
        "A": float(np.median(np.asarray(vals_a, dtype=np.float64))),
        "B": float(np.median(np.asarray(vals_b, dtype=np.float64))),
        "C": float(np.median(np.asarray(vals_c, dtype=np.float64))),
    }


def recompute_five_sample_stability() -> dict[str, ThreeWaySummary]:
    return {
        key: recompute_filtered_median(key, run_ids, split_csv_name)
        for key, (run_ids, split_csv_name) in SAMPLE_SPECS.items()
    }
