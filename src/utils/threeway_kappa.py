from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from src.schema.vocabulary import ADMISSION_REASON_TAGS

TRISTATE_TO_INT: dict[str, int] = {
    "yes": 1,
    "no": -1,
    "not_documented": 0,
}
INT_TO_TRISTATE: dict[int, str] = {value: key for key, value in TRISTATE_TO_INT.items()}


def encode_tristate(value: str) -> int:
    if value not in TRISTATE_TO_INT:
        raise ValueError(f"Invalid TriState value: {value}")
    return TRISTATE_TO_INT[value]


def decode_tristate(value: int) -> str:
    if value not in INT_TO_TRISTATE:
        raise ValueError(f"Invalid TriState code: {value}")
    return INT_TO_TRISTATE[value]


def encode_admission_reason_tags(tags: Iterable[str]) -> dict[str, int]:
    tag_set = set(tags)
    return {tag: int(tag in tag_set) for tag in ADMISSION_REASON_TAGS}


def cohen_kappa_safe(left: list[int] | np.ndarray, right: list[int] | np.ndarray) -> float:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape:
        raise ValueError("left and right label arrays must have the same shape")
    if left_arr.size == 0:
        return 0.0

    if np.unique(left_arr).size == 1 and np.unique(right_arr).size == 1:
        return 1.0 if np.array_equal(left_arr, right_arr) else 0.0

    score = float(cohen_kappa_score(left_arr, right_arr))
    if np.isnan(score):
        return 1.0 if np.array_equal(left_arr, right_arr) else 0.0
    return score


def percent_agreement(left: list[int] | np.ndarray, right: list[int] | np.ndarray) -> float:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape:
        raise ValueError("left and right label arrays must have the same shape")
    if left_arr.size == 0:
        return 0.0
    return float(np.mean(left_arr == right_arr))


def pabak_score(left: list[int] | np.ndarray, right: list[int] | np.ndarray) -> float:
    agreement = percent_agreement(left, right)
    return (2.0 * agreement) - 1.0


def intersect_successful_hadm_ids(
    successful_by_variant: dict[str, dict[int, object]],
) -> list[int]:
    if not successful_by_variant:
        return []
    key_sets = [set(entries.keys()) for entries in successful_by_variant.values()]
    if not key_sets:
        return []
    return sorted(set.intersection(*key_sets))


def align_values_for_hadm_ids(
    values_by_variant: dict[str, dict[int, object]],
) -> tuple[list[int], dict[str, list[int]]]:
    hadm_ids = intersect_successful_hadm_ids(values_by_variant)
    aligned: dict[str, list[int]] = {}
    for variant, value_map in values_by_variant.items():
        variant_values: list[int] = []
        for hadm_id in hadm_ids:
            value = value_map[hadm_id]
            if not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"Value for variant={variant}, hadm_id={hadm_id} is not an int: {value!r}"
                )
            variant_values.append(int(value))
        aligned[variant] = variant_values
    return hadm_ids, aligned


def count_positive_tristate(values: Iterable[str]) -> int:
    return int(sum(1 for value in values if value == "yes"))


def count_positive_admission_tag(tag_lists: Iterable[Iterable[str]], tag: str) -> int:
    return int(sum(1 for tags in tag_lists if tag in set(tags)))


def count_positive_enum(values: Iterable[str]) -> int:
    return int(sum(1 for value in values if value != "not_documented"))


def low_base_rate_flag(n_positive_total: int, threshold: int = 10) -> bool:
    return n_positive_total < threshold


def file_sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found for SHA-256 calculation: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_split_integrity(
    *,
    split_csv_path: Path,
    manifest_path: Path,
    checksum_key: str | None = None,
) -> tuple[bool, str, str]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = payload.get("checksums_sha256")
    if not isinstance(checksums, dict):
        raise ValueError(
            f"Invalid manifest format in {manifest_path}: missing checksums_sha256 map."
        )

    key = checksum_key or split_csv_path.name
    if key not in checksums:
        raise KeyError(f"Checksum key '{key}' not found in manifest {manifest_path}")

    expected = str(checksums[key])
    observed = file_sha256(split_csv_path)
    return observed == expected, observed, expected
