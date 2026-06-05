from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.schema.vocabulary import ADMISSION_REASON_TAGS
from src.utils.threeway_kappa import (
    cohen_kappa_safe,
    count_positive_admission_tag,
    count_positive_tristate,
    decode_tristate,
    encode_admission_reason_tags,
    encode_tristate,
    file_sha256,
    intersect_successful_hadm_ids,
    low_base_rate_flag,
    pabak_score,
    verify_split_integrity,
)


def test_tristate_encoding_is_deterministic_and_reversible() -> None:
    values = ["yes", "no", "not_documented"]
    for value in values:
        encoded = encode_tristate(value)
        decoded = decode_tristate(encoded)
        assert decoded == value


def test_admission_reason_tags_binary_encoding_preserves_tag_membership() -> None:
    tags = ["cardiac_hf", "renal_aki", "other"]
    encoded = encode_admission_reason_tags(tags)
    assert set(encoded.keys()) == set(ADMISSION_REASON_TAGS)
    assert encoded["cardiac_hf"] == 1
    assert encoded["renal_aki"] == 1
    assert encoded["other"] == 1
    assert encoded["neuro_stroke_tia"] == 0


def test_kappa_identical_arrays_is_one() -> None:
    labels = np.array([1, 0, 1, -1, 0, 1], dtype=np.int64)
    assert cohen_kappa_safe(labels, labels.copy()) == 1.0


def test_kappa_random_arrays_is_near_zero() -> None:
    rng = np.random.default_rng(42)
    left = rng.integers(0, 3, size=500, endpoint=False)
    right = rng.integers(0, 3, size=500, endpoint=False)
    score = cohen_kappa_safe(left, right)
    assert abs(score) < 0.2


def test_intersection_excludes_notes_missing_in_any_variant() -> None:
    successful = {
        "A": {1: {"x": 1}, 2: {"x": 2}, 3: {"x": 3}},
        "B": {1: {"x": 1}, 3: {"x": 3}},
        "C": {1: {"x": 1}, 2: {"x": 2}, 3: {"x": 3}},
    }
    assert intersect_successful_hadm_ids(successful) == [1, 3]


def test_n_positive_total_counts_yes_for_tristate_only() -> None:
    a_values = ["yes", "no", "not_documented", "yes"]
    b_values = ["no", "not_documented", "yes", "no"]
    c_values = ["not_documented", "yes", "no", "yes"]
    total = (
        count_positive_tristate(a_values)
        + count_positive_tristate(b_values)
        + count_positive_tristate(c_values)
    )
    assert total == 5


def test_n_positive_total_counts_admission_tag_presence() -> None:
    tag = "cardiac_hf"
    a_tags = [["cardiac_hf"], ["renal_aki"], []]
    b_tags = [["cardiac_hf"], ["cardiac_hf", "other"], ["other"]]
    c_tags = [[], ["cardiac_hf"], []]
    total = (
        count_positive_admission_tag(a_tags, tag)
        + count_positive_admission_tag(b_tags, tag)
        + count_positive_admission_tag(c_tags, tag)
    )
    assert total == 4


def test_pabak_identical_arrays_returns_one() -> None:
    left = np.array([1, 0, -1, 1, 0], dtype=np.int64)
    right = np.array([1, 0, -1, 1, 0], dtype=np.int64)
    assert pabak_score(left, right) == 1.0


def test_pabak_perfect_anti_correlation_returns_minus_one() -> None:
    left = np.array([1, 1, 0, 0], dtype=np.int64)
    right = np.array([0, 0, 1, 1], dtype=np.int64)
    assert pabak_score(left, right) == -1.0


def test_pabak_90_percent_agreement_returns_point_eight() -> None:
    left = np.array([1] * 10, dtype=np.int64)
    right = np.array([1] * 9 + [0], dtype=np.int64)
    assert pabak_score(left, right) == 0.8


def test_low_base_rate_flag_threshold_behavior() -> None:
    assert low_base_rate_flag(9) is True
    assert low_base_rate_flag(10) is False


def test_holdout_integrity_check_hash_match(tmp_path: Path) -> None:
    holdout_path = tmp_path / "holdout_150.csv"
    holdout_path.write_text("hadm_id\n1\n2\n", encoding="utf-8")
    checksum = file_sha256(holdout_path)

    manifest_path = tmp_path / "SPLITS_MANIFEST.json"
    manifest_path.write_text(
        json.dumps({"checksums_sha256": {"holdout_150.csv": checksum}}, ensure_ascii=True),
        encoding="utf-8",
    )

    matched, observed, expected = verify_split_integrity(
        split_csv_path=holdout_path,
        manifest_path=manifest_path,
    )
    assert matched is True
    assert observed == checksum
    assert expected == checksum


def test_holdout_integrity_check_hash_mismatch(tmp_path: Path) -> None:
    holdout_path = tmp_path / "holdout_150.csv"
    holdout_path.write_text("hadm_id\n1\n2\n", encoding="utf-8")

    manifest_path = tmp_path / "SPLITS_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {"checksums_sha256": {"holdout_150.csv": "0" * 64}},
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    matched, observed, expected = verify_split_integrity(
        split_csv_path=holdout_path,
        manifest_path=manifest_path,
    )
    assert matched is False
    assert observed != expected
