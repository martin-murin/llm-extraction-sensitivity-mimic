from __future__ import annotations

from src.labeling_functions.base import LFInput, Vote
from src.labeling_functions.icd_lf import ICD_LF_SPECS, build_all_icd_lfs, build_icd_lf


def test_icd_lf_abstains_when_no_match() -> None:
    lf = build_icd_lf(
        {
            "name": "test_hf",
            "target_field": "admission_reason_tags",
            "target_value": "cardiac_hf",
            "match_position": "primary_only",
            "icd10_prefixes": ["I50"],
            "icd9_prefixes": ["428"],
        }
    )
    output = lf(
        LFInput(
            hadm_id=1,
            note_text="note",
            icd_codes=[("J18.9", 10)],
            primary_icd_code="J18.9",
            primary_icd_version=10,
        )
    )
    assert output.vote == Vote.ABSTAIN


def test_icd_lf_positive_on_matching_primary_code() -> None:
    lf = build_icd_lf(
        {
            "name": "test_hf",
            "target_field": "admission_reason_tags",
            "target_value": "cardiac_hf",
            "match_position": "primary_only",
            "icd10_prefixes": ["I50"],
            "icd9_prefixes": ["428"],
        }
    )
    output = lf(
        LFInput(
            hadm_id=2,
            note_text="note",
            icd_codes=[("I50.23", 10)],
            primary_icd_code="I50.23",
            primary_icd_version=10,
        )
    )
    assert output.vote == Vote.POSITIVE


def test_primary_only_does_not_fire_for_non_primary_position_match() -> None:
    lf = build_icd_lf(
        {
            "name": "test_hf",
            "target_field": "admission_reason_tags",
            "target_value": "cardiac_hf",
            "match_position": "primary_only",
            "icd10_prefixes": ["I50"],
            "icd9_prefixes": ["428"],
        }
    )
    output = lf(
        LFInput(
            hadm_id=3,
            note_text="note",
            icd_codes=[("J18.9", 10), ("I50.9", 10)],
            primary_icd_code="J18.9",
            primary_icd_version=10,
        )
    )
    assert output.vote == Vote.ABSTAIN


def test_any_position_fires_when_non_primary_position_matches() -> None:
    lf = build_icd_lf(
        {
            "name": "test_sepsis_any",
            "target_field": "admission_reason_tags",
            "target_value": "sepsis_bacteremia",
            "match_position": "any",
            "icd10_prefixes": ["A41"],
            "icd9_prefixes": ["038"],
        }
    )
    output = lf(
        LFInput(
            hadm_id=4,
            note_text="note",
            icd_codes=[("J18.9", 10), ("A41.9", 10)],
            primary_icd_code="J18.9",
            primary_icd_version=10,
        )
    )
    assert output.vote == Vote.POSITIVE


def test_icd9_and_icd10_version_specific_matching() -> None:
    lf = build_icd_lf(
        {
            "name": "test_hf_versions",
            "target_field": "admission_reason_tags",
            "target_value": "cardiac_hf",
            "match_position": "any",
            "icd10_prefixes": ["I50"],
            "icd9_prefixes": ["428"],
        }
    )
    output_icd10 = lf(
        LFInput(
            hadm_id=5,
            note_text="note",
            icd_codes=[("I50.22", 10)],
            primary_icd_code="I50.22",
            primary_icd_version=10,
        )
    )
    output_icd9 = lf(
        LFInput(
            hadm_id=6,
            note_text="note",
            icd_codes=[("428.0", 9)],
            primary_icd_code="428.0",
            primary_icd_version=9,
        )
    )
    output_wrong_version = lf(
        LFInput(
            hadm_id=7,
            note_text="note",
            icd_codes=[("I50.22", 9)],
            primary_icd_code="I50.22",
            primary_icd_version=9,
        )
    )

    assert output_icd10.vote == Vote.POSITIVE
    assert output_icd9.vote == Vote.POSITIVE
    assert output_wrong_version.vote == Vote.ABSTAIN


def test_build_all_icd_lfs_count_and_unique_names() -> None:
    lfs = build_all_icd_lfs()
    assert len(lfs) == len(ICD_LF_SPECS)
    names = [lf.name for lf in lfs]
    assert len(names) == len(set(names))
