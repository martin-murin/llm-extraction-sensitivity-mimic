from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schema.fields import LLMNoteFeatures
from src.schema.vocabulary import ADMISSION_REASON_TAGS, ADMISSION_REASON_TAG_DESCRIPTIONS


def _valid_payload() -> dict[str, object]:
    return {
        "admission_reason_tags": ["cardiac_hf", "renal_aki"],
        "dominant_admission_reason": "cardiac_hf",
        "primary_diagnosis_text": "Acute decompensated heart failure",
        "shock_present": "no",
        "infection_as_trigger": "not_documented",
        "aki_present": "yes",
        "functional_status": "assisted",
        "mental_status": "intact",
        "discharge_condition_category": "improved",
        "lives_alone": "yes",
        "social_support_absent": "no",
        "financial_hardship": "not_documented",
        "substance_use_active": "no",
        "fall_risk_documented": "yes",
        "cognitive_impairment": "no",
        "goals_of_care_flag": "yes",
        "palliative_care_consult": "no",
        "dnr_dni_documented": "yes",
        "new_meds_started_count": 3,
        "meds_stopped_count": 1,
        "home_health_ordered": "yes",
        "cardiac_rehab_referred": "no",
        "discharge_delayed_reason": "no",
        "hospital_acquired_complication": "no",
        "unresolved_diagnosis_at_discharge": "no",
        "reasoning": "Findings align with ADHF and concomitant AKI.",
    }


def test_valid_full_input() -> None:
    model = LLMNoteFeatures(**_valid_payload())
    assert model.dominant_admission_reason == "cardiac_hf"


def test_dominant_not_in_tags_rejected() -> None:
    payload = _valid_payload()
    payload["admission_reason_tags"] = ["cardiac_hf"]
    payload["dominant_admission_reason"] = "respiratory_infection"

    with pytest.raises(ValidationError):
        LLMNoteFeatures(**payload)


def test_empty_tags_rejected() -> None:
    payload = _valid_payload()
    payload["admission_reason_tags"] = []

    with pytest.raises(ValidationError):
        LLMNoteFeatures(**payload)


@pytest.mark.parametrize("value", ["yes", "no", "not_documented"])
def test_tristate_accepts_three_values(value: str) -> None:
    payload = _valid_payload()
    payload["shock_present"] = value
    model = LLMNoteFeatures(**payload)
    assert model.shock_present == value


@pytest.mark.parametrize("value", ["maybe", "unknown", "true", "false"])
def test_tristate_rejects_invalid_values(value: str) -> None:
    payload = _valid_payload()
    payload["shock_present"] = value

    with pytest.raises(ValidationError):
        LLMNoteFeatures(**payload)


def test_invalid_admission_reason_tag_rejected() -> None:
    payload = _valid_payload()
    payload["admission_reason_tags"] = ["made_up_tag"]
    payload["dominant_admission_reason"] = "made_up_tag"

    with pytest.raises(ValidationError):
        LLMNoteFeatures(**payload)


@pytest.mark.parametrize("value", [None, 0, 15])
def test_count_field_accepts_none_and_non_negative_int(value: int | None) -> None:
    payload = _valid_payload()
    payload["new_meds_started_count"] = value
    model = LLMNoteFeatures(**payload)
    assert model.new_meds_started_count == value


def test_count_field_rejects_negative_int() -> None:
    payload = _valid_payload()
    payload["new_meds_started_count"] = -1

    with pytest.raises(ValidationError):
        LLMNoteFeatures(**payload)


def test_json_roundtrip() -> None:
    original = LLMNoteFeatures(**_valid_payload())
    dumped = original.model_dump_json()
    restored = LLMNoteFeatures.model_validate_json(dumped)
    assert restored == original


def test_vocabulary_integrity() -> None:
    assert set(ADMISSION_REASON_TAG_DESCRIPTIONS.keys()) == set(ADMISSION_REASON_TAGS)
