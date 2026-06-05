from __future__ import annotations

from src.labeling_functions.base import LFInput, Vote
from src.labeling_functions.llm_lf import FieldType, build_llm_lf
from src.schema.fields import LLMNoteFeatures


def _valid_features(**overrides: object) -> LLMNoteFeatures:
    payload: dict[str, object] = {
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
        "reasoning": "synthetic",
    }
    payload.update(overrides)
    return LLMNoteFeatures(**payload)


def test_llm_lf_abstains_when_variant_payload_missing() -> None:
    lf = build_llm_lf(
        target_field="aki_present",
        target_value="yes",
        variant="a",
        field_type=FieldType.TRISTATE,
    )

    out_none = lf(LFInput(hadm_id=1, note_text=""))
    assert out_none.vote == Vote.ABSTAIN

    out_missing_variant = lf(
        LFInput(
            hadm_id=1,
            note_text="",
            llm_extraction_by_variant={"b": _valid_features()},
        )
    )
    assert out_missing_variant.vote == Vote.ABSTAIN


def test_tristate_yes_lf_behavior() -> None:
    lf = build_llm_lf(
        target_field="aki_present",
        target_value="yes",
        variant="a",
        field_type=FieldType.TRISTATE,
    )

    out_yes = lf(
        LFInput(
            hadm_id=1,
            note_text="",
            llm_extraction_by_variant={"a": _valid_features(aki_present="yes")},
        )
    )
    out_no = lf(
        LFInput(
            hadm_id=2,
            note_text="",
            llm_extraction_by_variant={"a": _valid_features(aki_present="no")},
        )
    )
    out_nd = lf(
        LFInput(
            hadm_id=3,
            note_text="",
            llm_extraction_by_variant={"a": _valid_features(aki_present="not_documented")},
        )
    )

    assert out_yes.vote == Vote.POSITIVE
    assert out_no.vote == Vote.NEGATIVE
    assert out_nd.vote == Vote.ABSTAIN


def test_tristate_no_lf_behavior() -> None:
    lf = build_llm_lf(
        target_field="aki_present",
        target_value="no",
        variant="a",
        field_type=FieldType.TRISTATE,
    )

    out_yes = lf(
        LFInput(
            hadm_id=1,
            note_text="",
            llm_extraction_by_variant={"a": _valid_features(aki_present="yes")},
        )
    )
    out_no = lf(
        LFInput(
            hadm_id=2,
            note_text="",
            llm_extraction_by_variant={"a": _valid_features(aki_present="no")},
        )
    )
    out_nd = lf(
        LFInput(
            hadm_id=3,
            note_text="",
            llm_extraction_by_variant={"a": _valid_features(aki_present="not_documented")},
        )
    )

    assert out_yes.vote == Vote.NEGATIVE
    assert out_no.vote == Vote.POSITIVE
    assert out_nd.vote == Vote.ABSTAIN


def test_admission_tag_membership_lf_behavior() -> None:
    lf = build_llm_lf(
        target_field="admission_reason_tags",
        target_value="cardiac_hf",
        variant="a",
        field_type=FieldType.ADMISSION_TAG_MEMBERSHIP,
    )

    out_in = lf(
        LFInput(
            hadm_id=1,
            note_text="",
            llm_extraction_by_variant={
                "a": _valid_features(admission_reason_tags=["cardiac_hf", "renal_aki"])
            },
        )
    )
    out_not_in = lf(
        LFInput(
            hadm_id=2,
            note_text="",
            llm_extraction_by_variant={
                "a": _valid_features(
                    admission_reason_tags=["renal_aki"],
                    dominant_admission_reason="renal_aki",
                )
            },
        )
    )

    assert out_in.vote == Vote.POSITIVE
    assert out_not_in.vote == Vote.ABSTAIN


def test_enum_lf_behavior() -> None:
    lf = build_llm_lf(
        target_field="mental_status",
        target_value="intact",
        variant="a",
        field_type=FieldType.ENUM,
    )

    out_match = lf(
        LFInput(
            hadm_id=1,
            note_text="",
            llm_extraction_by_variant={"a": _valid_features(mental_status="intact")},
        )
    )
    out_other = lf(
        LFInput(
            hadm_id=2,
            note_text="",
            llm_extraction_by_variant={"a": _valid_features(mental_status="mild_impairment")},
        )
    )
    out_nd = lf(
        LFInput(
            hadm_id=3,
            note_text="",
            llm_extraction_by_variant={"a": _valid_features(mental_status="not_documented")},
        )
    )

    assert out_match.vote == Vote.POSITIVE
    assert out_other.vote == Vote.NEGATIVE
    assert out_nd.vote == Vote.ABSTAIN
