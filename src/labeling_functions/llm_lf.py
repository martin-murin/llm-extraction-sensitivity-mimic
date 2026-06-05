from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.labeling_functions.base import LFInput, LFOutput, Vote
from src.labeling_functions.icd_lf import ICD_LF_SPECS


class FieldType(Enum):
    TRISTATE = "tristate"
    ADMISSION_TAG_MEMBERSHIP = "admission_tag_membership"
    ENUM = "enum"


TRISTATE_FIELDS: tuple[str, ...] = (
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
)

ACTIVE_TRISTATE_FIELDS_FOR_SNORKEL: tuple[str, ...] = (
    "aki_present",
    "dnr_dni_documented",
    "palliative_care_consult",
    "home_health_ordered",
    "cardiac_rehab_referred",
    "substance_use_active",
    "fall_risk_documented",
    "cognitive_impairment",
    "goals_of_care_flag",
)

ICD_ANCHORED_ADMISSION_TAGS: tuple[str, ...] = tuple(
    sorted(
        {
            str(spec["target_value"])
            for spec in ICD_LF_SPECS
            if str(spec.get("target_field")) == "admission_reason_tags"
            and spec.get("target_value") is not None
        }
    )
)

SNORKEL_TARGET_FIELD_VALUE_PAIRS: list[tuple[str, str, FieldType]] = [
    *[
        ("admission_reason_tags", tag, FieldType.ADMISSION_TAG_MEMBERSHIP)
        for tag in ICD_ANCHORED_ADMISSION_TAGS
    ],
    *[
        (field_name, target_value, FieldType.TRISTATE)
        for field_name in ACTIVE_TRISTATE_FIELDS_FOR_SNORKEL
        for target_value in ("yes", "no")
    ],
]


@dataclass
class LLMLabelingFunction:
    name: str
    target_field: str
    target_value: str | None
    variant: str
    field_type: FieldType

    def __call__(self, inputs: LFInput) -> LFOutput:
        extraction_map = inputs.llm_extraction_by_variant
        if extraction_map is None:
            return LFOutput(vote=Vote.ABSTAIN, evidence="missing_llm_extractions")

        extraction = extraction_map.get(self.variant)
        if extraction is None:
            return LFOutput(vote=Vote.ABSTAIN, evidence=f"missing_variant:{self.variant}")

        if self.field_type == FieldType.TRISTATE:
            if self.target_value not in {"yes", "no"}:
                return LFOutput(vote=Vote.ABSTAIN, evidence="invalid_target_value")

            current_value = getattr(extraction, self.target_field, None)
            if current_value is None or current_value == "not_documented":
                return LFOutput(vote=Vote.ABSTAIN)
            if current_value == self.target_value:
                return LFOutput(vote=Vote.POSITIVE)
            if current_value in {"yes", "no"}:
                return LFOutput(vote=Vote.NEGATIVE)
            return LFOutput(vote=Vote.ABSTAIN)

        if self.field_type == FieldType.ADMISSION_TAG_MEMBERSHIP:
            if self.target_field == "admission_reason_tags":
                tags = list(getattr(extraction, "admission_reason_tags", []))
                if self.target_value is not None and self.target_value in tags:
                    return LFOutput(vote=Vote.POSITIVE)
                return LFOutput(vote=Vote.ABSTAIN)

            if self.target_field == "dominant_admission_reason":
                dominant = getattr(extraction, "dominant_admission_reason", None)
                if dominant == self.target_value:
                    return LFOutput(vote=Vote.POSITIVE)
                return LFOutput(vote=Vote.ABSTAIN)

            return LFOutput(vote=Vote.ABSTAIN, evidence="unsupported_membership_field")

        if self.field_type == FieldType.ENUM:
            current_value = getattr(extraction, self.target_field, None)
            if current_value is None or current_value == "not_documented":
                return LFOutput(vote=Vote.ABSTAIN)
            if current_value == self.target_value:
                return LFOutput(vote=Vote.POSITIVE)
            return LFOutput(vote=Vote.NEGATIVE)


def build_llm_lf(
    target_field: str,
    target_value: str,
    variant: str,
    field_type: FieldType,
) -> LLMLabelingFunction:
    return LLMLabelingFunction(
        name=f"llm_{variant}_{target_field}_{target_value}",
        target_field=target_field,
        target_value=target_value,
        variant=variant,
        field_type=field_type,
    )


def build_all_llm_lfs(
    variants: list[str],
    target_field_value_pairs: list[tuple[str, str, FieldType]],
) -> list[LLMLabelingFunction]:
    lfs: list[LLMLabelingFunction] = []
    for variant in variants:
        for target_field, target_value, field_type in target_field_value_pairs:
            lfs.append(
                build_llm_lf(
                    target_field=target_field,
                    target_value=target_value,
                    variant=variant,
                    field_type=field_type,
                )
            )
    return lfs
