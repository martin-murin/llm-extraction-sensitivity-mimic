from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.schema.vocabulary import AdmissionReasonTag


TriState = Literal["yes", "no", "not_documented"]
# Used for clinical booleans where "not documented" is semantically different from "false".
# "yes" = affirmative evidence in note
# "no" = explicit negative evidence in note ("patient denies", "no history of")
# "not_documented" = note does not mention this

FunctionalStatus = Literal[
    "independent",
    "assisted",
    "dependent",
    "not_documented",
]

MentalStatus = Literal[
    "intact",
    "mild_impairment",
    "confused_delirious",
    "not_documented",
]

DischargeCondition = Literal[
    "stable",
    "improved",
    "unchanged",
    "deteriorated",
    "expired",
    "not_documented",
]


class LLMNoteFeatures(BaseModel):
    """Structured features extracted from a single MIMIC-IV discharge note.

    One instance per hadm_id. All fields are discharge-note-derived, not ICD-derived.
    Multi-valued fields use list[Literal[...]] with closed vocabulary.
    Clinical booleans use TriState to distinguish absence-of-evidence from evidence-of-absence.
    """

    # === Admission reason ===
    admission_reason_tags: list[AdmissionReasonTag] = Field(
        ...,
        description=(
            "All admission reason categories that apply. Must be non-empty. "
            "If unclear, use ['other']. Closed vocabulary — pick only from allowed tags."
        ),
    )
    dominant_admission_reason: AdmissionReasonTag = Field(
        ...,
        description=(
            "The single most prominent reason driving this admission. "
            "Must be one of the tags present in admission_reason_tags."
        ),
    )
    primary_diagnosis_text: str = Field(
        ...,
        max_length=300,
        description=(
            "Free-text primary diagnosis as documented in the note, not an ICD code. "
            "Used offline for AHRQ CCS mapping. Keep concise (<300 chars)."
        ),
    )

    # === Cardiac / metabolic clinical flags ===
    shock_present: TriState = Field(
        ...,
        description=(
            "Any form of shock documented during admission (cardiogenic, septic, "
            "hypovolemic, distributive)."
        ),
    )
    infection_as_trigger: TriState = Field(
        ...,
        description=(
            "Infection identified as trigger or precipitant for the admission event, "
            "even if not the primary reason."
        ),
    )
    aki_present: TriState = Field(
        ...,
        description="Acute kidney injury present at any point during admission.",
    )

    # === Functional and cognitive status ===
    functional_status: FunctionalStatus = Field(
        ...,
        description="Baseline or pre-admission functional status per note documentation.",
    )
    mental_status: MentalStatus = Field(
        ...,
        description="Mental status at discharge. If multiple descriptions, use the most recent.",
    )

    # === Discharge condition ===
    discharge_condition_category: DischargeCondition = Field(
        ...,
        description="Overall condition at discharge per note's discharge condition statement.",
    )

    # === Social determinants of health ===
    lives_alone: TriState = Field(
        ...,
        description="Patient lives alone at home per social history.",
    )
    social_support_absent: TriState = Field(
        ...,
        description="Explicit documentation of lack of social support (isolated, no family, etc.).",
    )
    financial_hardship: TriState = Field(
        ...,
        description=(
            "Documented financial hardship, uninsured, cost-related medication nonadherence."
        ),
    )
    substance_use_active: TriState = Field(
        ...,
        description=(
            "Active substance use (alcohol, illicit drugs, tobacco excluded). "
            "Historical use without active = 'no'."
        ),
    )

    # === Risk flags ===
    fall_risk_documented: TriState = Field(..., description="Fall risk explicitly documented.")
    cognitive_impairment: TriState = Field(
        ...,
        description=(
            "Baseline cognitive impairment (dementia, MCI) documented — distinct from delirium."
        ),
    )

    # === Goals of care and palliative ===
    goals_of_care_flag: TriState = Field(
        ...,
        description=(
            "Goals-of-care discussion documented during admission, including phrases like "
            "'comfort-focused', 'discussed prognosis', 'family meeting re: goals'."
        ),
    )
    palliative_care_consult: TriState = Field(
        ...,
        description="Palliative care team formally consulted during this admission.",
    )
    dnr_dni_documented: TriState = Field(
        ...,
        description="DNR, DNI, or DNR/DNI code status documented (not just discussed).",
    )

    # === Medication changes ===
    new_meds_started_count: int | None = Field(
        None,
        ge=0,
        description=(
            "Count of medications newly started during this admission. "
            "None if the note does not contain a discernible medication reconciliation. "
            "Count distinct drugs, not prescriptions."
        ),
    )
    meds_stopped_count: int | None = Field(
        None,
        ge=0,
        description=(
            "Count of medications stopped/discontinued during this admission. "
            "None if indeterminate."
        ),
    )

    # === Discharge disposition and follow-up ===
    home_health_ordered: TriState = Field(
        ...,
        description="Home health services (nursing, PT/OT at home) ordered at discharge.",
    )
    cardiac_rehab_referred: TriState = Field(
        ...,
        description="Referral to cardiac rehabilitation program at discharge.",
    )
    discharge_delayed_reason: TriState = Field(
        ...,
        description=(
            "Discharge was delayed for non-medical reasons (placement, insurance, social). "
            "'yes' only if explicitly documented."
        ),
    )

    # === In-hospital adverse events ===
    hospital_acquired_complication: TriState = Field(
        ...,
        description=(
            "Any hospital-acquired complication documented: HAI, hospital-acquired AKI, "
            "hospital-acquired delirium, fall, pressure ulcer, etc."
        ),
    )
    unresolved_diagnosis_at_discharge: TriState = Field(
        ...,
        description=(
            "Language indicating the diagnosis remained unclear or workup pending at discharge "
            "('etiology unclear', 'workup pending', 'to be followed up as outpatient')."
        ),
    )

    # === Reasoning (optional — used during optimization, not production) ===
    reasoning: str | None = Field(
        None,
        max_length=2000,
        description=(
            "Brief rationale (<2000 chars) quoting relevant note excerpts per field group. "
            "Populated only when the prompt requests it."
        ),
    )

    # === Cross-field validation ===
    @model_validator(mode="after")
    def dominant_must_be_in_tags(self) -> LLMNoteFeatures:
        if self.dominant_admission_reason not in self.admission_reason_tags:
            raise ValueError(
                f"dominant_admission_reason '{self.dominant_admission_reason}' "
                f"must be present in admission_reason_tags {self.admission_reason_tags}"
            )
        return self

    @model_validator(mode="after")
    def admission_reason_tags_nonempty(self) -> LLMNoteFeatures:
        if not self.admission_reason_tags:
            raise ValueError(
                "admission_reason_tags must be non-empty; use ['other'] if truly unknown"
            )
        return self
