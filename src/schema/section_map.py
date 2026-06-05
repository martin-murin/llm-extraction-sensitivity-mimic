from __future__ import annotations

from src.schema.fields import LLMNoteFeatures

FIELD_SECTION_MAP: dict[str, list[str]] = {
    "admission_reason_tags": [
        "History of Present Illness",
        "Chief Complaint",
        "Discharge Diagnosis",
    ],
    "dominant_admission_reason": [
        "History of Present Illness",
        "Chief Complaint",
        "Discharge Diagnosis",
    ],
    "primary_diagnosis_text": ["Discharge Diagnosis", "History of Present Illness"],
    "shock_present": ["Brief Hospital Course"],
    "aki_present": ["Brief Hospital Course"],
    "infection_as_trigger": ["Brief Hospital Course", "History of Present Illness"],
    "hospital_acquired_complication": ["Brief Hospital Course"],
    "unresolved_diagnosis_at_discharge": [
        "Brief Hospital Course",
        "Discharge Diagnosis",
        "Discharge Condition",
    ],
    "functional_status": ["Discharge Condition"],
    "mental_status": ["Discharge Condition"],
    "discharge_condition_category": ["Discharge Condition"],
    "lives_alone": ["Social History", "Past Medical History"],
    "social_support_absent": ["Social History", "Past Medical History"],
    "financial_hardship": ["Social History", "Discharge Instructions"],
    "substance_use_active": ["Social History", "Past Medical History"],
    "fall_risk_documented": [
        "History of Present Illness",
        "Brief Hospital Course",
        "Discharge Instructions",
    ],
    "cognitive_impairment": ["Past Medical History", "Discharge Condition"],
    "goals_of_care_flag": ["Brief Hospital Course", "Discharge Condition"],
    "palliative_care_consult": ["Brief Hospital Course"],
    "dnr_dni_documented": ["Brief Hospital Course", "Discharge Condition"],
    "new_meds_started_count": ["Discharge Medications", "Brief Hospital Course"],
    "meds_stopped_count": ["Discharge Medications", "Brief Hospital Course"],
    "home_health_ordered": ["Discharge Instructions", "Discharge Disposition"],
    "cardiac_rehab_referred": ["Discharge Instructions", "Discharge Disposition"],
    "discharge_delayed_reason": ["Brief Hospital Course"],
}

_MODEL_FIELDS = set(LLMNoteFeatures.model_fields.keys())
_REQUIRED_FIELDS = _MODEL_FIELDS.difference({"reasoning"})
_MAP_FIELDS = set(FIELD_SECTION_MAP.keys())

_MISSING = _REQUIRED_FIELDS.difference(_MAP_FIELDS)
_EXTRA = _MAP_FIELDS.difference(_REQUIRED_FIELDS)

assert not _MISSING, f"FIELD_SECTION_MAP missing required fields: {sorted(_MISSING)}"
assert not _EXTRA, f"FIELD_SECTION_MAP contains unknown fields: {sorted(_EXTRA)}"
