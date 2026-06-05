from __future__ import annotations

from typing import Literal, cast, get_args


AdmissionReasonTag = Literal[
    # Cardiac
    "cardiac_hf",
    "cardiac_acs",
    "cardiac_arrhythmia",
    "cardiac_htn_emergency",
    "cardiac_valve_disease",
    "cardiac_other",
    # Respiratory
    "respiratory_infection",
    "respiratory_copd_asthma_exacerbation",
    "respiratory_pe_dvt",
    "respiratory_failure_other",
    # GI / Hepatic
    "gi_bleed",
    "gi_obstruction_ileus",
    "gi_pancreatitis",
    "gi_ibd_colitis",
    "hepatic_failure_cirrhosis",
    "gi_other",
    # Renal / GU
    "renal_aki",
    "renal_ckd_esrd_crisis",
    "gu_infection",
    "gu_other",
    # Infection / Sepsis
    "sepsis_bacteremia",
    "infection_skin_soft_tissue",
    "infection_cns",
    "infection_other",
    # Neuro
    "neuro_stroke_tia",
    "neuro_seizure",
    "neuro_altered_mental_status",
    "neuro_other",
    # Metabolic / Endocrine
    "metabolic_dka_hhs",
    "metabolic_electrolyte_crisis",
    "endocrine_other",
    # Heme / Onc
    "heme_anemia_bleed",
    "heme_onc_complication",
    "oncology_elective_treatment",
    # MSK / Trauma
    "trauma_fracture",
    "trauma_other",
    "msk_non_trauma",
    # Psych / Substance
    "psych_mood_anxiety",
    "psych_psychosis_crisis",
    "substance_intoxication_withdrawal",
    "substance_overdose",
    # Symptom workup (rule-out admissions)
    "symptom_workup_chest_pain",
    "symptom_workup_syncope",
    "symptom_workup_other",
    # Elective / OB / Other
    "elective_procedure_non_oncology",
    "obstetric",
    "other",
]

ADMISSION_REASON_TAGS: tuple[str, ...] = cast(tuple[str, ...], get_args(AdmissionReasonTag))

ADMISSION_REASON_TAG_DESCRIPTIONS: dict[str, str] = {
    "cardiac_hf": "Acute decompensated heart failure or cardiogenic pulmonary edema",
    "cardiac_acs": "Acute coronary syndrome (STEMI, NSTEMI, unstable angina)",
    "cardiac_arrhythmia": (
        "New or worsening arrhythmia as primary driver (AFib with RVR, VT, "
        "symptomatic bradyarrhythmia)"
    ),
    "cardiac_htn_emergency": "Hypertensive emergency or urgency with end-organ involvement",
    "cardiac_valve_disease": "Symptomatic valvular disease admission (e.g., severe AS, acute MR)",
    "cardiac_other": (
        "Other cardiac reason not matching the above (pericarditis, myocarditis, etc.)"
    ),
    "respiratory_infection": "Community or hospital acquired pneumonia, bronchitis, etc.",
    "respiratory_copd_asthma_exacerbation": "COPD or asthma exacerbation",
    "respiratory_pe_dvt": "Pulmonary embolism or DVT",
    "respiratory_failure_other": (
        "Hypoxemic or hypercapnic respiratory failure without infectious or COPD/asthma driver"
    ),
    "gi_bleed": "Upper or lower GI bleed",
    "gi_obstruction_ileus": "Small or large bowel obstruction, ileus, volvulus",
    "gi_pancreatitis": "Acute pancreatitis",
    "gi_ibd_colitis": "IBD flare, C. diff or other colitis",
    "hepatic_failure_cirrhosis": (
        "Acute or chronic liver failure, hepatic encephalopathy, cirrhosis decompensation"
    ),
    "gi_other": "Other GI reason (gastritis, hernia, etc.)",
    "renal_aki": "Acute kidney injury as primary admission driver",
    "renal_ckd_esrd_crisis": (
        "CKD/ESRD complication driving admission (uremia, fluid overload, missed dialysis)"
    ),
    "gu_infection": "UTI, pyelonephritis, prostatitis",
    "gu_other": "Other genitourinary reason",
    "sepsis_bacteremia": (
        "Sepsis or bacteremia as the primary admission reason regardless of source"
    ),
    "infection_skin_soft_tissue": "Cellulitis, abscess, necrotizing fasciitis",
    "infection_cns": "Meningitis, encephalitis, brain abscess",
    "infection_other": "Other infection (endocarditis, osteomyelitis, fungemia, etc.)",
    "neuro_stroke_tia": "Ischemic or hemorrhagic stroke, TIA",
    "neuro_seizure": "Seizure or status epilepticus",
    "neuro_altered_mental_status": "Encephalopathy or delirium as primary reason",
    "neuro_other": "Other neurologic reason (Parkinson, MS, neuromuscular, etc.)",
    "metabolic_dka_hhs": "Diabetic ketoacidosis or hyperosmolar hyperglycemic state",
    "metabolic_electrolyte_crisis": (
        "Severe electrolyte disturbance (hyperkalemia, hyponatremia crisis, etc.)"
    ),
    "endocrine_other": "Other endocrine reason (thyroid storm, adrenal crisis, etc.)",
    "heme_anemia_bleed": "Severe anemia or hemorrhage not clearly GI or trauma",
    "heme_onc_complication": (
        "Complication of cancer or its treatment (neutropenic fever, tumor lysis, etc.)"
    ),
    "oncology_elective_treatment": (
        "Planned chemotherapy, transplant, or oncology procedure admission"
    ),
    "trauma_fracture": "Fracture from trauma",
    "trauma_other": "Other trauma (blunt, penetrating, falls without fracture)",
    "msk_non_trauma": (
        "MSK admission without trauma (joint infection, non-traumatic back pain workup, etc.)"
    ),
    "psych_mood_anxiety": "Depression, anxiety, bipolar mood episode (non-psychotic)",
    "psych_psychosis_crisis": (
        "Psychotic episode, suicidal ideation with plan, acute psychiatric crisis"
    ),
    "substance_intoxication_withdrawal": "Alcohol or drug intoxication or withdrawal",
    "substance_overdose": "Intentional or unintentional overdose requiring admission",
    "symptom_workup_chest_pain": (
        "Chest pain admission for rule-out, workup inconclusive or ruled out"
    ),
    "symptom_workup_syncope": "Syncope admission for workup",
    "symptom_workup_other": (
        "Other symptom-based admission for workup without definitive diagnosis at discharge"
    ),
    "elective_procedure_non_oncology": (
        "Planned non-oncology procedure (elective surgery, cardiac cath, etc.)"
    ),
    "obstetric": "Pregnancy, labor, postpartum complications",
    "other": "Reason does not fit any of the above categories",
}

CHAPTER_TO_PLAUSIBLE_TAGS: dict[str, set[str]] = {
    "I. Infectious": {
        "sepsis_bacteremia",
        "infection_skin_soft_tissue",
        "infection_cns",
        "infection_other",
    },
    "II. Neoplasms": {
        "oncology_elective_treatment",
        "heme_onc_complication",
    },
    "IV. Endocrine/metabolic": {
        "metabolic_dka_hhs",
        "metabolic_electrolyte_crisis",
        "endocrine_other",
    },
    "V. Mental": {
        "psych_mood_anxiety",
        "psych_psychosis_crisis",
        "substance_intoxication_withdrawal",
        "substance_overdose",
    },
    "IX. Circulatory": {
        "cardiac_hf",
        "cardiac_acs",
        "cardiac_arrhythmia",
        "cardiac_htn_emergency",
        "cardiac_valve_disease",
        "cardiac_other",
        "symptom_workup_chest_pain",
        "symptom_workup_syncope",
    },
    "X. Respiratory": {
        "respiratory_infection",
        "respiratory_copd_asthma_exacerbation",
        "respiratory_pe_dvt",
        "respiratory_failure_other",
    },
    "XI. Digestive": {
        "gi_bleed",
        "gi_obstruction_ileus",
        "gi_pancreatitis",
        "gi_ibd_colitis",
        "hepatic_failure_cirrhosis",
        "gi_other",
    },
    "XIV. Genitourinary": {
        "gu_infection",
        "gu_other",
        "renal_aki",
        "renal_ckd_esrd_crisis",
    },
    "XV. Pregnancy": {
        "obstetric",
    },
    "XVIII. Symptoms/signs": {
        "symptom_workup_chest_pain",
        "symptom_workup_syncope",
        "symptom_workup_other",
    },
    "XIX. Injury/poisoning": {
        "trauma_fracture",
        "trauma_other",
        "substance_overdose",
    },
}

assert set(ADMISSION_REASON_TAG_DESCRIPTIONS.keys()) == set(ADMISSION_REASON_TAGS), (
    "ADMISSION_REASON_TAG_DESCRIPTIONS keys must exactly match ADMISSION_REASON_TAGS"
)

for chapter_name, tags in CHAPTER_TO_PLAUSIBLE_TAGS.items():
    unknown_tags = tags.difference(ADMISSION_REASON_TAGS)
    assert not unknown_tags, f"{chapter_name} contains unknown tags: {sorted(unknown_tags)}"
