# Definitions and Source Tables

This section collects the sample-design, schema, and labeling-function definitions referenced by the Methods and Results.

## Sample Design

| Sample name           | $N$    | Purpose                                                                              | Referenced in section                                            |
| --------------------- | -----: | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| Smoke                 | 200    | Initial prompt drafting and smoke validation before three-variant extraction         | Methods (three prompt variants)                                  |
| Refinement            | 150    | Prompt refinement and disagreement-audit development set                             | Methods (splits), refinement-to-holdout generalization; sample-size stability                               |
| Holdout               | 150    | Firewalled single-touch validation set                                               | Methods (holdout), refinement-to-holdout generalization; sample-size stability                              |
| Methodology           | 1,000  | Variant comparison and methodology validation sample                                 | Methods/Results, refinement-to-holdout generalization; sample-size stability                                |
| Methodology 5k Audit  | 500    | Pre-production audit subset for tri-variant diagnostics                              | Methods/Results, disagreement decomposition; sample-size stability                                |
| Methodology Paired    | 1,500  | Same-note paired small-vs-full model-size analyses                                  | Results model-size sections, Figure~\ref{fig:cross-prompt-model-size}/Figure~\ref{fig:per-field-deltas} and Figure~\ref{fig:per-variant-cross-model}, Figure~\ref{fig:cross-model-a}--Figure~\ref{fig:cross-model-c} |
| Extended              | 5,000  | Large post-lock tri-variant validation sample                                        | Results stability and confusion analyses, Figure~\ref{fig:per-pair-kappa}/Figure~\ref{fig:tag-prevalence}/Figure~\ref{fig:admission-confusion} and Figure~\ref{fig:cross-variant-ab}--Figure~\ref{fig:cross-variant-bc}, Figure~\ref{fig:sample-size-stability} |
| Pooled cross-variant  | {{cross_variant_pooled_n}}  | Intersection-pooled A/B/C sample (1k + 500 + extended) for cross-variant diagnostics | Figure~\ref{fig:per-pair-kappa}/Figure~\ref{fig:tag-prevalence}/Figure~\ref{fig:admission-confusion}, Figure~\ref{fig:lf-icd-concordance}--Figure~\ref{fig:aki-five-signal}, Figure~\ref{fig:cross-variant-ab}--Figure~\ref{fig:cross-variant-bc}, Figure~\ref{fig:disagreement-decomposition}, Figure~\ref{fig:enum-mental-status}--Figure~\ref{fig:enum-discharge-condition} |
| Production            | 331,793| Population-scale extraction cohort                                                   | Methods production section                                     |

Table: \label{tbl:splits} List of study samples, fixed sizes, and methodological purposes for all main and supplement analyses.

\clearpage

## Admission-Reason Tag Vocabulary

| Tag ID | Definition | Anchor codes or patterns |
| ------ | ---------- | ------------------------ |
| `cardiac_hf` | Acute decompensated heart failure or cardiogenic pulmonary edema | ICD-10-CM: I50, I11.0, I13.0, I13.2; ICD-9-CM: 428, 402.01, 402.11, 402.91, 404.01, 404.03, 404.11, 404.13, 404.91, 404.93 |
| `cardiac_acs` | Acute coronary syndrome (STEMI, NSTEMI, unstable angina) | ICD-10-CM: I21, I22; ICD-9-CM: 410, 411 |
| `cardiac_arrhythmia` | New or worsening arrhythmia as primary driver (AFib with RVR, VT, symptomatic bradyarrhythmia) | ICD-10-CM: I47, I48, I49; ICD-9-CM: 427 |
| `cardiac_htn_emergency` | Hypertensive emergency or urgency with end-organ involvement | --- |
| `cardiac_valve_disease` | Symptomatic valvular disease admission (e.g., severe AS, acute MR) | --- |
| `cardiac_other` | Other cardiac reason not matching the above (pericarditis, myocarditis, etc.) | --- |
| `respiratory_infection` | Community or hospital acquired pneumonia, bronchitis, etc. | ICD-10-CM: J12, J13, J14, J15, J16, J17, J18, J20, J21, J22; ICD-9-CM: 480, 481, 482, 483, 484, 485, 486, 487 |
| \texttt{respiratory\_copd\_asthma\_\newline exacerbation} | COPD or asthma exacerbation | ICD-10-CM: J44, J45, J46; ICD-9-CM: 491, 492, 493, 496 |
| `respiratory_pe_dvt` | Pulmonary embolism or DVT | ICD-10-CM: I26, I82; ICD-9-CM: 415.1, 451, 453 |
| `respiratory_failure_other` | Hypoxemic or hypercapnic respiratory failure without infectious or COPD/asthma driver | --- |
| `gi_bleed` | Upper or lower GI bleed | ICD-10-CM: K25.0, K25.2, K25.4, K25.6, K26.0, K26.2, K26.4, K26.6, K27.0, K27.2, K27.4, K27.6, K28.0, K28.2, K28.4, K28.6, K29.01, K29.21, K29.31, K29.41, K29.51, K29.61, K29.71, K29.81, K29.91, K62.5, K92.0, K92.1, K92.2; ICD-9-CM: 530.82, 531.00, 531.01, 531.20, 531.21, 531.40, 531.41, 531.60, 531.61, 532, 533, 534, 535.01, 535.11, 535.21, 535.31, 535.41, 535.51, 535.61, 535.71, 569.3, 578 |
| `gi_obstruction_ileus` | Small or large bowel obstruction, ileus, volvulus | --- |
| `gi_pancreatitis` | Acute pancreatitis | --- |
| `gi_ibd_colitis` | IBD flare, C. diff or other colitis | --- |
| `hepatic_failure_cirrhosis` | Acute or chronic liver failure, hepatic encephalopathy, cirrhosis decompensation | ICD-10-CM: K70, K71, K72, K73, K74, K76.6, K76.7; ICD-9-CM: 571, 572, 573 |
| `gi_other` | Other GI reason (gastritis, hernia, etc.) | --- |
| `renal_aki` | Acute kidney injury as primary admission driver | --- |
| `renal_ckd_esrd_crisis` | CKD/ESRD complication driving admission (uremia, fluid overload, missed dialysis) | --- |
| `gu_infection` | UTI, pyelonephritis, prostatitis | --- |
| `gu_other` | Other genitourinary reason | --- |
| `sepsis_bacteremia` | Sepsis or bacteremia as the primary admission reason regardless of source | ICD-10-CM: A40, A41, R65.2; ICD-9-CM: 038, 995.91, 995.92, 790.7 |
| `infection_skin_soft_tissue` | Cellulitis, abscess, necrotizing fasciitis | --- |
| `infection_cns` | Meningitis, encephalitis, brain abscess | --- |
| `infection_other` | Other infection (endocarditis, osteomyelitis, fungemia, etc.) | --- |
| `neuro_stroke_tia` | Ischemic or hemorrhagic stroke, TIA | ICD-10-CM: I60, I61, I62, I63, I64, G45; ICD-9-CM: 430, 431, 432, 433, 434, 435, 436 |
| `neuro_seizure` | Seizure or status epilepticus | --- |
| `neuro_altered_mental_status` | Encephalopathy or delirium as primary reason | --- |
| `neuro_other` | Other neurologic reason (Parkinson, MS, neuromuscular, etc.) | --- |
| `metabolic_dka_hhs` | Diabetic ketoacidosis or hyperosmolar hyperglycemic state | ICD-10-CM: E10.1, E11.0, E11.1, E13.0, E13.1; ICD-9-CM: 250.1, 250.2, 250.3 |
| `metabolic_electrolyte_crisis` | Severe electrolyte disturbance (hyperkalemia, hyponatremia crisis, etc.) | --- |
| `endocrine_other` | Other endocrine reason (thyroid storm, adrenal crisis, etc.) | --- |
| `heme_anemia_bleed` | Severe anemia or hemorrhage not clearly GI or trauma | --- |
| `heme_onc_complication` | Complication of cancer or its treatment (neutropenic fever, tumor lysis, etc.) | --- |
| `oncology_elective_treatment` | Planned chemotherapy, transplant, or oncology procedure admission | ICD-10-CM: Z51.0, Z51.1; ICD-9-CM: V58.0, V58.1 |
| `trauma_fracture` | Fracture from trauma | ICD-10-CM: S02, S12, S22, S32, S42, S52, S62, S72, S82, S92; ICD-9-CM: 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829 |
| `trauma_other` | Other trauma (blunt, penetrating, falls without fracture) | --- |
| `msk_non_trauma` | MSK admission without trauma (joint infection, non-traumatic back pain workup, etc.) | --- |
| `psych_mood_anxiety` | Depression, anxiety, bipolar mood episode (non-psychotic) | --- |
| `psych_psychosis_crisis` | Psychotic episode, suicidal ideation with plan, acute psychiatric crisis | --- |
| \texttt{substance\_intoxication\_\newline withdrawal} | Alcohol or drug intoxication or withdrawal | --- |
| `substance_overdose` | Intentional or unintentional overdose requiring admission | ICD-10-CM: T40, T42, T43, T50; ICD-9-CM: 965, 967, 969, 977 |
| `symptom_workup_chest_pain` | Chest pain admission for rule-out, workup inconclusive or ruled out | --- |
| `symptom_workup_syncope` | Syncope admission for workup | --- |
| `symptom_workup_other` | Other symptom-based admission for workup without definitive diagnosis at discharge | --- |
| `elective_procedure_non_oncology` | Planned non-oncology procedure (elective surgery, cardiac cath, etc.) | --- |
| `obstetric` | Pregnancy, labor, postpartum complications | --- |
| `other` | Reason does not fit any of the above categories | --- |

Table: \label{tbl:admission-tags} Forty-seven admission-reason tags with definitions and anchor ICD code families used for admission-reason analysis.

\clearpage

## TriState Field Definitions

| Field ID | Definition |
| -------- | ---------- |
| `shock_present` | Any form of shock documented during admission (cardiogenic, septic, hypovolemic, distributive). |
| `infection_as_trigger` | Infection identified as trigger or precipitant for the admission event, even if not the primary reason. |
| `aki_present` | Acute kidney injury present at any point during admission. |
| `lives_alone` | Patient lives alone at home per social history. |
| `social_support_absent` | Explicit documentation of lack of social support (isolated, no family, etc.). |
| `financial_hardship` | Documented financial hardship, uninsured, cost-related medication nonadherence. |
| `substance_use_active` | Active substance use (alcohol, illicit drugs, tobacco excluded). Historical use without active = 'no'. |
| `fall_risk_documented` | Fall risk explicitly documented. |
| `cognitive_impairment` | Baseline cognitive impairment (dementia, MCI) documented --- distinct from delirium. |
| `goals_of_care_flag` | Goals-of-care discussion documented during admission, including phrases like 'comfort-focused', 'discussed prognosis', 'family meeting re: goals'. |
| `palliative_care_consult` | Palliative care team formally consulted during this admission. |
| `dnr_dni_documented` | DNR, DNI, or DNR/DNI code status documented (not just discussed). |
| `home_health_ordered` | Home health services (nursing, PT/OT at home) ordered at discharge. |
| `cardiac_rehab_referred` | Referral to cardiac rehabilitation program at discharge. |
| `discharge_delayed_reason` | Discharge was delayed for non-medical reasons (placement, insurance, social). 'yes' only if explicitly documented. |
| `hospital_acquired_complication` | Any hospital-acquired complication documented: HAI, hospital-acquired AKI, hospital-acquired delirium, fall, pressure ulcer, etc. |
| `unresolved_diagnosis_at_discharge` | Language indicating the diagnosis remained unclear or workup pending at discharge ('etiology unclear', 'workup pending', 'to be followed up as outpatient'). |

Table: \label{tbl:tristate-fields} TriState field identifiers and definitions used for disagreement decomposition.

\clearpage

## Enum Fields and Value Sets

| Field ID | Value set | Value definitions |
| -------- | --------- | ----------------- |
| `functional_status` | `independent` | Performs activities of daily living independently. |
|  | `assisted` | Needs some assistance and/or assistive devices. |
|  | `dependent` | Requires substantial or full assistance for daily activities. |
|  | `not_documented` | No functional-status statement found in note. |
| `mental_status` | `intact` | Alert/oriented or documented at baseline mental status. |
|  | `mild_impairment` | Mild cognitive/mental-status impairment documented. |
|  | `confused_delirious` | Confusion, delirium, or marked disorientation documented. |
|  | `not_documented` | No mental-status statement found in note. |
| `discharge_condition_category` | `stable` | Discharge condition documented as stable. |
|  | `improved` | Discharge condition documented as improved. |
|  | `unchanged` | Discharge condition documented as unchanged. |
|  | `deteriorated` | Discharge condition documented as deteriorated. |
|  | `expired` | Patient died during admission. |
|  | `not_documented` | No discharge-condition category documented. |
| `new_meds_started_count` | `integer >= 0` | Count of distinct medications started during admission. |
|  | `null` | Medication reconciliation is indeterminate in note. |
| `meds_stopped_count` | `integer >= 0` | Count of distinct medications stopped during admission. |
|  | `null` | Medication reconciliation is indeterminate in note. |

Table: \label{tbl:enum-fields} Enum field value sets and operational definitions used for cross-variant confusion analysis.

\clearpage

## ICD-Based Labeling Functions

| LF Name | Target field | Match position | ICD codes/prefixes |
| ------- | ------------ | -------------- | ------------------ |
| `icd_aki_primary` | `aki_present` | Any | ICD-10-CM: N17\newline ICD-9-CM: 584 |
| `icd_hf_admission` | `cardiac_hf` | Primary | ICD-10-CM: I50, I11.0, I13.0, I13.2\newline ICD-9-CM: 428, 402.01, 402.11, 402.91, 404.01, 404.03, 404.11, 404.13, 404.91, 404.93 |
| `icd_acs_admission` | `cardiac_acs` | Primary | ICD-10-CM: I21, I22\newline ICD-9-CM: 410, 411 |
| `icd_stroke_admission` | `neuro_stroke_tia` | Primary | ICD-10-CM: I60, I61, I62, I63, I64, G45\newline ICD-9-CM: 430, 431, 432, 433, 434, 435, 436 |
| `icd_sepsis_admission` | `sepsis_bacteremia` | Any | ICD-10-CM: A40, A41, R65.2\newline ICD-9-CM: 038, 995.91, 995.92, 790.7 |
| `icd_afib_admission` | `cardiac_arrhythmia` | Primary | ICD-10-CM: I47, I48, I49\newline ICD-9-CM: 427 |
| `icd_copd_exacerbation` | \texttt{respiratory\_copd\_asthma\_\newline exacerbation} | Primary | ICD-10-CM: J44, J45, J46\newline ICD-9-CM: 491, 492, 493, 496 |
| `icd_pneumonia_admission` | `respiratory_infection` | Primary | ICD-10-CM: J12, J13, J14, J15, J16, J17, J18, J20, J21, J22\newline ICD-9-CM: 480, 481, 482, 483, 484, 485, 486, 487 |
| `icd_pe_admission` | `respiratory_pe_dvt` | Primary | ICD-10-CM: I26, I82\newline ICD-9-CM: 415.1, 451, 453 |
| `icd_gi_bleed_admission` | `gi_bleed` | Primary | ICD-10-CM: K25.0, K25.2, K25.4, K25.6, K26.0, K26.2, K26.4, K26.6, K27.0, K27.2, K27.4, K27.6, K28.0, K28.2, K28.4, K28.6, K29.01, K29.21, K29.31, K29.41, K29.51, K29.61, K29.71, K29.81, K29.91, K62.5, K92.0, K92.1, K92.2\newline ICD-9-CM: 530.82, 531.00, 531.01, 531.20, 531.21, 531.40, 531.41, 531.60, 531.61, 532, 533, 534, 535.01, 535.11, 535.21, 535.31, 535.41, 535.51, 535.61, 535.71, 569.3, 578 |
| `icd_cirrhosis_admission` | `hepatic_failure_cirrhosis` | Any | ICD-10-CM: K70, K71, K72, K73, K74, K76.6, K76.7\newline ICD-9-CM: 571, 572, 573 |
| `icd_dka_hhs_admission` | `metabolic_dka_hhs` | Primary | ICD-10-CM: E10.1, E11.0, E11.1, E13.0, E13.1\newline ICD-9-CM: 250.1, 250.2, 250.3 |
| \texttt{icd\_oncology\_treatment\_\newline admission} | `oncology_elective_treatment` | Primary | ICD-10-CM: Z51.0, Z51.1\newline ICD-9-CM: V58.0, V58.1 |
| `icd_fracture_admission` | `trauma_fracture` | Primary | ICD-10-CM: S02, S12, S22, S32, S42, S52, S62, S72, S82, S92\newline ICD-9-CM: 800-829 |
| `icd_overdose_admission` | `substance_overdose` | Primary | ICD-10-CM: T40, T42, T43, T50\newline ICD-9-CM: 965, 967, 969, 977 |

Table: \label{tbl:icd-lfs} ICD-based labeling functions used in the weak-supervision ensemble, including target mapping, match position, and anchor code families.

\clearpage

## Regex Labeling Functions

| LF Name | Target field | Regex pattern(s) |
| ------- | ------------ | ---------------- |
| `regex_aki_present_yes` | `aki_present` | `\bAKI\b`, `\bacute kidney injury\b`, `\bacute renal failure\b`, `\bARF\b` |
| `regex_cardiac_rehab_referred_yes` | `cardiac_rehab_referred` | `\bcardiac rehab(ilitation)?\b` |
| `regex_cognitive_impairment_yes` | `cognitive_impairment` | `\bdementia\b`, `\bAlzheimer('s)?\b`, `\bMCI\b`, `\bmild cognitive impairment\b`, `\bbaseline (confusion OR dementia)\b` |
| `regex_dnr_dni_documented_yes` | `dnr_dni_documented` | `\bDNR\s*/\s*DNI\b`, `\bDo Not Resuscitate\b`, `\bcomfort measures only\b` |
| `regex_fall_risk_documented_yes` | `fall_risk_documented` | `\bfall risk\b`, `\bhigh fall risk\b`; `compound(all of=['fell OR fall','at home OR mechanical OR witnessed OR unwitnessed'], window chars=20)` |
| `regex_goals_of_care_flag_yes` | `goals_of_care_flag` | `\bgoals[- ]of[- ]care\b`, `\bGOC\b`, `\bfamily meeting\b`, `\bcomfort[- ]focused\b`, `\bhospice\b` |
| `regex_home_health_ordered_yes` | `home_health_ordered` | `\bvisiting nurse\b`, `\bvisiting nursing\b`, `\bVNA\b` |
| `regex_palliative_care_consult_yes` | `palliative_care_consult` | `\bpalliative care\b` |
| `regex_substance_use_active_yes` | `substance_use_active` | `\bIVDU\b`, `\bpolysubstance\b`, `\bactive\s+(alcohol OR drug OR substance)\s+(use OR abuse)\b`, `\bCIWA\b`, `\bopiate withdrawal\b`, `\balcohol withdrawal\b`; `compound(all of=['alcohol OR ethanol OR etoh','abuse OR use OR consum OR overdose OR intox OR depend'], window chars=40)`; `compound(all of=['drink(er OR ing OR s)?','daily OR heavy OR continues OR currently OR active'], window chars=30)`; `compound(all of=['heroin OR cocaine OR fentanyl OR meth OR oxycodone OR opioid OR opiate','use OR abuse OR dependence OR active OR current'], window chars=40)` |


Table: \label{tbl:regex-lfs} Regex-based labeling functions used in the weak-supervision ensemble, including target mapping and anchor patterns.
