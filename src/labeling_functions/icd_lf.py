from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.labeling_functions.base import LFInput, LFOutput, LabelingFunction, Vote

ICD_LF_SPECS: list[dict[str, Any]] = [
    {
        "name": "icd_aki_primary",
        "target_field": "aki_present",
        "target_value": "yes",
        "match_position": "any",
        "icd10_prefixes": ["N17"],
        "icd9_prefixes": ["584"],
    },
    {
        "name": "icd_hf_admission",
        "target_field": "admission_reason_tags",
        "target_value": "cardiac_hf",
        "match_position": "primary_only",
        "icd10_prefixes": ["I50", "I11.0", "I13.0", "I13.2"],
        "icd9_prefixes": [
            "428",
            "402.01",
            "402.11",
            "402.91",
            "404.01",
            "404.03",
            "404.11",
            "404.13",
            "404.91",
            "404.93",
        ],
    },
    {
        "name": "icd_acs_admission",
        "target_field": "admission_reason_tags",
        "target_value": "cardiac_acs",
        "match_position": "primary_only",
        "icd10_prefixes": ["I21", "I22"],
        "icd9_prefixes": ["410", "411"],
    },
    {
        "name": "icd_stroke_admission",
        "target_field": "admission_reason_tags",
        "target_value": "neuro_stroke_tia",
        "match_position": "primary_only",
        "icd10_prefixes": ["I60", "I61", "I62", "I63", "I64", "G45"],
        "icd9_prefixes": ["430", "431", "432", "433", "434", "435", "436"],
    },
    {
        "name": "icd_sepsis_admission",
        "target_field": "admission_reason_tags",
        "target_value": "sepsis_bacteremia",
        "match_position": "any",
        "icd10_prefixes": ["A40", "A41", "R65.2"],
        "icd9_prefixes": ["038", "995.91", "995.92", "790.7"],
    },
    {
        "name": "icd_afib_admission",
        "target_field": "admission_reason_tags",
        "target_value": "cardiac_arrhythmia",
        "match_position": "primary_only",
        "icd10_prefixes": ["I47", "I48", "I49"],
        "icd9_prefixes": ["427"],
    },
    {
        "name": "icd_copd_exacerbation",
        "target_field": "admission_reason_tags",
        "target_value": "respiratory_copd_asthma_exacerbation",
        "match_position": "primary_only",
        "icd10_prefixes": ["J44", "J45", "J46"],
        "icd9_prefixes": ["491", "492", "493", "496"],
    },
    {
        "name": "icd_pneumonia_admission",
        "target_field": "admission_reason_tags",
        "target_value": "respiratory_infection",
        "match_position": "primary_only",
        "icd10_prefixes": ["J12", "J13", "J14", "J15", "J16", "J17", "J18", "J20", "J21", "J22"],
        "icd9_prefixes": ["480", "481", "482", "483", "484", "485", "486", "487"],
    },
    {
        "name": "icd_pe_admission",
        "target_field": "admission_reason_tags",
        "target_value": "respiratory_pe_dvt",
        "match_position": "primary_only",
        "icd10_prefixes": ["I26", "I82"],
        "icd9_prefixes": ["415.1", "451", "453"],
    },
    {
        "name": "icd_gi_bleed_admission",
        "target_field": "admission_reason_tags",
        "target_value": "gi_bleed",
        "match_position": "primary_only",
        "icd10_prefixes": [
            "K25.0",
            "K25.2",
            "K25.4",
            "K25.6",
            "K26.0",
            "K26.2",
            "K26.4",
            "K26.6",
            "K27.0",
            "K27.2",
            "K27.4",
            "K27.6",
            "K28.0",
            "K28.2",
            "K28.4",
            "K28.6",
            "K29.01",
            "K29.21",
            "K29.31",
            "K29.41",
            "K29.51",
            "K29.61",
            "K29.71",
            "K29.81",
            "K29.91",
            "K62.5",
            "K92.0",
            "K92.1",
            "K92.2",
        ],
        "icd9_prefixes": [
            "530.82",
            "531.00",
            "531.01",
            "531.20",
            "531.21",
            "531.40",
            "531.41",
            "531.60",
            "531.61",
            "532",
            "533",
            "534",
            "535.01",
            "535.11",
            "535.21",
            "535.31",
            "535.41",
            "535.51",
            "535.61",
            "535.71",
            "569.3",
            "578",
        ],
    },
    {
        "name": "icd_cirrhosis_admission",
        "target_field": "admission_reason_tags",
        "target_value": "hepatic_failure_cirrhosis",
        "match_position": "any",
        "icd10_prefixes": ["K70", "K71", "K72", "K73", "K74", "K76.6", "K76.7"],
        "icd9_prefixes": ["571", "572", "573"],
    },
    {
        "name": "icd_dka_hhs_admission",
        "target_field": "admission_reason_tags",
        "target_value": "metabolic_dka_hhs",
        "match_position": "primary_only",
        "icd10_prefixes": ["E10.1", "E11.0", "E11.1", "E13.0", "E13.1"],
        "icd9_prefixes": ["250.1", "250.2", "250.3"],
    },
    {
        "name": "icd_oncology_treatment_admission",
        "target_field": "admission_reason_tags",
        "target_value": "oncology_elective_treatment",
        "match_position": "primary_only",
        "icd10_prefixes": ["Z51.0", "Z51.1"],
        "icd9_prefixes": ["V58.0", "V58.1"],
    },
    {
        "name": "icd_fracture_admission",
        "target_field": "admission_reason_tags",
        "target_value": "trauma_fracture",
        "match_position": "primary_only",
        "icd10_prefixes": ["S02", "S12", "S22", "S32", "S42", "S52", "S62", "S72", "S82", "S92"],
        "icd9_prefixes": [
            "800",
            "801",
            "802",
            "803",
            "804",
            "805",
            "806",
            "807",
            "808",
            "809",
            "810",
            "811",
            "812",
            "813",
            "814",
            "815",
            "816",
            "817",
            "818",
            "819",
            "820",
            "821",
            "822",
            "823",
            "824",
            "825",
            "826",
            "827",
            "828",
            "829",
        ],
    },
    {
        "name": "icd_overdose_admission",
        "target_field": "admission_reason_tags",
        "target_value": "substance_overdose",
        "match_position": "primary_only",
        "icd10_prefixes": ["T40", "T42", "T43", "T50"],
        "icd9_prefixes": ["965", "967", "969", "977"],
    },
]


def _normalize_code(code: str) -> str:
    return code.strip().upper()


def _code_matches_patterns(code: str, patterns: list[str]) -> bool:
    normalized_code = _normalize_code(code)
    for pattern in patterns:
        normalized_pattern = _normalize_code(pattern)
        if "." in normalized_pattern:
            if normalized_code == normalized_pattern:
                return True
            continue
        if normalized_code.startswith(normalized_pattern):
            return True
    return False


def _select_patterns_for_version(spec: dict[str, Any], version: int) -> list[str]:
    if version == 10:
        return [str(value) for value in spec.get("icd10_prefixes", [])]
    if version == 9:
        return [str(value) for value in spec.get("icd9_prefixes", [])]
    return []


@dataclass
class ICDLabelingFunction:
    name: str
    target_field: str
    target_value: str | None
    match_position: str
    spec: dict[str, Any]

    def _check_single_code(self, code: str, version: int) -> bool:
        patterns = _select_patterns_for_version(self.spec, version)
        if not patterns:
            return False
        return _code_matches_patterns(code, patterns)

    def __call__(self, inputs: LFInput) -> LFOutput:
        if self.match_position == "primary_only":
            code = inputs.primary_icd_code
            version = inputs.primary_icd_version
            if code is None or version is None:
                if inputs.icd_codes:
                    code, version = inputs.icd_codes[0]
                else:
                    return LFOutput(vote=Vote.ABSTAIN)

            if self._check_single_code(str(code), int(version)):
                return LFOutput(
                    vote=Vote.POSITIVE,
                    confidence=None,
                    evidence=f"ICD match: {code}",
                )
            return LFOutput(vote=Vote.ABSTAIN)

        if self.match_position == "any":
            for code, version in inputs.icd_codes or []:
                if self._check_single_code(str(code), int(version)):
                    return LFOutput(
                        vote=Vote.POSITIVE,
                        confidence=None,
                        evidence=f"ICD match: {code}",
                    )
            return LFOutput(vote=Vote.ABSTAIN)

        raise ValueError(f"Unknown match_position '{self.match_position}' for LF '{self.name}'.")


def build_icd_lf(spec: dict[str, Any]) -> LabelingFunction:
    return ICDLabelingFunction(
        name=str(spec["name"]),
        target_field=str(spec["target_field"]),
        target_value=spec.get("target_value"),
        match_position=str(spec["match_position"]),
        spec=spec,
    )


def build_all_icd_lfs() -> list[LabelingFunction]:
    return [build_icd_lf(spec) for spec in ICD_LF_SPECS]
