from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

SECTION_HEADER_PATTERN = re.compile(r"(?m)^([A-Z][A-Za-z ()/-]{2,60}):\s*\n")

SECTION_ALIASES: dict[str, list[str]] = {
    "Brief Hospital Course": ["Brief Hospital Course", "Hospital Course", "Active Issues"],
    "History of Present Illness": ["History of Present Illness", "HPI", "Presenting Illness"],
    "Past Medical History": ["Past Medical History", "PMH"],
    "Social History": ["Social History", "Social Hx"],
    "Discharge Medications": ["Discharge Medications", "Discharge Meds"],
    "Discharge Disposition": ["Discharge Disposition", "Disposition"],
    "Discharge Condition": ["Discharge Condition"],
    "Discharge Diagnosis": ["Discharge Diagnosis", "Discharge Diagnoses", "Final Diagnosis"],
    "Discharge Instructions": ["Discharge Instructions"],
    "Physical Exam": ["Physical Exam", "Physical Examination", "Admission Exam", "Discharge Exam"],
    "Chief Complaint": ["Chief Complaint", "CC"],
    "Pertinent Results": ["Pertinent Results", "Labs", "Results"],
    "Medications on Admission": ["Medications on Admission", "Admission Meds"],
}


def parse_sections(note_text: str) -> dict[str, str]:
    matches = list(SECTION_HEADER_PATTERN.finditer(note_text))
    if not matches:
        return {"__full_note__": note_text}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        section_name = match.group(1).strip().rstrip(":")
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(note_text)
        section_body = note_text[body_start:body_end].strip()

        if section_name in sections:
            logger.debug(
                "Duplicate section encountered; keeping first",
                extra={"section": section_name},
            )
            continue
        sections[section_name] = section_body

    return sections


def get_section(sections: dict[str, str], canonical_name: str) -> str | None:
    candidates = [canonical_name]
    candidates.extend(SECTION_ALIASES.get(canonical_name, []))

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in sections:
            return sections[candidate]
    return None


def coverage_report(notes: dict[int, str]) -> pd.DataFrame:
    if not notes:
        return pd.DataFrame(
            columns=[
                "canonical_section",
                "n_present",
                "n_absent",
                "coverage_pct",
                "median_length_chars",
            ]
        )

    parsed_by_hadm = {hadm_id: parse_sections(text) for hadm_id, text in notes.items()}
    n_notes = len(parsed_by_hadm)

    rows: list[dict[str, float | int | str]] = []
    for canonical_name in SECTION_ALIASES:
        lengths: list[int] = []
        n_present = 0
        for sections in parsed_by_hadm.values():
            section_body = get_section(sections, canonical_name)
            if section_body is None:
                continue
            n_present += 1
            lengths.append(len(section_body))

        n_absent = n_notes - n_present
        median_len = float(np.median(np.asarray(lengths, dtype=np.int64))) if lengths else 0.0
        rows.append(
            {
                "canonical_section": canonical_name,
                "n_present": n_present,
                "n_absent": n_absent,
                "coverage_pct": (n_present / n_notes) * 100.0,
                "median_length_chars": median_len,
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        by="coverage_pct",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
