from __future__ import annotations

from src.labeling_functions.section_parser import get_section, parse_sections


def test_parse_sections_three_well_formed_sections() -> None:
    note = (
        "Patient intro text.\n\n"
        "Chief Complaint:\n"
        "Chest pain\n\n"
        "History of Present Illness:\n"
        "Two days of worsening dyspnea.\n\n"
        "Discharge Condition:\n"
        "Improved and stable for discharge.\n"
    )
    sections = parse_sections(note)
    assert set(sections.keys()) == {
        "Chief Complaint",
        "History of Present Illness",
        "Discharge Condition",
    }
    assert sections["Chief Complaint"] == "Chest pain"
    assert sections["History of Present Illness"] == "Two days of worsening dyspnea."


def test_parse_sections_no_headers_returns_full_note_sentinel() -> None:
    note = "Freeform narrative with no recognizable headers."
    sections = parse_sections(note)
    assert sections == {"__full_note__": note}


def test_get_section_alias_resolution() -> None:
    sections = {"Hospital Course": "Improved with diuresis and antibiotics."}
    resolved = get_section(sections, "Brief Hospital Course")
    assert resolved == "Improved with diuresis and antibiotics."


def test_parse_sections_duplicate_section_keeps_first() -> None:
    note = (
        "Physical Exam:\n"
        "Initial exam text.\n\n"
        "Physical Exam:\n"
        "Later duplicate should be ignored.\n"
    )
    sections = parse_sections(note)
    assert "Physical Exam" in sections
    assert sections["Physical Exam"] == "Initial exam text."


def test_parse_sections_strips_body_whitespace() -> None:
    note = (
        "Discharge Diagnosis:\n"
        "   Acute heart failure   \n\n"
        "Discharge Condition:\n"
        "\t Improved \t\n"
    )
    sections = parse_sections(note)
    assert sections["Discharge Diagnosis"] == "Acute heart failure"
    assert sections["Discharge Condition"] == "Improved"


def test_parse_sections_mimic_style_sample() -> None:
    note = (
        "Name: ___\n\n"
        "Chief Complaint:\n"
        "shortness of breath\n\n"
        "History of Present Illness:\n"
        "___ yo male with CHF presents with worsening edema and dyspnea.\n\n"
        "Brief Hospital Course:\n"
        "Treated with IV diuretics. Creatinine rose then improved.\n\n"
        "Discharge Condition:\n"
        "Hemodynamically stable.\n"
    )
    sections = parse_sections(note)
    assert "Chief Complaint" in sections
    assert "History of Present Illness" in sections
    assert "Brief Hospital Course" in sections
    assert "Discharge Condition" in sections
