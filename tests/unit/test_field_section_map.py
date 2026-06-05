from __future__ import annotations

from src.labeling_functions.section_parser import SECTION_ALIASES
from src.schema.fields import LLMNoteFeatures
from src.schema.section_map import FIELD_SECTION_MAP


def test_every_llm_field_except_reasoning_is_mapped() -> None:
    expected_fields = set(LLMNoteFeatures.model_fields.keys()).difference({"reasoning"})
    assert set(FIELD_SECTION_MAP.keys()) == expected_fields


def test_field_section_map_names_align_with_section_aliases() -> None:
    known_sections = set(SECTION_ALIASES.keys())
    for aliases in SECTION_ALIASES.values():
        known_sections.update(aliases)

    for field_name, section_names in FIELD_SECTION_MAP.items():
        for section_name in section_names:
            assert section_name in known_sections, (
                f"Unknown section '{section_name}' referenced by field '{field_name}'."
            )
