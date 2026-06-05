from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.labeling_functions.pattern_bootstrap import (
    EXTENSION_PILOT_FIELDS,
    derive_embedding_seed_phrases,
    derive_regex_patterns,
    extract_anchor_phrases,
    write_pattern_yaml,
)
from src.labeling_functions.regex_lf import load_pattern_yaml
from src.schema.fields import LLMNoteFeatures
from src.utils.logging import BudgetExceededError


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        assert mode == "json"
        return {
            "choices": [
                {
                    "message": {
                        "content": self._content,
                    }
                }
            ]
        }


class _BudgetAwareFakeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **_: Any) -> _FakeResponse:
        self.calls += 1
        if self.calls == 2:
            raise BudgetExceededError("budget exceeded")
        return _FakeResponse(json.dumps(["DNR/DNI documented in chart"]))


def test_derive_regex_patterns_empty() -> None:
    assert derive_regex_patterns([]) == []


def test_derive_regex_patterns_finds_do_not_resuscitate() -> None:
    phrases = [
        "Family agreed to do not resuscitate order.",
        "Code status changed to do not resuscitate after discussion.",
        "Patient signed do not resuscitate paperwork.",
    ]
    patterns = derive_regex_patterns(phrases)
    assert any("do not resuscitate" in pattern.lower() for pattern in patterns)


def test_derive_regex_patterns_drops_single_occurrence_ngrams() -> None:
    phrases = [
        "Do not resuscitate order is active.",
        "Do not resuscitate status documented.",
        "Comfort care consult was completed.",
    ]
    patterns = derive_regex_patterns(phrases)
    assert all("comfort care" not in pattern.lower() for pattern in patterns)


def test_derive_regex_patterns_prefers_longer_over_shorter_substrings() -> None:
    phrases = [
        "Code status DNR documented.",
        "Code status DNR confirmed by family.",
        "Code status DNR discussed today.",
    ]
    patterns = derive_regex_patterns(phrases)
    assert any("code status dnr" in pattern.lower() for pattern in patterns)
    assert all(pattern.lower() != r"\bcode status\b" for pattern in patterns)


def test_derive_embedding_seed_phrases_deduplicates_near_duplicates() -> None:
    phrases = [
        "Patient states he uses cocaine daily with recent relapse.",
        "Patient states he uses cocaine daily but denied withdrawal symptoms.",
        "Patient reports active heroin use before admission.",
    ]
    seeds = derive_embedding_seed_phrases(phrases)
    assert len(seeds) == 2


def test_write_pattern_yaml_and_load_roundtrip(tmp_path: Path) -> None:
    output_path = write_pattern_yaml(
        field_name="dnr_dni_documented",
        target_value="yes",
        regex_patterns=[r"\bdnr\b", r"\bdo not resuscitate\b"],
        seed_phrases=["DNR status documented", "Do not resuscitate order signed"],
        source_run_id="coverage_v2",
        n_source_notes=10,
        output_dir=tmp_path,
    )

    loaded = load_pattern_yaml(output_path)
    assert loaded["field_name"] == "dnr_dni_documented"
    assert loaded["target_value"] == "yes"
    assert loaded["regex_patterns"] == [r"\bdnr\b", r"\bdo not resuscitate\b"]
    assert loaded["embedding_seed_phrases"] == [
        "DNR status documented",
        "Do not resuscitate order signed",
    ]


def test_extract_anchor_phrases_stops_on_budget_exceeded() -> None:
    results = [
        {
            "hadm_id": 1,
            "parse_ok": True,
            "features": {"dnr_dni_documented": "yes"},
            "reasoning": "Reasoning with quoted DNR statement.",
        },
        {
            "hadm_id": 2,
            "parse_ok": True,
            "features": {"dnr_dni_documented": "yes"},
            "reasoning": "Another reasoning block.",
        },
    ]

    client = _BudgetAwareFakeClient()
    anchors = extract_anchor_phrases(
        results=results,
        field_name="dnr_dni_documented",
        target_value="yes",
        llm_client=client,
    )

    assert list(anchors.keys()) == [1]
    assert anchors[1] == ["DNR/DNI documented in chart"]


def test_extension_pilot_fields_are_valid_schema_fields() -> None:
    schema_fields = set(LLMNoteFeatures.model_fields.keys())
    assert len(EXTENSION_PILOT_FIELDS) == 5
    for field_name, target_value in EXTENSION_PILOT_FIELDS:
        assert field_name in schema_fields
        assert target_value == "yes"
