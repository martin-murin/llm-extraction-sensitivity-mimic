from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from src.llm.extractor import build_strict_json_schema, extract_note


def _valid_payload() -> dict[str, Any]:
    return {
        "admission_reason_tags": ["cardiac_hf", "renal_aki"],
        "dominant_admission_reason": "cardiac_hf",
        "primary_diagnosis_text": "Acute decompensated heart failure",
        "shock_present": "no",
        "infection_as_trigger": "not_documented",
        "aki_present": "yes",
        "functional_status": "assisted",
        "mental_status": "intact",
        "discharge_condition_category": "improved",
        "lives_alone": "yes",
        "social_support_absent": "no",
        "financial_hardship": "not_documented",
        "substance_use_active": "no",
        "fall_risk_documented": "yes",
        "cognitive_impairment": "no",
        "goals_of_care_flag": "yes",
        "palliative_care_consult": "no",
        "dnr_dni_documented": "yes",
        "new_meds_started_count": 3,
        "meds_stopped_count": 1,
        "home_health_ordered": "yes",
        "cardiac_rehab_referred": "no",
        "discharge_delayed_reason": "no",
        "hospital_acquired_complication": "no",
        "unresolved_diagnosis_at_discharge": "no",
        "reasoning": "Short rationale.",
    }


@dataclass
class _FakeResponse:
    content: str
    prompt_tokens: int = 100
    completion_tokens: int = 50

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        _ = mode
        return {
            "choices": [{"message": {"content": self.content}}],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
        }


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def chat(self, **_: Any) -> _FakeResponse:
        return self._response


@pytest.mark.asyncio
async def test_extract_note_valid_json_parses() -> None:
    payload = _valid_payload()
    client = _FakeClient(_FakeResponse(content=json.dumps(payload)))
    result = await extract_note("synthetic note", client=client, include_reasoning=True)
    assert result.features is not None
    assert result.parse_error is None
    assert result.features.dominant_admission_reason == "cardiac_hf"


@pytest.mark.asyncio
async def test_extract_note_invalid_json_returns_parse_error() -> None:
    client = _FakeClient(_FakeResponse(content="{not valid json"))
    result = await extract_note("synthetic note", client=client, include_reasoning=True)
    assert result.features is None
    assert result.parse_error is not None
    assert "JSONDecodeError" in result.parse_error


@pytest.mark.asyncio
async def test_extract_note_schema_violation_returns_parse_error() -> None:
    payload = _valid_payload()
    payload["admission_reason_tags"] = ["cardiac_hf"]
    payload["dominant_admission_reason"] = "respiratory_infection"
    client = _FakeClient(_FakeResponse(content=json.dumps(payload)))

    result = await extract_note("synthetic note", client=client, include_reasoning=True)
    assert result.features is None
    assert result.parse_error is not None
    assert "ValidationError" in result.parse_error


def test_strict_schema_postprocessing_enforces_object_requirements() -> None:
    schema = build_strict_json_schema()
    object_nodes = 0

    def walk(node: Any) -> None:
        nonlocal object_nodes
        if isinstance(node, dict):
            is_object = node.get("type") == "object" or "properties" in node
            if is_object:
                object_nodes += 1
                assert node.get("additionalProperties") is False
                properties = node.get("properties")
                keys = list(properties.keys()) if isinstance(properties, dict) else []
                assert node.get("required") == keys
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    assert object_nodes > 0
