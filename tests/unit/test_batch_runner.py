from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.llm.batch_runner import run_batch
from src.llm.client import build_retrying, is_retryable_exception
from src.llm.extractor import ExtractionResult
from src.schema.fields import LLMNoteFeatures
from src.utils.logging import BudgetExceededError


def _valid_features() -> LLMNoteFeatures:
    return LLMNoteFeatures(
        admission_reason_tags=["cardiac_hf"],
        dominant_admission_reason="cardiac_hf",
        primary_diagnosis_text="Acute decompensated heart failure",
        shock_present="no",
        infection_as_trigger="not_documented",
        aki_present="yes",
        functional_status="assisted",
        mental_status="intact",
        discharge_condition_category="improved",
        lives_alone="yes",
        social_support_absent="no",
        financial_hardship="not_documented",
        substance_use_active="no",
        fall_risk_documented="yes",
        cognitive_impairment="no",
        goals_of_care_flag="yes",
        palliative_care_consult="no",
        dnr_dni_documented="yes",
        new_meds_started_count=3,
        meds_stopped_count=1,
        home_health_ordered="yes",
        cardiac_rehab_referred="no",
        discharge_delayed_reason="no",
        hospital_acquired_complication="no",
        unresolved_diagnosis_at_discharge="no",
        reasoning="Short rationale.",
    )


class _FakeCostTracker:
    def summary(self) -> dict[str, float]:
        return {"total_cost_usd": 0.0}


class _FakeClient:
    def __init__(self) -> None:
        self.cost_tracker = _FakeCostTracker()


@pytest.mark.asyncio
async def test_run_batch_writes_three_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_extract_note(**_: Any) -> ExtractionResult:
        return ExtractionResult(
            features=_valid_features(),
            raw_response={"choices": [{"message": {"content": "{}"}}], "usage": {}},
            parse_error=None,
            input_tokens=100,
            output_tokens=50,
            latency_seconds=0.1,
        )

    monkeypatch.setattr("src.llm.batch_runner.extract_note", fake_extract_note)
    notes = {101: "note A", 102: "note B", 103: "note C"}

    summary = await run_batch(
        notes=notes,
        client=_FakeClient(),
        run_id="test_run",
        output_dir=tmp_path,
        checkpoint_every=1,
    )

    assert summary.n_successful_parse == 3
    assert (tmp_path / "101.json").exists()
    assert (tmp_path / "102.json").exists()
    assert (tmp_path / "103.json").exists()


@pytest.mark.asyncio
async def test_run_batch_resume_skips_existing_note(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing_payload = {
        "hadm_id": 101,
        "parse_ok": True,
        "parse_error": None,
        "input_tokens": 10,
        "output_tokens": 5,
        "latency_seconds": 0.05,
        "features_json": _valid_features().model_dump(mode="json"),
        "raw_response": {"choices": [{"message": {"content": "{}"}}], "usage": {}},
    }
    (tmp_path / "101.json").write_text(
        json.dumps(existing_payload),
        encoding="utf-8",
    )

    call_hadm_ids: list[int] = []

    async def fake_extract_note(note_text: str, **_: Any) -> ExtractionResult:
        hadm_id = int(note_text.split("_")[1])
        call_hadm_ids.append(hadm_id)
        return ExtractionResult(
            features=_valid_features(),
            raw_response={"choices": [{"message": {"content": "{}"}}], "usage": {}},
            parse_error=None,
            input_tokens=100,
            output_tokens=50,
            latency_seconds=0.1,
        )

    monkeypatch.setattr("src.llm.batch_runner.extract_note", fake_extract_note)

    notes = {
        101: "hadm_101",
        102: "hadm_102",
        103: "hadm_103",
    }
    summary = await run_batch(
        notes=notes,
        client=_FakeClient(),
        run_id="resume_run",
        output_dir=tmp_path,
        checkpoint_every=1,
        resume=True,
    )

    assert summary.n_successful_parse == 3
    assert len(call_hadm_ids) == 2
    assert set(call_hadm_ids) == {102, 103}


@pytest.mark.asyncio
async def test_run_batch_continues_after_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    call_count = 0

    async def fake_extract_note(**_: Any) -> ExtractionResult:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return ExtractionResult(
                features=None,
                raw_response={"choices": [{"message": {"content": "{bad json}"}}], "usage": {}},
                parse_error="ValidationError: bad schema",
                input_tokens=90,
                output_tokens=40,
                latency_seconds=0.2,
            )
        return ExtractionResult(
            features=_valid_features(),
            raw_response={"choices": [{"message": {"content": "{}"}}], "usage": {}},
            parse_error=None,
            input_tokens=100,
            output_tokens=50,
            latency_seconds=0.1,
        )

    monkeypatch.setattr("src.llm.batch_runner.extract_note", fake_extract_note)
    notes = {201: "note A", 202: "note B", 203: "note C"}
    summary = await run_batch(
        notes=notes,
        client=_FakeClient(),
        run_id="parse_failure_run",
        output_dir=tmp_path,
        checkpoint_every=1,
    )

    assert summary.n_successful_parse == 2
    assert summary.n_failed_parse == 1
    assert summary.n_api_error == 0


@pytest.mark.asyncio
async def test_run_batch_handles_budget_exceeded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    call_count = 0

    async def fake_extract_note(**_: Any) -> ExtractionResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise BudgetExceededError("Budget cap hit")
        return ExtractionResult(
            features=_valid_features(),
            raw_response={"choices": [{"message": {"content": "{}"}}], "usage": {}},
            parse_error=None,
            input_tokens=100,
            output_tokens=50,
            latency_seconds=0.1,
        )

    monkeypatch.setattr("src.llm.batch_runner.extract_note", fake_extract_note)
    notes = {301: "note A", 302: "note B", 303: "note C"}
    summary = await run_batch(
        notes=notes,
        client=_FakeClient(),
        run_id="budget_run",
        output_dir=tmp_path,
        max_concurrency=1,
        checkpoint_every=1,
    )

    assert summary.n_api_error >= 1
    written_files = list(tmp_path.glob("*.json"))
    assert len(written_files) == 2
    assert (tmp_path / "run_metadata.json").exists()


def test_extraction_result_serializes_with_asdict() -> None:
    result = ExtractionResult(
        features=_valid_features(),
        raw_response={"choices": []},
        parse_error=None,
        input_tokens=123,
        output_tokens=456,
        latency_seconds=0.789,
    )
    serialized = asdict(result)
    assert serialized["input_tokens"] == 123
    assert serialized["output_tokens"] == 456
    assert serialized["parse_error"] is None


def test_production_yaml_default_concurrency_is_8() -> None:
    payload = yaml.safe_load(Path("configs/production.yaml").read_text(encoding="utf-8")) or {}
    assert int(payload["max_concurrent_requests"]) == 8


def test_llm_retry_policy_has_5_attempt_cap() -> None:
    retrying = build_retrying()
    assert int(getattr(retrying.stop, "max_attempt_number", 0)) == 5


class _Fake429Error(Exception):
    status_code = 429


def test_retryable_exception_logic() -> None:
    assert is_retryable_exception(_Fake429Error("429"))
    assert not is_retryable_exception(ValueError("bad input"))
