from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src import config
from src.llm.client import LLMClient, retry_policy_label
from src.llm.extractor import extract_note
from src.utils.logging import BudgetExceededError

logger = logging.getLogger(__name__)


@dataclass
class BatchSummary:
    run_id: str
    variant: str
    include_reasoning: bool
    n_total: int
    n_successful_parse: int
    n_failed_parse: int
    n_api_error: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    elapsed_seconds: float
    median_latency_seconds: float
    p95_latency_seconds: float
    per_note_results_path: Path


@dataclass
class _Outcome:
    hadm_id: int
    parse_ok: bool
    parse_error: str | None
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    features_json: dict[str, Any] | None
    raw_response: dict[str, Any]
    budget_exceeded: bool


def _is_api_error(parse_error: str | None) -> bool:
    return bool(parse_error and parse_error.startswith("api_error:"))


def _extract_metadata_from_saved(payload: dict[str, Any]) -> _Outcome:
    parse_ok = bool(payload.get("parse_ok", False))
    parse_error = payload.get("parse_error")
    if parse_error is not None and not isinstance(parse_error, str):
        parse_error = str(parse_error)

    input_tokens = int(payload.get("input_tokens", 0) or 0)
    output_tokens = int(payload.get("output_tokens", 0) or 0)
    latency_seconds = float(payload.get("latency_seconds", 0.0) or 0.0)
    features_json = payload.get("features_json")
    if features_json is not None and not isinstance(features_json, dict):
        features_json = None

    raw_response = payload.get("raw_response")
    if not isinstance(raw_response, dict):
        raw_response = {}

    hadm_id = int(payload.get("hadm_id", 0) or 0)
    return _Outcome(
        hadm_id=hadm_id,
        parse_ok=parse_ok,
        parse_error=parse_error,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency_seconds,
        features_json=features_json,
        raw_response=raw_response,
        budget_exceeded=False,
    )


def _to_per_note_payload(
    *,
    hadm_id: int,
    run_id: str,
    variant: str,
    include_reasoning: bool,
    outcome: _Outcome,
) -> dict[str, Any]:
    return {
        "hadm_id": hadm_id,
        "run_id": run_id,
        "variant": variant,
        "include_reasoning": include_reasoning,
        "processed_at_utc": datetime.now(tz=UTC).isoformat(),
        "parse_ok": outcome.parse_ok,
        "parse_error": outcome.parse_error,
        "input_tokens": outcome.input_tokens,
        "output_tokens": outcome.output_tokens,
        "latency_seconds": outcome.latency_seconds,
        "features_json": outcome.features_json,
        "raw_response": outcome.raw_response,
    }


def _to_jsonl_record(*, hadm_id: int, outcome: _Outcome) -> dict[str, Any]:
    return {
        "hadm_id": hadm_id,
        "parse_ok": outcome.parse_ok,
        "features_json": outcome.features_json,
        "parse_error": outcome.parse_error,
        "input_tokens": outcome.input_tokens,
        "output_tokens": outcome.output_tokens,
        "latency_seconds": outcome.latency_seconds,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _write_run_metadata(
    *,
    output_dir: Path,
    run_id: str,
    variant: str,
    include_reasoning: bool,
    max_concurrency: int,
    checkpoint_every: int,
    resume: bool,
    client: LLMClient,
    started_at_utc: str,
    completed_at_utc: str | None = None,
    summary: BatchSummary | None = None,
) -> None:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "variant": variant,
        "include_reasoning": include_reasoning,
        "max_concurrency_requested": int(max_concurrency),
        "client_semaphore_limit": int(getattr(client, "semaphore_limit", max_concurrency)),
        "checkpoint_every": int(checkpoint_every),
        "resume": bool(resume),
        "max_retries": int(config.MAX_RETRIES),
        "retry_policy": retry_policy_label(),
        "started_at_utc": started_at_utc,
    }
    if completed_at_utc is not None:
        payload["completed_at_utc"] = completed_at_utc
    if summary is not None:
        payload.update(
            {
                "n_total": int(summary.n_total),
                "n_successful_parse": int(summary.n_successful_parse),
                "n_failed_parse": int(summary.n_failed_parse),
                "n_api_error": int(summary.n_api_error),
                "elapsed_seconds": float(summary.elapsed_seconds),
                "median_latency_seconds": float(summary.median_latency_seconds),
                "p95_latency_seconds": float(summary.p95_latency_seconds),
                "total_cost_usd": float(summary.total_cost_usd),
                "total_input_tokens": int(summary.total_input_tokens),
                "total_output_tokens": int(summary.total_output_tokens),
            }
        )
    path = output_dir / "run_metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


async def run_batch(
    notes: dict[int, str],
    client: LLMClient,
    run_id: str,
    output_dir: Path,
    variant: str = "a",
    include_reasoning: bool = True,
    max_concurrency: int = 20,
    checkpoint_every: int = 50,
    resume: bool = True,
) -> BatchSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_note_results_path = output_dir / "results.jsonl"
    started_at_utc = datetime.now(tz=UTC).isoformat()
    _write_run_metadata(
        output_dir=output_dir,
        run_id=run_id,
        variant=variant,
        include_reasoning=include_reasoning,
        max_concurrency=max_concurrency,
        checkpoint_every=checkpoint_every,
        resume=resume,
        client=client,
        started_at_utc=started_at_utc,
    )

    ordered_items = sorted(
        ((int(hadm_id), text) for hadm_id, text in notes.items()),
        key=lambda x: x[0],
    )
    n_total = len(ordered_items)

    if not resume:
        existing = [output_dir / f"{hadm_id}.json" for hadm_id, _ in ordered_items]
        if any(path.exists() for path in existing):
            raise FileExistsError(
                f"Refusing to overwrite existing per-note JSON files in {output_dir}. "
                "Delete them explicitly or rerun with resume=True."
            )
        if per_note_results_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite consolidated results file {per_note_results_path}. "
                "Delete it explicitly or rerun with resume=True."
            )

    existing_outcomes: list[_Outcome] = []
    to_process: list[tuple[int, str]] = []
    for hadm_id, note_text in ordered_items:
        result_path = output_dir / f"{hadm_id}.json"
        if resume and result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            existing_outcome = _extract_metadata_from_saved(saved)
            existing_outcome.hadm_id = hadm_id
            existing_outcomes.append(existing_outcome)
            continue
        to_process.append((hadm_id, note_text))

    logger.info(
        "Found %s existing results; processing remaining %s notes.",
        len(existing_outcomes),
        len(to_process),
    )

    n_successful_parse = 0
    n_failed_parse = 0
    n_api_error = 0
    total_input_tokens = 0
    total_output_tokens = 0
    latencies: list[float] = []

    for outcome in existing_outcomes:
        if outcome.parse_ok:
            n_successful_parse += 1
        elif _is_api_error(outcome.parse_error):
            n_api_error += 1
        else:
            n_failed_parse += 1
        total_input_tokens += outcome.input_tokens
        total_output_tokens += outcome.output_tokens
        if outcome.latency_seconds > 0:
            latencies.append(outcome.latency_seconds)

    file_mode = "a" if resume else "w"
    jsonl_handle = per_note_results_path.open(file_mode, encoding="utf-8")
    batch_semaphore = asyncio.Semaphore(max_concurrency)

    async def _process_one(hadm_id: int, note_text: str) -> _Outcome:
        async with batch_semaphore:
            try:
                result = await extract_note(
                    note_text=note_text,
                    client=client,
                    variant=variant,
                    include_reasoning=include_reasoning,
                )
            except BudgetExceededError as exc:
                logger.error("Budget exceeded while processing hadm_id=%s: %s", hadm_id, exc)
                return _Outcome(
                    hadm_id=hadm_id,
                    parse_ok=False,
                    parse_error=f"api_error: BudgetExceededError: {exc}",
                    input_tokens=0,
                    output_tokens=0,
                    latency_seconds=0.0,
                    features_json=None,
                    raw_response={},
                    budget_exceeded=True,
                )
            except Exception as exc:
                logger.exception("API call failed for hadm_id=%s", hadm_id)
                return _Outcome(
                    hadm_id=hadm_id,
                    parse_ok=False,
                    parse_error=f"api_error: {type(exc).__name__}: {exc}",
                    input_tokens=0,
                    output_tokens=0,
                    latency_seconds=0.0,
                    features_json=None,
                    raw_response={},
                    budget_exceeded=False,
                )

            features_json = None
            parse_ok = result.features is not None
            if result.features is not None:
                features_json = result.features.model_dump(mode="json")

            return _Outcome(
                hadm_id=hadm_id,
                parse_ok=parse_ok,
                parse_error=result.parse_error,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_seconds=result.latency_seconds,
                features_json=features_json,
                raw_response=result.raw_response,
                budget_exceeded=False,
            )

    started_at = perf_counter()
    pending_index = 0
    in_flight: set[asyncio.Task[_Outcome]] = set()
    stop_dispatch = False
    processed_new = 0

    try:
        while pending_index < len(to_process) or in_flight:
            while (
                not stop_dispatch
                and pending_index < len(to_process)
                and len(in_flight) < max_concurrency
            ):
                hadm_id, note_text = to_process[pending_index]
                pending_index += 1
                in_flight.add(asyncio.create_task(_process_one(hadm_id, note_text)))

            if not in_flight:
                break

            done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                outcome = await task
                hadm_id = outcome.hadm_id

                result_path = output_dir / f"{hadm_id}.json"
                if result_path.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite existing raw response file: {result_path}"
                    )

                per_note_payload = _to_per_note_payload(
                    hadm_id=hadm_id,
                    run_id=run_id,
                    variant=variant,
                    include_reasoning=include_reasoning,
                    outcome=outcome,
                )
                result_path.write_text(
                    json.dumps(per_note_payload, ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )

                jsonl_record = _to_jsonl_record(hadm_id=hadm_id, outcome=outcome)
                jsonl_handle.write(json.dumps(jsonl_record, ensure_ascii=True) + "\n")

                processed_new += 1
                if outcome.parse_ok:
                    n_successful_parse += 1
                elif _is_api_error(outcome.parse_error):
                    n_api_error += 1
                else:
                    n_failed_parse += 1

                total_input_tokens += outcome.input_tokens
                total_output_tokens += outcome.output_tokens
                if outcome.latency_seconds > 0:
                    latencies.append(outcome.latency_seconds)

                if outcome.budget_exceeded:
                    stop_dispatch = True

                should_checkpoint = (
                    processed_new > 0 and processed_new % max(1, checkpoint_every) == 0
                ) or stop_dispatch
                if should_checkpoint:
                    jsonl_handle.flush()
                    elapsed = max(perf_counter() - started_at, 1e-9)
                    throughput = processed_new / elapsed
                    remaining = max(len(to_process) - pending_index, 0) + len(in_flight)
                    eta_seconds = (remaining / throughput) if throughput > 0 else float("inf")
                    running_cost = float(client.cost_tracker.summary()["total_cost_usd"])
                    logger.info(
                        (
                            "Checkpoint %s/%s processed | cost=$%.4f | median_latency=%.3fs "
                            "| eta=%ss"
                        ),
                        len(existing_outcomes) + processed_new,
                        n_total,
                        running_cost,
                        _percentile(latencies, 50),
                        "inf" if not np.isfinite(eta_seconds) else f"{eta_seconds:.1f}",
                    )
    finally:
        jsonl_handle.flush()
        jsonl_handle.close()

    elapsed_seconds = perf_counter() - started_at
    total_cost_usd = float(client.cost_tracker.summary()["total_cost_usd"])

    summary = BatchSummary(
        run_id=run_id,
        variant=variant,
        include_reasoning=include_reasoning,
        n_total=n_total,
        n_successful_parse=n_successful_parse,
        n_failed_parse=n_failed_parse,
        n_api_error=n_api_error,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=total_cost_usd,
        elapsed_seconds=elapsed_seconds,
        median_latency_seconds=_percentile(latencies, 50),
        p95_latency_seconds=_percentile(latencies, 95),
        per_note_results_path=per_note_results_path,
    )
    _write_run_metadata(
        output_dir=output_dir,
        run_id=run_id,
        variant=variant,
        include_reasoning=include_reasoning,
        max_concurrency=max_concurrency,
        checkpoint_every=checkpoint_every,
        resume=resume,
        client=client,
        started_at_utc=started_at_utc,
        completed_at_utc=datetime.now(tz=UTC).isoformat(),
        summary=summary,
    )
    return summary
