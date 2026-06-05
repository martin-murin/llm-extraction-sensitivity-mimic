from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.llm.extractor import extract_content_from_raw_response
from src.schema.fields import LLMNoteFeatures
from src.utils.logging import BudgetExceededError

logger = logging.getLogger(__name__)

REGEX_PILOT_FIELDS: list[tuple[str, str]] = [
    ("dnr_dni_documented", "yes"),
    ("palliative_care_consult", "yes"),
    ("home_health_ordered", "yes"),
    ("cardiac_rehab_referred", "yes"),
    ("substance_use_active", "yes"),
]

EXTENSION_PILOT_FIELDS: list[tuple[str, str]] = [
    ("aki_present", "yes"),
    ("fall_risk_documented", "yes"),
    ("cognitive_impairment", "yes"),
    ("goals_of_care_flag", "yes"),
    ("hospital_acquired_complication", "yes"),
]

_SCHEMA_FIELDS = set(LLMNoteFeatures.model_fields.keys())
for _field_name, _target_value in [*REGEX_PILOT_FIELDS, *EXTENSION_PILOT_FIELDS]:
    if _field_name not in _SCHEMA_FIELDS:
        raise ValueError(f"Unknown field in bootstrap field set: {_field_name}")
    if _target_value != "yes":
        raise ValueError(
            f"Bootstrap field set currently supports yes-target only, got {_target_value}"
        )

_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*")


def load_coverage_v2_results(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        raise FileNotFoundError(f"Coverage results not found: {results_path}")

    loaded: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not bool(payload.get("parse_ok", False)):
                continue
            features = payload.get("features_json")
            if not isinstance(features, dict):
                continue
            loaded.append(
                {
                    "hadm_id": int(payload["hadm_id"]),
                    "parse_ok": True,
                    "features": features,
                    "reasoning": features.get("reasoning"),
                }
            )
    return loaded


async def _extract_anchor_phrases_async(
    results: list[dict[str, Any]],
    field_name: str,
    target_value: str,
    llm_client: Any,
) -> dict[int, list[str]]:
    response_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "anchor_phrase_list",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "phrases": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["phrases"],
                "additionalProperties": False,
            },
        },
    }

    anchors: dict[int, list[str]] = {}
    source_rows = [
        row
        for row in results
        if isinstance(row.get("features"), dict)
        and row["features"].get(field_name) == target_value
    ]

    for row in source_rows:
        hadm_id = int(row["hadm_id"])
        reasoning = row.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            anchors[hadm_id] = []
            continue

        prompt = (
            "You are analyzing an LLM extraction rationale. The LLM just decided that field "
            f"`{field_name}` should be `{target_value}` for a clinical note. Here is the LLM's "
            "reasoning:\n\n"
            "---\n"
            f"{reasoning}\n"
            "---\n\n"
            "Return a JSON array of 0-3 verbatim note excerpts (strings) that the reasoning "
            f"quotes as evidence for `{field_name} = {target_value}`. Return [] if no verbatim "
            "excerpts are present. Do not paraphrase. Do not invent."
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await llm_client.chat(
                messages=messages,
                response_format=response_format,
                max_completion_tokens=200,
            )
        except BudgetExceededError:
            logger.warning(
                "Bootstrap budget exceeded while extracting anchors for %s=%s.",
                field_name,
                target_value,
            )
            break

        raw_response: dict[str, Any]
        if hasattr(response, "model_dump"):
            raw_response = dict(response.model_dump(mode="json"))
        elif isinstance(response, dict):
            raw_response = response
        else:
            raw_response = {"response_repr": repr(response)}

        content = extract_content_from_raw_response(raw_response)
        if not content:
            anchors[hadm_id] = []
            continue

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            anchors[hadm_id] = []
            continue

        phrases_obj: Any = parsed.get("phrases") if isinstance(parsed, dict) else parsed

        if not isinstance(phrases_obj, list):
            anchors[hadm_id] = []
            continue

        phrases: list[str] = []
        for item in phrases_obj:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned:
                phrases.append(cleaned)
        anchors[hadm_id] = phrases[:3]

    return anchors


def extract_anchor_phrases(
    results: list[dict[str, Any]],
    field_name: str,
    target_value: str,
    llm_client: Any,
) -> dict[int, list[str]]:
    return asyncio.run(
        _extract_anchor_phrases_async(
            results=results,
            field_name=field_name,
            target_value=target_value,
            llm_client=llm_client,
        )
    )


def _tokenize(text: str) -> list[str]:
    return [match.group(0) for match in _WORD_PATTERN.finditer(text)]


def _to_regex_pattern(phrase: str) -> str:
    escaped = re.escape(phrase).replace(r"\ ", " ")
    if phrase and phrase[0].isalnum() and phrase[-1].isalnum():
        return rf"\b{escaped}\b"
    return escaped


def derive_regex_patterns(anchor_phrases: list[str]) -> list[str]:
    cleaned_phrases = [phrase.strip() for phrase in anchor_phrases if phrase and phrase.strip()]
    if not cleaned_phrases:
        return []

    ngram_phrase_ids: dict[str, set[int]] = defaultdict(set)
    ngram_original_case: dict[str, str] = {}

    for phrase_index, phrase in enumerate(cleaned_phrases):
        tokens = _tokenize(phrase)
        if len(tokens) < 2:
            continue

        seen_in_phrase: set[str] = set()
        for n in range(2, 6):
            if len(tokens) < n:
                continue
            for start in range(0, len(tokens) - n + 1):
                original_ngram = " ".join(tokens[start : start + n])
                lower_key = original_ngram.lower()
                if lower_key in seen_in_phrase:
                    continue
                seen_in_phrase.add(lower_key)
                ngram_phrase_ids[lower_key].add(phrase_index)
                ngram_original_case.setdefault(lower_key, original_ngram)

    candidates: list[dict[str, Any]] = []
    for key, phrase_ids in ngram_phrase_ids.items():
        if len(phrase_ids) < 2:
            continue
        candidates.append(
            {
                "key": key,
                "example": ngram_original_case[key],
                "freq": len(phrase_ids),
                "token_len": len(key.split()),
            }
        )

    if not candidates:
        return []

    candidates.sort(
        key=lambda item: (
            -int(item["token_len"]),
            -len(str(item["key"])),
            -int(item["freq"]),
            str(item["key"]),
        )
    )

    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        key = str(candidate["key"])
        if any(key in str(existing["key"]) for existing in deduped):
            continue
        deduped.append(candidate)

    deduped.sort(
        key=lambda item: (
            -int(item["freq"]),
            -int(item["token_len"]),
            str(item["key"]),
        )
    )

    output_patterns: list[str] = []
    for candidate in deduped:
        pattern = _to_regex_pattern(str(candidate["example"]))
        if pattern not in output_patterns:
            output_patterns.append(pattern)
        if len(output_patterns) >= 15:
            break

    return output_patterns


def derive_embedding_seed_phrases(anchor_phrases: list[str]) -> list[str]:
    seeds: list[str] = []
    seen_prefixes: set[str] = set()

    for raw_phrase in anchor_phrases:
        phrase = raw_phrase.strip()
        if not phrase:
            continue
        words = [token.lower() for token in phrase.split() if token.strip()]
        if not words:
            continue
        prefix = " ".join(words[:5])
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        seeds.append(phrase)
        if len(seeds) >= 10:
            break

    return seeds


def write_pattern_yaml(
    field_name: str,
    target_value: str,
    regex_patterns: list[str],
    seed_phrases: list[str],
    source_run_id: str,
    n_source_notes: int,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{field_name}__{target_value}.yaml"

    payload = {
        "field_name": field_name,
        "target_value": target_value,
        "generated_from_run_id": source_run_id,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "n_source_notes": int(n_source_notes),
        "regex_patterns": list(regex_patterns),
        "embedding_seed_phrases": list(seed_phrases),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)

    return output_path
