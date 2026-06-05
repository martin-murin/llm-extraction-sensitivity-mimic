from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any

import tiktoken
from pydantic import ValidationError

from src import config
from src.llm.client import LLMClient
from src.schema.fields import LLMNoteFeatures
from src.schema.vocabulary import ADMISSION_REASON_TAGS

logger = logging.getLogger(__name__)

_ENCODING_CACHE: tuple[tiktoken.Encoding, str] | None = None


class PromptSchemaDriftError(Exception):
    """Raised when the prompt template is missing one or more schema vocabulary tags."""


@dataclass
class ExtractionResult:
    features: LLMNoteFeatures | None
    raw_response: dict[str, Any]
    parse_error: str | None
    input_tokens: int
    output_tokens: int
    latency_seconds: float


@lru_cache(maxsize=8)
def load_prompt_template(variant: str) -> str:
    prompt_path = config.REPO_ROOT / "src" / "schema" / "prompts" / f"variant_{variant}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8")
    if not prompt_text.strip():
        raise FileNotFoundError(f"Prompt template is empty: {prompt_path}")

    return prompt_text


def check_prompt_vocabulary_sync(prompt_text: str) -> None:
    missing = [tag for tag in ADMISSION_REASON_TAGS if tag not in prompt_text]
    if missing:
        raise PromptSchemaDriftError(
            "Prompt template is missing vocabulary tags: " + ", ".join(sorted(missing))
        )


def _normalize_strict_schema_node(node: Any, stats: dict[str, int]) -> None:
    if isinstance(node, dict):
        is_object = node.get("type") == "object" or "properties" in node
        if is_object:
            stats["object_nodes"] += 1

            properties = node.get("properties")
            property_names = list(properties.keys()) if isinstance(properties, dict) else []
            required = node.get("required")
            if required != property_names:
                node["required"] = property_names
                stats["required_adjustments"] += 1

            if node.get("additionalProperties") is not False:
                node["additionalProperties"] = False
                stats["additional_properties_set"] += 1

        for value in node.values():
            _normalize_strict_schema_node(value, stats)
        return

    if isinstance(node, list):
        for item in node:
            _normalize_strict_schema_node(item, stats)


@lru_cache(maxsize=1)
def build_strict_json_schema() -> dict[str, Any]:
    schema = deepcopy(LLMNoteFeatures.model_json_schema())
    stats = {
        "object_nodes": 0,
        "required_adjustments": 0,
        "additional_properties_set": 0,
    }
    _normalize_strict_schema_node(schema, stats)
    if stats["required_adjustments"] or stats["additional_properties_set"]:
        logger.info(
            "Applied strict JSON schema normalization: %s",
            stats,
        )
    return schema


@lru_cache(maxsize=1)
def build_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "LLMNoteFeatures",
            "strict": True,
            "schema": build_strict_json_schema(),
        },
    }


_REASONING_ON = (
    "# Reasoning field\n\n"
    "In your JSON output, include a non-null `reasoning` field (max 2000 chars) with a brief "
    "rationale. For each field group (admission reasons, clinical flags, SDOH, discharge), quote "
    "the specific note excerpt that drove your choice. This helps our evaluation team trace your "
    "answers. Keep it compact — one short sentence per group is enough."
)

_REASONING_OFF = (
    "# Reasoning field\n\n"
    "Set the `reasoning` field to null. Do not include any explanation. This is production mode — "
    "only the structured data is needed."
)


def build_messages(
    note_text: str,
    variant: str = "a",
    include_reasoning: bool = True,
) -> list[dict[str, str]]:
    prompt_template = load_prompt_template(variant)
    check_prompt_vocabulary_sync(prompt_template)

    reasoning_instructions = _REASONING_ON if include_reasoning else _REASONING_OFF
    system_message = prompt_template.replace("{{REASONING_INSTRUCTIONS}}", reasoning_instructions)

    assert "{{REASONING_INSTRUCTIONS}}" not in system_message

    user_message = (
        "# Discharge note to analyze\n\n"
        "---\n"
        f"{note_text}\n"
        "---\n\n"
        "Return the JSON object now."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _resolve_encoding() -> tuple[tiktoken.Encoding, str]:
    global _ENCODING_CACHE
    if _ENCODING_CACHE is not None:
        return _ENCODING_CACHE

    for model_name in (config.MODEL_ID, "gpt-5.4-nano"):
        try:
            _ENCODING_CACHE = (tiktoken.encoding_for_model(model_name), model_name)
            return _ENCODING_CACHE
        except KeyError:
            continue

    _ENCODING_CACHE = (tiktoken.get_encoding("o200k_base"), "o200k_base")
    logger.info(
        "Using tiktoken encoding `o200k_base` as fallback for model `%s`.",
        config.MODEL_ID,
    )
    return _ENCODING_CACHE


def count_prompt_tokens(messages: list[dict[str, str]]) -> int:
    encoding, _ = _resolve_encoding()
    content_tokens = sum(
        len(encoding.encode(str(message.get("content", ""))))
        for message in messages
    )
    framing_overhead = 4 * len(messages)
    return int(content_tokens + framing_overhead)


def _response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return dict(response.model_dump(mode="json"))
    if isinstance(response, dict):
        return response
    return {"response_repr": repr(response)}


def extract_content_from_raw_response(raw_response: dict[str, Any]) -> str:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
        return "".join(chunks)
    return ""


async def extract_note(
    note_text: str,
    client: LLMClient,
    variant: str = "a",
    include_reasoning: bool = True,
) -> ExtractionResult:
    messages = build_messages(
        note_text=note_text,
        variant=variant,
        include_reasoning=include_reasoning,
    )

    max_completion_tokens = 1200 if include_reasoning else 800
    start = perf_counter()
    response = await client.chat(
        messages=messages,
        response_format=build_response_format(),
        max_completion_tokens=max_completion_tokens,
    )
    latency_seconds = perf_counter() - start

    raw_response = _response_to_dict(response)
    usage = raw_response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)

    content = extract_content_from_raw_response(raw_response)
    if not content:
        return ExtractionResult(
            features=None,
            raw_response=raw_response,
            parse_error="No response content returned by model.",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency_seconds,
        )

    try:
        parsed_json = json.loads(content)
        features = LLMNoteFeatures.model_validate(parsed_json)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        return ExtractionResult(
            features=None,
            raw_response=raw_response,
            parse_error=f"{type(exc).__name__}: {exc}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency_seconds,
        )

    return ExtractionResult(
        features=features,
        raw_response=raw_response,
        parse_error=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency_seconds,
    )
