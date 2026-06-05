from __future__ import annotations

import pytest

from src.llm import extractor
from src.llm.extractor import (
    PromptSchemaDriftError,
    build_messages,
    count_prompt_tokens,
    load_prompt_template,
)


def test_load_prompt_template_a_non_empty() -> None:
    prompt = load_prompt_template("a")
    assert prompt.strip()


def test_load_prompt_template_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt_template("zzz_missing")


def test_build_messages_has_two_messages_roles_ordered() -> None:
    messages = build_messages("example note")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_messages_reasoning_true_contains_non_null() -> None:
    messages = build_messages("example note", include_reasoning=True)
    assert "non-null" in messages[0]["content"]


def test_build_messages_reasoning_false_contains_set_null_phrase() -> None:
    messages = build_messages("example note", include_reasoning=False)
    assert "Set the `reasoning` field to null" in messages[0]["content"]


def test_build_messages_user_contains_note_marker() -> None:
    marker = "SYNTHETIC_NOTE_MARKER_12345"
    messages = build_messages(marker, include_reasoning=True)
    assert marker in messages[1]["content"]


@pytest.mark.parametrize("include_reasoning", [True, False])
def test_placeholder_removed_from_build_messages(include_reasoning: bool) -> None:
    messages = build_messages("example note", include_reasoning=include_reasoning)
    assert "{{REASONING_INSTRUCTIONS}}" not in messages[0]["content"]


def test_schema_drift_guard_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    bad_prompt = """# Role

Vocabulary sample includes respiratory_infection but not other tags.

{{REASONING_INSTRUCTIONS}}
"""
    monkeypatch.setattr(extractor, "load_prompt_template", lambda variant: bad_prompt)

    with pytest.raises(PromptSchemaDriftError):
        build_messages("example note")


def test_count_prompt_tokens_positive_int() -> None:
    messages = build_messages("example note", include_reasoning=True)
    tokens = count_prompt_tokens(messages)
    assert isinstance(tokens, int)
    assert tokens > 0
