from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.labeling_functions.base import LFInput, LFOutput, LabelingFunction, Vote
from src.labeling_functions.section_parser import get_section, parse_sections
from src.schema.section_map import FIELD_SECTION_MAP

logger = logging.getLogger(__name__)

NEGATION_CUES: tuple[str, ...] = (
    "no ",
    "not ",
    "denies",
    "denied",
    "without",
    "neg for ",
    "negative for ",
)
NEGATION_WINDOW_CHARS: int = 30


def eval_compound_pattern(
    text: str,
    all_of: list[str],
    window_chars: int,
) -> tuple[int, int, str] | None:
    """Return (match_start, match_end, evidence_excerpt) if all regexes match in a window."""
    position_lists: list[list[int]] = []
    for pattern in all_of:
        matches = [match.start() for match in re.finditer(pattern, text, re.IGNORECASE)]
        if not matches:
            return None
        position_lists.append(matches)

    best: tuple[int, int] | None = None
    for combo in product(*position_lists):
        span = max(combo) - min(combo)
        if span <= window_chars:
            start, end = min(combo), max(combo)
            if best is None or (end - start) < (best[1] - best[0]):
                best = (start, end)

    if best is None:
        return None

    start, end = best
    excerpt_start = max(0, start - 10)
    excerpt_end = min(len(text), end + 30)
    excerpt = text[excerpt_start:excerpt_end]
    if len(excerpt) > 200:
        excerpt = excerpt[:200]
    return (start, end, excerpt)


def _validate_compound_patterns(
    raw_value: Any,
    *,
    path_display: str,
) -> list[dict[str, Any]]:
    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        raise ValueError(f"Pattern YAML compound_patterns must be a list: {path_display}")

    validated: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_value):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Pattern YAML compound_patterns[{index}] must be a mapping: {path_display}"
            )
        if "all_of" not in entry:
            raise ValueError(
                f"Pattern YAML compound_patterns[{index}] missing 'all_of': {path_display}"
            )
        if "window_chars" not in entry:
            raise ValueError(
                f"Pattern YAML compound_patterns[{index}] missing 'window_chars': {path_display}"
            )

        all_of = entry["all_of"]
        window_chars = entry["window_chars"]
        description = entry.get("description")

        if (
            not isinstance(all_of, list)
            or not all_of
            or any(not isinstance(item, str) for item in all_of)
        ):
            raise ValueError(
                f"Pattern YAML compound_patterns[{index}].all_of must be a non-empty list[str]: "
                f"{path_display}"
            )
        if not isinstance(window_chars, int):
            raise ValueError(
                f"Pattern YAML compound_patterns[{index}].window_chars must be int: "
                f"{path_display}"
            )
        if description is not None and not isinstance(description, str):
            raise ValueError(
                f"Pattern YAML compound_patterns[{index}].description must be str if present: "
                f"{path_display}"
            )

        validated.append(
            {
                "all_of": [str(item) for item in all_of],
                "window_chars": int(window_chars),
                "description": str(description) if description is not None else None,
            }
        )

    return validated


def load_pattern_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pattern YAML does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Pattern YAML must contain a mapping: {path}")

    required = {"field_name", "target_value", "regex_patterns"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ValueError(f"Pattern YAML missing keys {missing}: {path}")

    patterns = payload.get("regex_patterns")
    if not isinstance(patterns, list):
        raise ValueError(f"Pattern YAML regex_patterns must be a list: {path}")

    if any(not isinstance(item, str) for item in patterns):
        raise ValueError(f"Pattern YAML regex_patterns must contain only strings: {path}")

    payload["regex_patterns"] = [str(item) for item in patterns]
    payload["compound_patterns"] = _validate_compound_patterns(
        payload.get("compound_patterns", []),
        path_display=str(path),
    )
    return payload


def is_negated(text: str, match_start: int) -> bool:
    window_start = max(0, match_start - NEGATION_WINDOW_CHARS)
    context = text[window_start:match_start].lower()
    return any(cue in context for cue in NEGATION_CUES)


def _pattern_flags(pattern: str) -> re.RegexFlag:
    if "(?-i)" in pattern:
        return re.NOFLAG
    return re.IGNORECASE


@dataclass
class RegexLabelingFunction:
    name: str
    target_field: str
    target_value: str | None
    patterns: list[str]
    compound_patterns: list[dict[str, Any]]
    compiled_patterns: list[tuple[str, re.Pattern[str]]]
    pattern_source: str | None = None

    def __call__(self, inputs: LFInput) -> LFOutput:
        sections = (
            inputs.sections
            if inputs.sections is not None
            else parse_sections(inputs.note_text)
        )

        search_chunks: list[str] = []
        full_note = sections.get("__full_note__")
        if isinstance(full_note, str) and full_note:
            search_chunks.append(full_note)
        else:
            for canonical_name in FIELD_SECTION_MAP[self.target_field]:
                section_text = get_section(sections, canonical_name)
                if section_text:
                    search_chunks.append(section_text)

        if not search_chunks:
            return LFOutput(vote=Vote.ABSTAIN, evidence="no_section")

        search_text = "\n---\n".join(search_chunks)
        for pattern_text, compiled in self.compiled_patterns:
            for match in compiled.finditer(search_text):
                if is_negated(search_text, match.start()):
                    continue
                return LFOutput(vote=Vote.POSITIVE, evidence=f"regex match: {pattern_text}")

        for compound_pattern in self.compound_patterns:
            all_of = [str(item) for item in compound_pattern["all_of"]]
            window_chars = int(compound_pattern["window_chars"])
            description_raw = compound_pattern.get("description")
            description = str(description_raw) if description_raw is not None else str(all_of)

            match_info = eval_compound_pattern(search_text, all_of, window_chars)
            if match_info is None:
                continue
            match_start, _, _ = match_info
            if is_negated(search_text, match_start):
                continue
            return LFOutput(vote=Vote.POSITIVE, evidence=f"compound match: {description}")

        return LFOutput(vote=Vote.ABSTAIN)


def build_regex_lf(
    field_name: str,
    target_value: str,
    patterns: list[str],
    compound_patterns: list[dict[str, Any]] | None = None,
) -> RegexLabelingFunction:
    if field_name not in FIELD_SECTION_MAP:
        raise KeyError(f"Unknown field for regex LF: {field_name}")

    compiled_patterns = [
        (pattern, re.compile(pattern, flags=_pattern_flags(pattern)))
        for pattern in patterns
    ]
    normalized_compound = _validate_compound_patterns(
        compound_patterns if compound_patterns is not None else [],
        path_display=f"<runtime:{field_name}>",
    )

    return RegexLabelingFunction(
        name=f"regex_{field_name}_{target_value}",
        target_field=field_name,
        target_value=target_value,
        patterns=list(patterns),
        compound_patterns=normalized_compound,
        compiled_patterns=compiled_patterns,
        pattern_source=None,
    )


def build_all_regex_lfs(patterns_dir: Path) -> list[LabelingFunction]:
    if not patterns_dir.exists():
        logger.info("Patterns directory does not exist: %s", patterns_dir)
        return []

    lfs: list[LabelingFunction] = []
    for yaml_path in sorted(patterns_dir.glob("*.yaml")):
        payload = load_pattern_yaml(yaml_path)
        field_name = str(payload["field_name"])
        target_value = str(payload["target_value"])
        patterns = [str(item) for item in payload.get("regex_patterns", [])]
        compound_patterns = list(payload.get("compound_patterns", []))
        lf = build_regex_lf(
            field_name=field_name,
            target_value=target_value,
            patterns=patterns,
            compound_patterns=compound_patterns,
        )
        lf.pattern_source = yaml_path.name
        lfs.append(lf)

    logger.info("Loaded %s regex LFs from %s", len(lfs), patterns_dir)
    return lfs
