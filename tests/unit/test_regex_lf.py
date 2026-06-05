from __future__ import annotations

from pathlib import Path

import yaml

from src.labeling_functions.base import LFInput, Vote
from src.labeling_functions.regex_lf import (
    build_all_regex_lfs,
    build_regex_lf,
    eval_compound_pattern,
    load_pattern_yaml,
)


def test_regex_lf_abstains_when_no_target_sections_present() -> None:
    lf = build_regex_lf("dnr_dni_documented", "yes", [r"\bdnr\b"])
    output = lf(
        LFInput(
            hadm_id=1,
            note_text="Chief complaint mentions DNR.",
            sections={"Chief Complaint": "DNR discussed in ED."},
        )
    )
    assert output.vote == Vote.ABSTAIN


def test_regex_lf_votes_positive_on_match_in_target_section() -> None:
    lf = build_regex_lf("dnr_dni_documented", "yes", [r"\bdnr\b"])
    output = lf(
        LFInput(
            hadm_id=2,
            note_text="",
            sections={"Discharge Condition": "Code status DNR/DNI confirmed."},
        )
    )
    assert output.vote == Vote.POSITIVE
    assert output.evidence == "regex match: \\bdnr\\b"


def test_regex_lf_abstains_when_match_only_in_non_target_section() -> None:
    lf = build_regex_lf("home_health_ordered", "yes", [r"\bhome health\b"])
    output = lf(
        LFInput(
            hadm_id=3,
            note_text="",
            sections={"Past Medical History": "Home health was used years ago."},
        )
    )
    assert output.vote == Vote.ABSTAIN


def test_regex_lf_negation_window_blocks_positive_vote() -> None:
    lf = build_regex_lf("dnr_dni_documented", "yes", [r"\bdnr\b"])
    output = lf(
        LFInput(
            hadm_id=4,
            note_text="",
            sections={"Discharge Condition": "Patient denies DNR status at discharge."},
        )
    )
    assert output.vote == Vote.ABSTAIN


def test_regex_lf_uses_full_note_sentinel_mode() -> None:
    lf = build_regex_lf("substance_use_active", "yes", [r"\bactive cocaine use\b"])
    output = lf(
        LFInput(
            hadm_id=5,
            note_text="",
            sections={"__full_note__": "The team documents active cocaine use this admission."},
        )
    )
    assert output.vote == Vote.POSITIVE


def test_build_all_regex_lfs_loads_one_per_yaml(tmp_path: Path) -> None:
    payloads = [
        {
            "field_name": "dnr_dni_documented",
            "target_value": "yes",
            "regex_patterns": [r"\bdnr\b"],
        },
        {
            "field_name": "home_health_ordered",
            "target_value": "yes",
            "regex_patterns": [r"\bhome health\b"],
        },
    ]

    for idx, payload in enumerate(payloads, start=1):
        path = tmp_path / f"pattern_{idx}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    lfs = build_all_regex_lfs(tmp_path)
    assert len(lfs) == 2
    assert len({lf.name for lf in lfs}) == 2


def test_eval_compound_pattern_none_when_one_regex_missing() -> None:
    text = "Patient with active alcohol use and CIWA protocol."
    result = eval_compound_pattern(text, [r"alcohol", r"opioid"], window_chars=40)
    assert result is None


def test_eval_compound_pattern_none_when_window_too_small() -> None:
    text = "alcohol use was noted. " + ("x" * 100) + "abuse was documented later."
    result = eval_compound_pattern(text, [r"alcohol", r"abuse"], window_chars=20)
    assert result is None


def test_eval_compound_pattern_returns_match_info_within_window() -> None:
    text = "Patient has active alcohol abuse this admission."
    result = eval_compound_pattern(text, [r"alcohol", r"abuse"], window_chars=30)
    assert result is not None
    start, end, excerpt = result
    assert 0 <= start <= end
    assert "alcohol" in excerpt.lower()


def test_compound_regex_lf_votes_positive_when_compound_matches() -> None:
    lf = build_regex_lf(
        "substance_use_active",
        "yes",
        patterns=[],
        compound_patterns=[
            {
                "all_of": [r"alcohol", r"abuse"],
                "window_chars": 40,
                "description": "alcohol-use compound",
            }
        ],
    )
    output = lf(
        LFInput(
            hadm_id=6,
            note_text="",
            sections={"Social History": "Patient reports active alcohol abuse."},
        )
    )
    assert output.vote == Vote.POSITIVE
    assert output.evidence == "compound match: alcohol-use compound"


def test_compound_regex_lf_respects_negation_window() -> None:
    lf = build_regex_lf(
        "substance_use_active",
        "yes",
        patterns=[],
        compound_patterns=[
            {
                "all_of": [r"alcohol", r"abuse"],
                "window_chars": 40,
                "description": "alcohol-use compound",
            }
        ],
    )
    output = lf(
        LFInput(
            hadm_id=7,
            note_text="",
            sections={"Social History": "Patient denies alcohol abuse currently."},
        )
    )
    assert output.vote == Vote.ABSTAIN


def test_load_pattern_yaml_with_compound_patterns(tmp_path: Path) -> None:
    path = tmp_path / "substance_use_active__yes.yaml"
    payload = {
        "field_name": "substance_use_active",
        "target_value": "yes",
        "regex_patterns": [r"\bCIWA\b"],
        "compound_patterns": [
            {
                "all_of": [r"alcohol", r"abuse"],
                "window_chars": 40,
                "description": "alcohol-token + use-noun",
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    loaded = load_pattern_yaml(path)
    assert loaded["field_name"] == "substance_use_active"
    assert isinstance(loaded["compound_patterns"], list)
    assert loaded["compound_patterns"][0]["window_chars"] == 40


def test_load_pattern_yaml_raises_on_malformed_compound_patterns(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    payload = {
        "field_name": "substance_use_active",
        "target_value": "yes",
        "regex_patterns": [r"\bCIWA\b"],
        "compound_patterns": [{"window_chars": 40, "description": "missing all_of"}],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    try:
        load_pattern_yaml(path)
    except ValueError as exc:
        assert "missing 'all_of'" in str(exc)
    else:
        raise AssertionError("Expected ValueError for malformed compound_patterns entry")
