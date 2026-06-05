from __future__ import annotations

from src.optimization.audit_corpus import (
    has_any_disagreement,
    select_representative_examples,
    should_include_record,
    summarize_disagreement_pattern,
)


def test_disagreement_pattern_summary_identifies_gt_60_percent_outlier() -> None:
    disagreements = [
        {"votes": {"a": "not_documented", "b": "not_documented", "c": "no"}},
        {"votes": {"a": "not_documented", "b": "not_documented", "c": "no"}},
        {"votes": {"a": "not_documented", "b": "not_documented", "c": "no"}},
        {"votes": {"a": "yes", "b": "yes", "c": "no"}},
        {"votes": {"a": "yes", "b": "yes", "c": "no"}},
    ]
    summary = summarize_disagreement_pattern(disagreements)
    assert (
        summary
        == "C votes 'no' where the other two vote 'not_documented' on 5 of 5 disagreements."
    )


def test_disagreement_pattern_summary_handles_even_distribution() -> None:
    disagreements = [
        {"votes": {"a": "yes", "b": "yes", "c": "no"}},
        {"votes": {"a": "yes", "b": "no", "c": "yes"}},
        {"votes": {"a": "no", "b": "yes", "c": "yes"}},
    ]
    summary = summarize_disagreement_pattern(disagreements)
    assert summary == "Disagreements distributed across all three variants (no consistent outlier)."


def test_audit_generation_skips_when_all_variants_agree() -> None:
    vote_tuples = [("yes", "yes", "yes"), ("not_documented", "not_documented", "not_documented")]
    assert has_any_disagreement(vote_tuples) is False


def test_example_selection_caps_at_ten_examples_per_field() -> None:
    cases = [
        {
            "hadm_id": 1000 + idx,
            "pattern_key": "a=b!=c" if idx % 2 == 0 else "a=c!=b",
            "chapter": f"Chapter-{idx % 3}",
        }
        for idx in range(20)
    ]
    selected = select_representative_examples(cases, max_examples=10)
    assert len(selected) == 10


def test_low_base_rate_records_excluded_when_flag_false() -> None:
    assert should_include_record(9, include_low_base_rate=False) is False
    assert should_include_record(10, include_low_base_rate=False) is True
    assert should_include_record(1, include_low_base_rate=True) is True
