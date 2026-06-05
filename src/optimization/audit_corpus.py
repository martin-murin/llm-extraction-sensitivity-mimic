from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VotePattern:
    pattern_key: str
    outlier_variant: str | None
    consensus_value: str | None
    outlier_value: str | None


def classify_vote_pattern(votes: dict[str, str]) -> VotePattern:
    a = str(votes["a"])
    b = str(votes["b"])
    c = str(votes["c"])

    if a == b == c:
        return VotePattern("all_equal", None, a, None)
    if a == b and c != a:
        return VotePattern("a=b!=c", "c", a, c)
    if a == c and b != a:
        return VotePattern("a=c!=b", "b", a, b)
    if b == c and a != b:
        return VotePattern("b=c!=a", "a", b, a)
    return VotePattern("all_different", None, None, None)


def summarize_disagreement_pattern(disagreements: list[dict[str, Any]]) -> str:
    if not disagreements:
        return "No disagreements observed."

    outlier_counts: Counter[str] = Counter()
    phrase_counts: Counter[tuple[str, str, str]] = Counter()

    for item in disagreements:
        votes = {
            "a": str(item["votes"]["a"]),
            "b": str(item["votes"]["b"]),
            "c": str(item["votes"]["c"]),
        }
        pattern = classify_vote_pattern(votes)
        if pattern.outlier_variant is not None:
            outlier_counts[pattern.outlier_variant] += 1
            assert pattern.consensus_value is not None
            assert pattern.outlier_value is not None
            phrase_counts[
                (pattern.outlier_variant, pattern.consensus_value, pattern.outlier_value)
            ] += 1

    total = len(disagreements)
    if outlier_counts:
        outlier_variant, outlier_count = outlier_counts.most_common(1)[0]
        if outlier_count / total > 0.6:
            candidate_phrases = [
                (key, count)
                for key, count in phrase_counts.items()
                if key[0] == outlier_variant
            ]
            if candidate_phrases:
                (variant, consensus_value, outlier_value), _ = sorted(
                    candidate_phrases,
                    key=lambda x: x[1],
                    reverse=True,
                )[0]
                return (
                    f"{variant.upper()} votes '{outlier_value}' where the other two vote "
                    f"'{consensus_value}' on {outlier_count} of {total} disagreements."
                )
            return (
                f"{outlier_variant.upper()} is the outlier on {outlier_count} of {total} "
                "disagreements."
            )

    if len(outlier_counts) == 3:
        values = [outlier_counts["a"], outlier_counts["b"], outlier_counts["c"]]
        if max(values) - min(values) <= 1:
            return "Disagreements distributed across all three variants (no consistent outlier)."

    return "No consistent outlier; disagreements vary by case."


def has_any_disagreement(vote_tuples: list[tuple[str, str, str]]) -> bool:
    return any(len({a, b, c}) > 1 for a, b, c in vote_tuples)


def should_include_record(n_positive_total: int, include_low_base_rate: bool) -> bool:
    if include_low_base_rate:
        return True
    return n_positive_total >= 10


def select_representative_examples(
    cases: list[dict[str, Any]],
    max_examples: int = 10,
) -> list[dict[str, Any]]:
    if max_examples <= 0 or not cases:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in sorted(cases, key=lambda row: int(row["hadm_id"])):
        grouped.setdefault(str(case["pattern_key"]), []).append(case)

    sorted_patterns = sorted(
        grouped.keys(),
        key=lambda key: (-len(grouped[key]), key),
    )[:5]

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    used_chapters: set[str] = set()

    def pop_next(pattern_key: str) -> dict[str, Any] | None:
        bucket = grouped.get(pattern_key, [])
        if not bucket:
            return None

        chapter_idx = None
        for idx, item in enumerate(bucket):
            chapter = str(item.get("chapter", ""))
            if chapter and chapter not in used_chapters:
                chapter_idx = idx
                break
        if chapter_idx is None:
            chapter_idx = 0

        chosen = bucket.pop(chapter_idx)
        return chosen

    while len(selected) < max_examples:
        progressed = False
        for pattern_key in sorted_patterns:
            if len(selected) >= max_examples:
                break
            next_case = pop_next(pattern_key)
            if next_case is None:
                continue
            hadm_id = int(next_case["hadm_id"])
            if hadm_id in selected_ids:
                continue
            selected.append(next_case)
            selected_ids.add(hadm_id)
            chapter = str(next_case.get("chapter", ""))
            if chapter:
                used_chapters.add(chapter)
            progressed = True
        if not progressed:
            break

    return selected[:max_examples]
