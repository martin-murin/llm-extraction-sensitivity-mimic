from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class PatternCluster:
    cluster_id: str
    cluster_label: str
    affected_variant: str | None
    member_fields: list[dict[str, Any]]
    total_disagreement_count: int
    representative_examples: list[dict[str, Any]]


_OUTLIER_RE = re.compile(
    r"^\s*([ABCabc])\s+votes\s+'([^']+)'\s+where\s+the\s+other\s+two",
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def _cluster_identity(summary: str) -> tuple[str, str, str | None]:
    matched = _OUTLIER_RE.search(summary)
    if matched:
        variant = matched.group(1).lower()
        voted_value = _slugify(matched.group(2))
        cluster_id = f"{variant}_overasserts_{voted_value}"
        cluster_label = f"{matched.group(1).upper()} over-asserts '{matched.group(2)}'"
        return cluster_id, cluster_label, variant

    if "No consistent outlier" in summary:
        return (
            "no_consistent_outlier",
            "No consistent outlier",
            None,
        )

    return (
        "unclassified_disagreement",
        "Unclassified disagreement pattern",
        None,
    )


def _pick_representative_examples(
    records: list[dict[str, Any]],
    *,
    max_examples: int = 8,
) -> list[dict[str, Any]]:
    if not records:
        return []

    total_disagreements = sum(int(record.get("disagreement_count", 0)) for record in records) or 1
    selected: list[dict[str, Any]] = []
    seen_hadm_ids: set[int] = set()

    quotas: list[tuple[dict[str, Any], int]] = []
    for record in records:
        weight = int(record.get("disagreement_count", 0))
        quota = max(1, round((weight / total_disagreements) * max_examples))
        quotas.append((record, quota))

    for record, quota in quotas:
        examples = list(record.get("examples", []))
        examples.sort(key=lambda item: int(item.get("hadm_id", 0)))
        taken = 0
        for item in examples:
            if len(selected) >= max_examples or taken >= quota:
                break
            hadm_id = int(item.get("hadm_id", 0))
            if hadm_id in seen_hadm_ids:
                continue
            selected.append(item)
            seen_hadm_ids.add(hadm_id)
            taken += 1

    if len(selected) < max_examples:
        for record in records:
            examples = list(record.get("examples", []))
            examples.sort(key=lambda item: int(item.get("hadm_id", 0)))
            for item in examples:
                if len(selected) >= max_examples:
                    break
                hadm_id = int(item.get("hadm_id", 0))
                if hadm_id in seen_hadm_ids:
                    continue
                selected.append(item)
                seen_hadm_ids.add(hadm_id)
            if len(selected) >= max_examples:
                break

    return selected[:max_examples]


def cluster_corpus(corpus_records: list[dict[str, Any]]) -> list[PatternCluster]:
    grouped: dict[str, dict[str, Any]] = {}

    for record in corpus_records:
        summary = str(record.get("disagreement_pattern_summary", "")).strip()
        cluster_id, cluster_label, affected_variant = _cluster_identity(summary)
        entry = grouped.setdefault(
            cluster_id,
            {
                "cluster_label": cluster_label,
                "affected_variant": affected_variant,
                "records": [],
            },
        )
        entry["records"].append(record)

    clusters: list[PatternCluster] = []
    for cluster_id, entry in grouped.items():
        records = sorted(
            list(entry["records"]),
            key=lambda row: int(row.get("disagreement_count", 0)),
            reverse=True,
        )

        member_fields: list[dict[str, Any]] = []
        total_disagreement_count = 0
        for record in records:
            field_name = str(record.get("field", ""))
            target_value = record.get("target_value")
            member_fields.append(
                {
                    "field": field_name,
                    "target_value": target_value,
                    "kappa_mean": float(record.get("kappa_mean", 0.0)),
                    "disagreement_count": int(record.get("disagreement_count", 0)),
                    "n_positive_total": int(record.get("n_positive_total", 0)),
                }
            )
            total_disagreement_count += int(record.get("disagreement_count", 0))

        representative_examples = _pick_representative_examples(records, max_examples=8)
        clusters.append(
            PatternCluster(
                cluster_id=cluster_id,
                cluster_label=str(entry["cluster_label"]),
                affected_variant=entry["affected_variant"],
                member_fields=member_fields,
                total_disagreement_count=total_disagreement_count,
                representative_examples=representative_examples,
            )
        )

    clusters.sort(key=lambda cluster: cluster.total_disagreement_count, reverse=True)
    return clusters

