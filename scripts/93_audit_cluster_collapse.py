"""
Analyzes audit clusters under collapsed labels.

Reads: data/optimization/audit_corpus_extended_5k.jsonl, codex_outputs/93_audit_clusters_collapse.md.
Writes: data/optimization/audit_corpus_extended_5k.jsonl, codex_outputs/93_audit_clusters_collapse.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/93_audit_cluster_collapse.py` unless the script's argparse help says otherwise.
"""

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.audit_corpus import summarize_disagreement_pattern
from src.optimization.pattern_clustering import PatternCluster, cluster_corpus

NO_ND_CLUSTER_RE = re.compile(r"^[abc]_overasserts_(no|not_documented)$")
VARIANT_SURVIVAL_RE = re.compile(r"^[abc]_overasserts_(yes|not_yes)$")


@dataclass
class TriStateCollapseStats:
    field: str
    original_disagreement_count: int
    collapsed_disagreement_count: int
    original_example_disagreements: int
    collapsed_example_disagreements: int
    survival_rate: float


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if value is None:
        return ""
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        rendered = [
            _format_number(row.get(column, "")).replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *lines])


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            records.append(loaded)
    return records


def _normalize_vote(value: Any) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower() if isinstance(value, str) else str(value).strip().lower()
    if normalized == "none":
        return None
    return normalized


def _collapse_vote(value: str | None) -> str | None:
    if value is None:
        return None
    if value == "yes":
        return "yes"
    if value in {"no", "not_documented"}:
        return "not_yes"
    return value


def _is_disagreement(votes: dict[str, str | None]) -> bool:
    return len({votes["a"], votes["b"], votes["c"]}) > 1


def _render_vote(value: str | None) -> str:
    return "null" if value is None else value


def _collapse_tristate_record_from_examples(
    record: dict[str, Any],
) -> tuple[dict[str, Any], TriStateCollapseStats]:
    collapsed_record = copy.deepcopy(record)

    original_examples = list(record.get("examples", []))
    collapsed_examples: list[dict[str, Any]] = []
    original_disagreement_examples = 0
    collapsed_disagreement_examples = 0
    collapsed_disagreements_for_summary: list[dict[str, Any]] = []

    for example in original_examples:
        votes = dict(example.get("votes", {}))
        original_votes = {
            "a": _normalize_vote(votes.get("a")),
            "b": _normalize_vote(votes.get("b")),
            "c": _normalize_vote(votes.get("c")),
        }
        collapsed_votes = {k: _collapse_vote(v) for k, v in original_votes.items()}

        if _is_disagreement(original_votes):
            original_disagreement_examples += 1
        if _is_disagreement(collapsed_votes):
            collapsed_disagreement_examples += 1
            collapsed_disagreements_for_summary.append(
                {
                    "votes": {
                        "a": _render_vote(collapsed_votes["a"]),
                        "b": _render_vote(collapsed_votes["b"]),
                        "c": _render_vote(collapsed_votes["c"]),
                    }
                }
            )

        transformed = dict(example)
        transformed["votes"] = {
            "a": _render_vote(collapsed_votes["a"]),
            "b": _render_vote(collapsed_votes["b"]),
            "c": _render_vote(collapsed_votes["c"]),
        }
        collapsed_examples.append(transformed)

    original_count = int(record.get("disagreement_count", 0))
    if original_disagreement_examples > 0:
        survival_rate = collapsed_disagreement_examples / original_disagreement_examples
    else:
        survival_rate = 0.0

    collapsed_count = round(original_count * survival_rate)
    collapsed_count = max(0, min(original_count, collapsed_count))

    if collapsed_count == 0:
        collapsed_summary = "No disagreements observed after collapse."
    else:
        collapsed_summary = summarize_disagreement_pattern(collapsed_disagreements_for_summary)

    collapsed_record["examples"] = collapsed_examples
    collapsed_record["disagreement_count"] = collapsed_count
    collapsed_record["disagreement_pattern_summary"] = collapsed_summary

    stats = TriStateCollapseStats(
        field=str(record.get("field", "")),
        original_disagreement_count=original_count,
        collapsed_disagreement_count=collapsed_count,
        original_example_disagreements=original_disagreement_examples,
        collapsed_example_disagreements=collapsed_disagreement_examples,
        survival_rate=survival_rate,
    )
    return collapsed_record, stats


def _cluster_rows(clusters: list[PatternCluster]) -> list[dict[str, Any]]:
    total = sum(c.total_disagreement_count for c in clusters)
    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        share = (cluster.total_disagreement_count / total * 100.0) if total > 0 else 0.0
        rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "cluster_label": cluster.cluster_label,
                "affected_variant": cluster.affected_variant or "",
                "total_disagreements": cluster.total_disagreement_count,
                "share_pct": share,
                "n_member_fields": len(cluster.member_fields),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute audit cluster composition under TriState collapse "
            "(no + not_documented -> not_yes)."
        )
    )
    parser.add_argument(
        "--audit-path",
        default="data/optimization/audit_corpus_extended_5k.jsonl",
    )
    parser.add_argument(
        "--output",
        default="codex_outputs/93_audit_clusters_collapse.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    audit_path = Path(args.audit_path)
    output_path = Path(args.output)

    original_records = _load_jsonl(audit_path)
    original_clusters = cluster_corpus(original_records)

    collapsed_records: list[dict[str, Any]] = []
    tristate_stats: list[TriStateCollapseStats] = []

    for record in original_records:
        if str(record.get("field_type", "")) != "tristate":
            collapsed_records.append(copy.deepcopy(record))
            continue

        collapsed_record, stats = _collapse_tristate_record_from_examples(record)
        collapsed_records.append(collapsed_record)
        tristate_stats.append(stats)

    collapsed_records_with_disagreements = [
        record for record in collapsed_records if int(record.get("disagreement_count", 0)) > 0
    ]
    collapsed_clusters = cluster_corpus(collapsed_records_with_disagreements)

    original_total = int(sum(int(r.get("disagreement_count", 0)) for r in original_records))
    collapsed_total = int(
        sum(int(r.get("disagreement_count", 0)) for r in collapsed_records_with_disagreements)
    )

    original_tristate_total = int(
        sum(int(s.original_disagreement_count) for s in tristate_stats)
    )
    collapsed_tristate_total = int(
        sum(int(s.collapsed_disagreement_count) for s in tristate_stats)
    )

    dissolved_total_pct = (
        (original_total - collapsed_total) / original_total * 100.0 if original_total else 0.0
    )
    dissolved_tristate_pct = (
        (original_tristate_total - collapsed_tristate_total) / original_tristate_total * 100.0
        if original_tristate_total
        else 0.0
    )

    collapsed_by_field_target = {
        (str(record.get("field", "")), str(record.get("target_value"))): int(
            record.get("disagreement_count", 0)
        )
        for record in collapsed_records
    }
    record_type_by_field_target = {
        (str(record.get("field", "")), str(record.get("target_value"))): str(
            record.get("field_type", "")
        )
        for record in collapsed_records
    }

    no_nd_cluster_rows: list[dict[str, Any]] = []
    no_nd_orig_total = 0
    no_nd_collapsed_total = 0

    for cluster in original_clusters:
        if not NO_ND_CLUSTER_RE.match(cluster.cluster_id):
            continue

        collapsed_cluster_total = 0
        for member in cluster.member_fields:
            key = (str(member.get("field", "")), str(member.get("target_value")))
            collapsed_cluster_total += int(collapsed_by_field_target.get(key, 0))

        orig_total_cluster = int(cluster.total_disagreement_count)
        dissolved_pct = (
            (orig_total_cluster - collapsed_cluster_total) / orig_total_cluster * 100.0
            if orig_total_cluster
            else 0.0
        )

        no_nd_cluster_rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "orig_total_disagreements": orig_total_cluster,
                "collapsed_total_disagreements": collapsed_cluster_total,
                "dissolved_pct": dissolved_pct,
                "n_member_fields": len(cluster.member_fields),
            }
        )

        no_nd_orig_total += orig_total_cluster
        no_nd_collapsed_total += collapsed_cluster_total

    no_nd_dissolved_pct = (
        (no_nd_orig_total - no_nd_collapsed_total) / no_nd_orig_total * 100.0
        if no_nd_orig_total
        else 0.0
    )

    surviving_variant_specific_rows: list[dict[str, Any]] = []
    for cluster in collapsed_clusters:
        if not VARIANT_SURVIVAL_RE.match(cluster.cluster_id):
            continue

        tri_member_fields = []
        for member in cluster.member_fields:
            key = (str(member.get("field", "")), str(member.get("target_value")))
            if record_type_by_field_target.get(key) == "tristate":
                tri_member_fields.append(str(member.get("field", "")))

        if not tri_member_fields:
            continue

        surviving_variant_specific_rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "affected_variant": cluster.affected_variant or "",
                "total_disagreements": cluster.total_disagreement_count,
                "n_tristate_member_fields": len(tri_member_fields),
                "tristate_members": ", ".join(sorted(set(tri_member_fields))),
            }
        )

    original_rows = _cluster_rows(original_clusters)
    collapsed_rows = _cluster_rows(collapsed_clusters)

    top_tristate_shift_rows = sorted(
        (
            {
                "field": stat.field,
                "original_disagreements": stat.original_disagreement_count,
                "collapsed_disagreements": stat.collapsed_disagreement_count,
                "dissolved_pct": (
                    (stat.original_disagreement_count - stat.collapsed_disagreement_count)
                    / stat.original_disagreement_count
                    * 100.0
                    if stat.original_disagreement_count
                    else 0.0
                ),
                "example_survival_rate_pct": stat.survival_rate * 100.0,
            }
            for stat in tristate_stats
        ),
        key=lambda row: cast(float, row["dissolved_pct"]),
        reverse=True,
    )

    lines = [
        "# Audit Cluster Collapse (Task 3)",
        "",
        "## Metadata",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "input_audit_corpus": str(audit_path),
                    "records_in_input": len(original_records),
                    "records_after_collapse_with_disagreement": len(
                        collapsed_records_with_disagreements
                    ),
                }
            ],
            [
                "timestamp_utc",
                "input_audit_corpus",
                "records_in_input",
                "records_after_collapse_with_disagreement",
            ],
        ),
        "",
        "## Methods (concise)",
        (
            "- Collapse mapping for TriState votes: `yes -> yes`, `no -> not_yes`, "
            "`not_documented -> not_yes`, `null -> null`."
        ),
        "- Non-TriState records were left unchanged.",
        (
            "- For each TriState record, representative example votes in `examples` were "
            "transformed, disagreement survival was measured on transformed examples, and "
            "the field-level disagreement count was scaled by that survival rate."
        ),
        (
            "- Cluster IDs were recomputed from post-collapse "
            "`disagreement_pattern_summary` using the same Phase 3f logic."
        ),
        (
            "- Example transformations: `(no, not_documented, not_documented) -> "
            "(not_yes, not_yes, not_yes)` (dissolves), `(yes, no, not_documented) -> "
            "(yes, not_yes, not_yes)` (survives), `(null, no, not_documented) -> "
            "(null, not_yes, not_yes)` (null preserved)."
        ),
        "",
        "## Headline",
        (
            f"- Total disagreements: {original_total} -> {collapsed_total} "
            f"({dissolved_total_pct:.2f}% dissolved)."
        ),
        (
            f"- TriState disagreements: {original_tristate_total} -> {collapsed_tristate_total} "
            f"({dissolved_tristate_pct:.2f}% dissolved)."
        ),
        (
            f"- No/not_documented-driven clusters combined: {no_nd_orig_total} -> "
            f"{no_nd_collapsed_total} ({no_nd_dissolved_pct:.2f}% dissolved)."
        ),
        "",
        "## Original Cluster Table",
        _markdown_table(
            original_rows,
            [
                "cluster_id",
                "cluster_label",
                "affected_variant",
                "total_disagreements",
                "share_pct",
                "n_member_fields",
            ],
        ),
        "",
        "## Collapsed Cluster Table",
        _markdown_table(
            collapsed_rows,
            [
                "cluster_id",
                "cluster_label",
                "affected_variant",
                "total_disagreements",
                "share_pct",
                "n_member_fields",
            ],
        ),
        "",
        "## No/Not_Documented Cluster Dissolve",
        _markdown_table(
            no_nd_cluster_rows,
            [
                "cluster_id",
                "orig_total_disagreements",
                "collapsed_total_disagreements",
                "dissolved_pct",
                "n_member_fields",
            ],
        ),
        "",
        "## Surviving TriState Variant-Specific Patterns (Yes vs Not_Yes)",
        _markdown_table(
            surviving_variant_specific_rows,
            [
                "cluster_id",
                "affected_variant",
                "total_disagreements",
                "n_tristate_member_fields",
                "tristate_members",
            ],
        ),
        "",
        "## TriState Field Impact (Estimated from transformed examples)",
        _markdown_table(
            top_tristate_shift_rows,
            [
                "field",
                "original_disagreements",
                "collapsed_disagreements",
                "dissolved_pct",
                "example_survival_rate_pct",
            ],
        ),
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote collapse report to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
