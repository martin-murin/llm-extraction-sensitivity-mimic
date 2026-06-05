"""
Compares audit cluster composition across samples.

Reads: data/optimization/audit_corpus_v3.jsonl, data/optimization/audit_corpus_methodology_1k.jsonl, codex_outputs/22_audit_corpus_1k_vs_refinement.md.
Writes: data/optimization/audit_corpus_v3.jsonl, data/optimization/audit_corpus_methodology_1k.jsonl, codex_outputs/22_audit_corpus_1k_vs_refinement.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/23_compare_audit_clusters.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from src.optimization.pattern_clustering import cluster_corpus


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        vals = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare refinement vs methodology_1k audit clusters."
    )
    parser.add_argument("--refinement", default="data/optimization/audit_corpus_v3.jsonl")
    parser.add_argument(
        "--methodology",
        default="data/optimization/audit_corpus_methodology_1k.jsonl",
    )
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--output", default="codex_outputs/22_audit_corpus_1k_vs_refinement.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refinement_records = _load_jsonl(Path(args.refinement))
    methodology_records = _load_jsonl(Path(args.methodology))

    ref_clusters = cluster_corpus(refinement_records)
    meth_clusters = cluster_corpus(methodology_records)

    ref_sig = [c for c in ref_clusters if c.total_disagreement_count > int(args.threshold)]
    meth_sig = [c for c in meth_clusters if c.total_disagreement_count > int(args.threshold)]

    ref_ids = {c.cluster_id for c in ref_sig}
    new_clusters = [c for c in meth_sig if c.cluster_id not in ref_ids]

    ref_rows = [
        {
            "cluster_id": c.cluster_id,
            "cluster_label": c.cluster_label,
            "affected_variant": c.affected_variant or "",
            "total_disagreements": c.total_disagreement_count,
            "n_member_fields": len(c.member_fields),
        }
        for c in ref_sig
    ]
    meth_rows = [
        {
            "cluster_id": c.cluster_id,
            "cluster_label": c.cluster_label,
            "affected_variant": c.affected_variant or "",
            "total_disagreements": c.total_disagreement_count,
            "n_member_fields": len(c.member_fields),
        }
        for c in meth_sig
    ]
    new_rows = [
        {
            "cluster_id": c.cluster_id,
            "cluster_label": c.cluster_label,
            "affected_variant": c.affected_variant or "",
            "total_disagreements": c.total_disagreement_count,
            "n_member_fields": len(c.member_fields),
        }
        for c in new_clusters
    ]

    lines = [
        "# Audit Corpus Comparison (Methodology 1k vs Refinement)",
        "",
        "## Run metadata",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "refinement_path": str(Path(args.refinement)),
                    "methodology_path": str(Path(args.methodology)),
                    "threshold_disagreements": int(args.threshold),
                }
            ],
            ["timestamp_utc", "refinement_path", "methodology_path", "threshold_disagreements"],
        ),
        "",
        "## Significant clusters in refinement (> threshold)",
        _markdown_table(
            ref_rows,
            [
                "cluster_id",
                "cluster_label",
                "affected_variant",
                "total_disagreements",
                "n_member_fields",
            ],
        ),
        "",
        "## Significant clusters in methodology_1k (> threshold)",
        _markdown_table(
            meth_rows,
            [
                "cluster_id",
                "cluster_label",
                "affected_variant",
                "total_disagreements",
                "n_member_fields",
            ],
        ),
        "",
        "## New significant clusters in methodology_1k (not seen in refinement)",
        _markdown_table(
            new_rows,
            [
                "cluster_id",
                "cluster_label",
                "affected_variant",
                "total_disagreements",
                "n_member_fields",
            ],
        ),
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit-cluster comparison report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
