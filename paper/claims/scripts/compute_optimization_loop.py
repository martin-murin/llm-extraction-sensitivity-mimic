from __future__ import annotations

# Release documentation:
# Computes claim-registry values for optimization loop.
#
# Reads: data/raw_responses/refinement_v1_a/results.jsonl, data/raw_responses/refinement_v1_b/results.jsonl, data/raw_responses/refinement_v3_c/results.jsonl.
# Writes: data/raw_responses/refinement_v1_a/results.jsonl, data/raw_responses/refinement_v1_b/results.jsonl, data/raw_responses/refinement_v3_c/results.jsonl.
# Backs paper claim registry entries for optimization loop.

from pathlib import Path

from paper.claims.scripts._common import claim_entry, require_input_files
from paper.claims.scripts._optimization_metrics_primary import (
    compute_optimization_primary_metrics,
)
from paper.claims.scripts._receipt import build_receipt, merge_into_claims_json, now_utc_iso

CLAIMS_PATH = Path(__file__).resolve().parent.parent / "claims.json"
SCRIPT_PATH = Path(__file__).resolve()

INPUT_FILES = [
    "logs/optimization/iteration_1.json",
    "logs/optimization/iteration_2.json",
    "data/raw_responses/refinement_v1_a/results.jsonl",
    "data/raw_responses/refinement_v1_b/results.jsonl",
    "data/raw_responses/refinement_v3_c/results.jsonl",
    "src/optimization/audit_corpus.py",
    "src/optimization/pattern_clustering.py",
    "src/schema/vocabulary.py",
    "src/utils/threeway_kappa.py",
]


def compute_all() -> dict:
    require_input_files(INPUT_FILES)
    metrics = compute_optimization_primary_metrics()

    timestamp = now_utc_iso()
    receipt = build_receipt(SCRIPT_PATH, "compute_all", INPUT_FILES)
    return {
        "optimization_loop_iterations": claim_entry(
            value=metrics.iterations_applied,
            format_default="d",
            description="Number of applied optimization iterations in Phase 3f loop",
            sample="refinement_150",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "optimization_loop_initial_cluster_disagreements": claim_entry(
            value=metrics.initial_cluster_disagreements,
            format_default=",d",
            description="Initial targeted-cluster disagreement volume before optimization loop",
            sample="refinement_150",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "optimization_loop_final_cluster_disagreements": claim_entry(
            value=metrics.final_cluster_disagreements,
            format_default=",d",
            description=(
                "Final targeted-cluster disagreement volume after optimization loop, "
                "recomputed from final run_ids using pattern-clustering logic"
            ),
            sample="refinement_150",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "optimization_loop_cluster_reduction_pct": claim_entry(
            value=metrics.reduction_pct,
            format_default=".2f",
            unit="%",
            description=(
                "Relative reduction in targeted-cluster disagreement volume "
                "across optimization loop"
            ),
            sample="refinement_150",
            computed_at=timestamp,
            receipt=receipt,
        ),
    }


def main() -> int:
    new_claims = compute_all()
    n = merge_into_claims_json(CLAIMS_PATH, new_claims)
    print(f"Updated {n} claims in {CLAIMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
