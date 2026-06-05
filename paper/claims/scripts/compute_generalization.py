from __future__ import annotations

# Release documentation:
# Computes claim-registry values for generalization.
#
# Reads: data/raw_responses/*/results.jsonl, data/raw_responses/refinement_v1_a/results.jsonl, data/raw_responses/refinement_v1_b/results.jsonl, data/raw_responses/refinement_v3_c/results.jsonl, data/raw_responses/holdout_v1_a/results.jsonl, data/raw_responses/holdout_v1_b/results.jsonl.
# Writes: data/raw_responses/*/results.jsonl, data/raw_responses/refinement_v1_a/results.jsonl, data/raw_responses/refinement_v1_b/results.jsonl, data/raw_responses/refinement_v3_c/results.jsonl, data/raw_responses/holdout_v1_a/results.jsonl, data/raw_responses/holdout_v1_b/results.jsonl.
# Backs paper claim registry entries for generalization.

from pathlib import Path

from paper.claims.scripts._common import claim_entry, require_input_files
from paper.claims.scripts._receipt import build_receipt, merge_into_claims_json, now_utc_iso
from paper.claims.scripts._threeway_kappa_primary import recompute_five_sample_stability

CLAIMS_PATH = Path(__file__).resolve().parent.parent / "claims.json"
SCRIPT_PATH = Path(__file__).resolve()

INPUT_FILES = [
    "data/raw_responses/refinement_v1_a/results.jsonl",
    "data/raw_responses/refinement_v1_b/results.jsonl",
    "data/raw_responses/refinement_v3_c/results.jsonl",
    "data/raw_responses/holdout_v1_a/results.jsonl",
    "data/raw_responses/holdout_v1_b/results.jsonl",
    "data/raw_responses/holdout_v1_c/results.jsonl",
    "data/raw_responses/methodology_1k_a/results.jsonl",
    "data/raw_responses/methodology_1k_b/results.jsonl",
    "data/raw_responses/methodology_1k_c/results.jsonl",
    "data/raw_responses/methodology_5k_a_subset500/results.jsonl",
    "data/raw_responses/methodology_5k_audit_b/results.jsonl",
    "data/raw_responses/methodology_5k_audit_c/results.jsonl",
    "data/raw_responses/production_v1/results.jsonl",
    "data/raw_responses/extended_5k_b/results.jsonl",
    "data/raw_responses/extended_5k_c/results.jsonl",
    "data/splits/refinement_150.csv",
    "data/splits/holdout_150.csv",
    "data/splits/methodology_1k.csv",
    "data/splits/methodology_5k_audit_500.csv",
    "data/splits/extended_5k.csv",
]


def compute_all() -> dict:
    require_input_files(INPUT_FILES)
    summaries = recompute_five_sample_stability()

    refinement = float(summaries["refinement_150"].median_kappa_filtered)
    holdout = float(summaries["holdout_150"].median_kappa_filtered)
    meth_1k = float(summaries["methodology_1k"].median_kappa_filtered)
    meth_5k = float(summaries["methodology_5k_audit_500"].median_kappa_filtered)
    ext_5k = float(summaries["extended_5k"].median_kappa_filtered)
    delta_pp = (holdout - refinement) * 100.0

    timestamp = now_utc_iso()
    receipt = build_receipt(SCRIPT_PATH, "compute_all", INPUT_FILES)
    return {
        "refinement_150_kappa": claim_entry(
            value=refinement,
            format_default=".4f",
            description="Filtered median cross-variant kappa on refinement_150",
            sample="refinement_150",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "holdout_150_kappa": claim_entry(
            value=holdout,
            format_default=".4f",
            description="Filtered median cross-variant kappa on holdout_150",
            sample="holdout_150",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "refinement_holdout_delta_pp": claim_entry(
            value=delta_pp,
            format_default=".4f",
            unit="pp",
            description="Holdout minus refinement filtered-median kappa delta",
            sample="refinement_150+holdout_150",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "methodology_1k_kappa": claim_entry(
            value=meth_1k,
            format_default=".4f",
            description="Filtered median cross-variant kappa on methodology_1k",
            sample="methodology_1000",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "methodology_5k_audit_kappa": claim_entry(
            value=meth_5k,
            format_default=".4f",
            description="Filtered median cross-variant kappa on methodology_5k_audit_500",
            sample="methodology_5k_audit_500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "extended_5k_kappa": claim_entry(
            value=ext_5k,
            format_default=".4f",
            description="Filtered median cross-variant kappa on extended_5000",
            sample="extended_5000",
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
