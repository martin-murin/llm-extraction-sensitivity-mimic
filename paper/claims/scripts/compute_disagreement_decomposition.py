from __future__ import annotations

# Release documentation:
# Computes claim-registry values for disagreement decomposition.
#
# Reads: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Writes: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Backs paper claim registry entries for disagreement decomposition.

from pathlib import Path

from paper.claims.scripts._common import claim_entry
from paper.claims.scripts._disagreement_metrics_primary import (
    compute_cross_variant_pooled_disagreement_metrics,
)
from paper.claims.scripts._receipt import build_receipt, merge_into_claims_json, now_utc_iso

CLAIMS_PATH = Path(__file__).resolve().parent.parent / "claims.json"
SCRIPT_PATH = Path(__file__).resolve()
INPUT_FILES = [
    "data/raw_responses/methodology_1k_a/results.jsonl",
    "data/raw_responses/methodology_1k_b/results.jsonl",
    "data/raw_responses/methodology_1k_c/results.jsonl",
    "data/raw_responses/methodology_5k_a_subset500/results.jsonl",
    "data/raw_responses/methodology_5k_audit_b/results.jsonl",
    "data/raw_responses/methodology_5k_audit_c/results.jsonl",
    "data/raw_responses/production_v1/results.jsonl",
    "data/raw_responses/extended_5k_b/results.jsonl",
    "data/raw_responses/extended_5k_c/results.jsonl",
    "src/schema/fields.py",
]


def compute_all() -> dict:
    metrics = compute_cross_variant_pooled_disagreement_metrics()

    timestamp = now_utc_iso()
    receipt = build_receipt(SCRIPT_PATH, "compute_all", INPUT_FILES)
    return {
        "disagreement_soft_pct": claim_entry(
            value=float(metrics["disagreement_soft_pct"]),
            format_default=".2f",
            unit="%",
            description=(
                "Percent of TriState disagreements on soft axes "
                "(no/not_documented + yes/not_documented)"
            ),
            sample="cross_variant_pooled_6500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "disagreement_hard_pct": claim_entry(
            value=float(metrics["disagreement_hard_pct"]),
            format_default=".2f",
            unit="%",
            description="Percent of TriState disagreements that are hard yes/no flips",
            sample="cross_variant_pooled_6500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "disagreement_soft_yes_vs_not_documented_pct": claim_entry(
            value=float(metrics["disagreement_soft_yes_vs_not_documented_pct"]),
            format_default=".2f",
            unit="%",
            description="Percent of TriState disagreements on yes vs not_documented axis",
            sample="cross_variant_pooled_6500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "disagreement_soft_no_vs_not_documented_pct": claim_entry(
            value=float(metrics["disagreement_soft_no_vs_not_documented_pct"]),
            format_default=".2f",
            unit="%",
            description="Percent of TriState disagreements on no vs not_documented axis",
            sample="cross_variant_pooled_6500",
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
