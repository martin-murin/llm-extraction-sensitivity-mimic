from __future__ import annotations

# Release documentation:
# Computes claim-registry values for cross prompt kappa.
#
# Reads: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Writes: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Backs paper claim registry entries for cross prompt kappa.

from pathlib import Path
import numpy as np

from paper.claims.scripts._common import claim_entry, require_input_files
from paper.claims.scripts._model_size_primary import (
    load_paired_model_size_data,
    pooled_kappa_levels,
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
    "data/raw_responses/paired_gold_methodology_1k_a/results.jsonl",
    "data/raw_responses/paired_gold_methodology_1k_b/results.jsonl",
    "data/raw_responses/paired_gold_methodology_1k_c/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_a/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_b/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_c/results.jsonl",
]


def _compute_pooled_levels_from_primary() -> tuple[float, float, float, float, int, int]:
    """Return tri_small, tri_full, col_small, col_full, n_fields, n_notes."""
    data = load_paired_model_size_data(base_rate_threshold=10)
    tri_small, tri_full, col_small, col_full = pooled_kappa_levels(data)
    return (
        tri_small,
        tri_full,
        col_small,
        col_full,
        len(data.included_fields),
        len(data.common_hadm_ids),
    )


def compute_all() -> dict:
    require_input_files(INPUT_FILES)
    tri_small, tri_full, col_small, col_full, n_fields, n_notes = _compute_pooled_levels_from_primary()
    tri_delta = (tri_full - tri_small) * 100.0

    timestamp = now_utc_iso()
    receipt = build_receipt(
        script_path=SCRIPT_PATH,
        function_name="compute_all",
        input_files=INPUT_FILES,
    )

    return {
        "paired_kappa_small_tristate": claim_entry(
            value=tri_small,
            format_default=".4f",
            description=(
                "Pooled paired cross-prompt kappa (small model, TriState): median over "
                "all included (variant-pair, field) cells on the shared methodology_1500 notes"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "paired_kappa_full_tristate": claim_entry(
            value=tri_full,
            format_default=".4f",
            description=(
                "Pooled paired cross-prompt kappa (full model, TriState): median over "
                "all included (variant-pair, field) cells on the shared methodology_1500 notes"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "paired_kappa_small_collapsed": claim_entry(
            value=col_small,
            format_default=".4f",
            description=(
                "Pooled paired cross-prompt kappa (small model, collapsed): median over "
                "all included (variant-pair, field) cells on the shared methodology_1500 notes"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "paired_kappa_full_collapsed": claim_entry(
            value=col_full,
            format_default=".4f",
            description=(
                "Pooled paired cross-prompt kappa (full model, collapsed): median over "
                "all included (variant-pair, field) cells on the shared methodology_1500 notes"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "model_size_pooled_delta_tristate": claim_entry(
            value=tri_delta,
            format_default=".2f",
            unit="pp",
            description=(
                "Pooled paired TriState model-size delta (full minus small), computed on "
                f"{n_fields} included TriState fields across {n_notes} shared notes"
            ),
            sample="methodology_1500",
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
