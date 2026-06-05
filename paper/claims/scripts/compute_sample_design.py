from __future__ import annotations

# Release documentation:
# Computes claim-registry values for sample design.
#
# Reads: data/splits/SPLITS_MANIFEST.json, data/splits/refinement_150.csv, data/splits/holdout_150.csv, data/splits/methodology_1k.csv, data/splits/methodology_5k_audit_500.csv, data/splits/extended_5k.csv.
# Writes: data/splits/SPLITS_MANIFEST.json, data/splits/refinement_150.csv, data/splits/holdout_150.csv, data/splits/methodology_1k.csv, data/splits/methodology_5k_audit_500.csv, data/splits/extended_5k.csv.
# Backs paper claim registry entries for sample design.

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from paper.claims.scripts._common import claim_entry, require_input_files
from paper.claims.scripts._receipt import build_receipt, merge_into_claims_json, now_utc_iso

CLAIMS_PATH = Path(__file__).resolve().parent.parent / "claims.json"
SCRIPT_PATH = Path(__file__).resolve()
REPO = Path(__file__).resolve().parents[3]

INPUT_FILES = [
    "data/splits/SPLITS_MANIFEST.json",
    "data/splits/refinement_150.csv",
    "data/splits/holdout_150.csv",
    "data/splits/methodology_1k.csv",
    "data/splits/methodology_5k_audit_500.csv",
    "data/splits/extended_5k.csv",
]


def compute_all() -> dict:
    require_input_files(INPUT_FILES)
    n_ref = len(pd.read_csv(REPO / "data/splits/refinement_150.csv"))
    n_hold = len(pd.read_csv(REPO / "data/splits/holdout_150.csv"))
    n_1k = len(pd.read_csv(REPO / "data/splits/methodology_1k.csv"))
    n_500 = len(pd.read_csv(REPO / "data/splits/methodology_5k_audit_500.csv"))
    n_ext = len(pd.read_csv(REPO / "data/splits/extended_5k.csv"))
    n_1500 = int(n_1k + n_500)

    timestamp = now_utc_iso()
    receipt = build_receipt(SCRIPT_PATH, "compute_all", INPUT_FILES)
    return {
        "refinement_150_n": claim_entry(
            value=n_ref,
            format_default=",d",
            description="Refinement split size",
            sample="refinement_150",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "holdout_150_n": claim_entry(
            value=n_hold,
            format_default=",d",
            description="Holdout split size",
            sample="holdout_150",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "methodology_1000_n": claim_entry(
            value=n_1k,
            format_default=",d",
            description="Methodology_1k split size",
            sample="methodology_1000",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "methodology_5k_audit_500_n": claim_entry(
            value=n_500,
            format_default=",d",
            description="Methodology_5k_audit_500 split size",
            sample="methodology_5k_audit_500",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "extended_5000_n": claim_entry(
            value=n_ext,
            format_default=",d",
            description="Extended_5k split size",
            sample="extended_5000",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "extended_5k_n": claim_entry(
            value=n_ext,
            format_default=",d",
            description="Extended validation sample size",
            sample="extended_5000",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "methodology_1500_n": claim_entry(
            value=n_1500,
            format_default=",d",
            description="Combined methodology paired sample size",
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
