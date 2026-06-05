from __future__ import annotations

# Release documentation:
# Computes claim-registry values for reasoning instructions.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs paper claim registry entries for reasoning instructions.

from pathlib import Path

from paper.claims.scripts._receipt import merge_into_claims_json

CLAIMS_PATH = Path(__file__).resolve().parent.parent / "claims.json"


def compute_all() -> dict:
    return {}


def main() -> int:
    new_claims = compute_all()
    n = merge_into_claims_json(CLAIMS_PATH, new_claims)
    print(f"Updated {n} claims in {CLAIMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
