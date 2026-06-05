"""Verify claims by comparing stored receipts to current filesystem hashes.

This is the default verification: receipt-match only. If the script file or
any input file has changed since the claim was computed, the receipt no longer
matches. This catches data drift and code drift without re-running the
computations.

To actually recompute all claims and check for value drift, use:
  python paper/claims/recompute_all.py

Exit codes:
  0 — all receipts match
  1 — one or more receipts do not match
  2 — script or data files missing (cannot verify)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from paper.claims.scripts._receipt import build_receipt
except ModuleNotFoundError:
    from scripts._receipt import build_receipt  # type: ignore[import-not-found,no-redef]

CLAIMS_PATH = Path(__file__).resolve().parent / "claims.json"


def verify_all() -> int:
    """Compare every claim's stored receipt to a freshly-computed one."""
    if not CLAIMS_PATH.exists():
        print(f"No claims.json found at {CLAIMS_PATH}", file=sys.stderr)
        print("Run paper/claims/recompute_all.py first.", file=sys.stderr)
        return 2

    claims = json.loads(CLAIMS_PATH.read_text())
    if not claims:
        print("claims.json is empty.", file=sys.stderr)
        return 2

    matches = []
    mismatches = []
    errors = []

    for key, claim in sorted(claims.items()):
        receipt = claim.get("receipt")
        if not receipt:
            errors.append((key, "no receipt stored"))
            continue

        try:
            current = build_receipt(
                script_path=receipt["script_path"],
                function_name=receipt.get("function_name", "unknown"),
                input_files=receipt["input_files"],
            )
        except FileNotFoundError as exc:
            errors.append((key, f"missing file: {exc}"))
            continue
        except Exception as exc:
            errors.append((key, f"receipt build failed: {exc}"))
            continue

        if current["combined_hash"] == receipt["combined_hash"]:
            matches.append(key)
        else:
            differences = []
            if current["script_sha256"] != receipt["script_sha256"]:
                differences.append(f"script changed: {receipt['script_path']}")
            for f in receipt["input_files"]:
                old = receipt["input_files_sha256"].get(f)
                new = current["input_files_sha256"].get(f)
                if old != new:
                    differences.append(f"input changed: {f}")
            if not differences:
                differences.append("hash differs (cause unclear)")
            mismatches.append((key, differences))

    # Report
    total = len(claims)
    print(f"Verified {total} claims:")
    print(f"  {len(matches)} match")
    print(f"  {len(mismatches)} mismatch")
    print(f"  {len(errors)} error")
    print()

    if mismatches:
        print("MISMATCHES:")
        for key, diffs in mismatches:
            print(f"  {key}")
            for d in diffs:
                print(f"    - {d}")
        print()
        print("Re-run the corresponding compute_*.py scripts to refresh.")

    if errors:
        print("ERRORS:")
        for key, reason in errors:
            print(f"  {key}: {reason}")

    if mismatches or errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(verify_all())
