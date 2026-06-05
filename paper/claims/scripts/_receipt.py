from __future__ import annotations

# Release documentation:
# Provides shared helpers for claim-registry recomputation.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Supports paper claim recomputation and receipt verification.

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path


def hash_file(path: str | Path) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash: file does not exist: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_receipt(
    script_path: str | Path,
    function_name: str,
    input_files: Sequence[str | Path],
) -> dict:
    """Construct a receipt documenting how a claim was computed.

    Args:
        script_path: path to the .py file that computed the claim
        function_name: name of the function within the script (for traceability)
        input_files: list of source data paths the script read

    Returns:
        Dict with script_sha256, input_files_sha256 (sorted), and combined_hash.
    """
    script_path_str = str(script_path)
    script_sha = hash_file(script_path)

    inputs_sha: dict[str, str] = {}
    for f in sorted(str(p) for p in input_files):
        inputs_sha[f] = hash_file(f)

    # Combined hash: deterministic from script SHA + sorted input SHAs.
    combined = hashlib.sha256()
    combined.update(script_sha.encode())
    for input_sha in sorted(inputs_sha.values()):
        combined.update(input_sha.encode())

    return {
        "script_path": script_path_str,
        "script_sha256": script_sha,
        "function_name": function_name,
        "input_files": list(inputs_sha.keys()),
        "input_files_sha256": inputs_sha,
        "combined_hash": combined.hexdigest(),
    }


def now_utc_iso() -> str:
    """Return current time as ISO 8601 UTC string."""
    return datetime.now(UTC).isoformat()


def merge_into_claims_json(claims_path: Path, new_claims: dict) -> int:
    """Merge new claim entries into claims.json, returning count merged.

    If claims.json does not exist, it is created. Existing claim entries are
    overwritten by the new ones (this is the expected behavior when recomputing).
    """
    import json

    existing = json.loads(claims_path.read_text()) if claims_path.exists() else {}

    existing.update(new_claims)
    claims_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    return len(new_claims)
