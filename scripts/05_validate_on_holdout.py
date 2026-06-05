"""
Runs holdout extraction/validation checks.

Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
Writes: local reports/artifacts determined by CLI defaults.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/05_validate_on_holdout.py` unless the script's argparse help says otherwise.
"""

