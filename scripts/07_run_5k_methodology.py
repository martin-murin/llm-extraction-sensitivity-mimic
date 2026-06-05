"""
Runs methodology-5k/audit extraction variants.

Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
Writes: local reports/artifacts determined by CLI defaults.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/07_run_5k_methodology.py` unless the script's argparse help says otherwise.
"""

