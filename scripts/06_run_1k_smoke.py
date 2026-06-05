"""
Runs methodology-1k extraction smoke pass.

Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
Writes: local reports/artifacts determined by CLI defaults.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/06_run_1k_smoke.py` unless the script's argparse help says otherwise.
"""

