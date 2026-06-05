"""
Checks prompt/schema vocabulary coverage.

Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
Writes: local reports/artifacts determined by CLI defaults.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/03_vocabulary_coverage_test.py` unless the script's argparse help says otherwise.
"""

