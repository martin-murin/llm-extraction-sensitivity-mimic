"""
Summarizes post-run QA for extraction outputs.

Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
Writes: local reports/artifacts determined by CLI defaults.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/09_post_run_qa.py` unless the script's argparse help says otherwise.
"""

