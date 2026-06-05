#!/usr/bin/env bash
# Release documentation: launches the production extraction workflow.
# Reads local MIMIC-derived split/note artifacts and OpenAI configuration.
# Writes local raw responses, logs, and monitoring artifacts; backs the production-scale extraction claims.

set -euo pipefail

# Source env
set -a && source ~/.env && set +a

# Configuration
RUN_ID="production_v1"
SPLIT="full"
VARIANT="a"
LOG_DIR="logs/production"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date -u +%Y%m%dT%H%M%S)_${RUN_ID}.log"

echo "============================================================"
echo "DryLabz MIMIC-IV LLM Extraction — Production Run"
echo "Run ID: $RUN_ID"
echo "Started: $(date -u)"
echo "Log file: $LOG_FILE"
echo "Expected duration: ~35 hours"
echo "Expected cost: ~\$652"
echo "============================================================"
echo ""
echo "Pre-flight checklist..."
python scripts/30_preflight_production.py || { echo "Pre-flight FAILED. Aborting."; exit 1; }

echo ""
echo "Launching extraction. Detach with Ctrl+B then D in tmux."
echo "Monitor progress in another pane with:"
echo "  python scripts/32_monitor_production.py --run-id $RUN_ID"
echo ""

# Launch with output to both stdout (visible if attached) and logfile
python scripts/05_run_smoke_coverage.py \
  --run-id "$RUN_ID" \
  --split "$SPLIT" \
  --variant "$VARIANT" \
  --config "configs/production.yaml" \
  --no-include-reasoning \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Run complete. Final log: $LOG_FILE"
echo "Run post-run reports with:"
echo "  python scripts/33_postrun_qa.py --run-id $RUN_ID"
echo "============================================================"
