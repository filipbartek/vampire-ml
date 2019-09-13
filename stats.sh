#!/usr/bin/env bash

set -euo pipefail

source env.sh

RESULTS_BASE=${RESULTS_BASE:-out/default}
STRATEGY_ID=${STRATEGY_ID:-probe}
# JOB_ID may be a glob pattern.
JOB_ID=${JOB_ID:-*/*}
RESULTS=${RESULTS:-$RESULTS_BASE/jobs/$STRATEGY_ID/$JOB_ID/job.json}
OUTPUT_ID=${OUTPUT_ID:-$STRATEGY_ID}
OUTPUT=${OUTPUT:-$RESULTS_BASE/stats/$OUTPUT_ID}
PROBLEMS=${PROBLEMS:-problems_cnf_fof.txt}

python -O vampire-ml.py stats "$RESULTS" --output "$OUTPUT" --problem-list "$PROBLEMS" "$@"
