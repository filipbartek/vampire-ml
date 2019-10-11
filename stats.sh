#!/usr/bin/env bash

#SBATCH --mem=1G

set -euo pipefail

source env.sh

STRATEGY_ID=${STRATEGY_ID:-probe}
RESULTS_BASE=${RESULTS_BASE:-out/default/$STRATEGY_ID}
RESULTS=${RESULTS:-$RESULTS_BASE/problems/**/result.json}
OUTPUT_ID=${OUTPUT_ID:-default}
OUTPUT=${OUTPUT:-$RESULTS_BASE/stats/$OUTPUT_ID}
PROBLEMS=${PROBLEMS:-problems_cnf_fof.txt}

python -O vampire-ml.py stats "$RESULTS" --output "$OUTPUT" --problem-base-path "$TPTP_PROBLEMS" --problem-list "$PROBLEMS" "$@"
