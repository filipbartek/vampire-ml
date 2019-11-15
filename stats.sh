#!/usr/bin/env bash

#SBATCH --mem=4G
#SBATCH --time=240

set -euo pipefail

source env.sh

RESULTS_BASE=${RESULTS_BASE:-out/default}
RESULTS=${RESULTS:-$RESULTS_BASE/problems/**/result.json}
OUTPUT_ID=${OUTPUT_ID:-default}
OUTPUT=${OUTPUT:-$RESULTS_BASE/stats/$OUTPUT_ID}
PROBLEMS=${PROBLEMS:-problems_cnf_fof.txt}

python -O vampire-ml.py stats "$RESULTS" --output "$OUTPUT" --problem-base-path "$TPTP_PROBLEMS" --problem-list "$PROBLEMS" "$@"
