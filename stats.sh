#!/usr/bin/env bash

set -euo pipefail

source env.sh

OUTPUT=${OUTPUT:-out/default}
STRATEGY_ID=${STRATEGY_ID:-probe}
PROBLEMS=${PROBLEMS:-problems_cnf_fof.txt}

python -O vampire-ml.py stats "$OUTPUT/jobs/$STRATEGY_ID/*/job.json" --output "$OUTPUT/stats/$STRATEGY_ID" --problem-list "$PROBLEMS" "$@"
