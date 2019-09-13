#!/usr/bin/env bash

set -euo pipefail

OUTPUT=${OUTPUT:-out/default}
STRATEGY_ID=${STRATEGY_ID:-probe}
PROBLEMS=${PROBLEMS:-problems_cnf_fof.txt}

./vampire-ml.py stats "$OUTPUT/jobs/$STRATEGY_ID/*/job.json" --output "$OUTPUT/stats/$STRATEGY_ID" --problem_list "$PROBLEMS" "$@"
