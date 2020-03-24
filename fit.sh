#!/usr/bin/env bash

#SBATCH --mem-per-cpu=3128
#SBATCH --time=600
#SBATCH --requeue

set -euo pipefail

# https://stackoverflow.com/a/949391/4054250
echo Git commit: "$(git rev-parse --verify HEAD)"

env | sort

source env.sh

VAMPIRE_TIME_LIMIT=${VAMPIRE_TIME_LIMIT:-10}

python -O -m vampire_ml fit --vampire "$VAMPIRE" --vampire-options "{time_limit: $VAMPIRE_TIME_LIMIT, memory_limit: ${VAMPIRE_MEMORY_LIMIT:-3000}}" --timeout $((VAMPIRE_TIME_LIMIT + 10)) --include "$TPTP" --output "${OUTPUT:-out/default}" --problem-base-path "$TPTP_PROBLEMS" --solve-runs "${SOLVE_RUNS_PER_PROBLEM:-1000}" --problem-list "${PROBLEMS:-problems_selected_aggregated.txt}" --n-splits "${N_SPLITS:-5}" --train-size "${TRAIN_SIZE:-100}" --test-size "${TEST_SIZE:-100}" "$@"
