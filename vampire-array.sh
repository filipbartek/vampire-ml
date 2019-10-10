#!/usr/bin/env bash

# This script is intended to be run in cluster.ciirc.cvut.cz.

set -euo pipefail

PROBLEMS=${PROBLEMS:-problems_cnf_fof.txt}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
TIME_PER_TASK=${TIME_PER_TASK:-10}
PARTITION=${PARTITION:-compute}
ARRAY_TASK_COUNT=${ARRAY_TASK_COUNT:-1000}
ARRAY=${ARRAY:-0-$((ARRAY_TASK_COUNT - 1))}

mkdir -p "${OUTPUT:-out}/slurm"
sbatch --input="$PROBLEMS" --cpus-per-task="$CPUS_PER_TASK" --time="$TIME_PER_TASK" --partition="$PARTITION" --array="$ARRAY" --mail-type=FAIL,REQUEUE,STAGE_OUT --job-name="${STRATEGY_ID:-vampire}" --comment="$(git rev-parse --verify HEAD)" --output="${OUTPUT:-out}/slurm/%A_%a.out" vampire.sh "$@"
