#!/usr/bin/env bash

# This script is intended to be run in cluster.ciirc.cvut.cz.

# Example calls:
# STRATEGY_ID=probe    VAMPIRE_MODE=clausify TIME_PER_TASK=8  CPUS_PER_TASK=4  vampire-array.sh
# STRATEGY_ID=scramble VAMPIRE_MODE=vampire  TIME_PER_TASK=68 CPUS_PER_TASK=16 PROBLEMS=problems_probed_0.txt VAMPIRE_SYMBOL_PRECEDENCE=scramble SOLVE_RUNS_PER_PROBLEM=16 vampire-array.sh

set -euo pipefail

PROBLEMS=${PROBLEMS:-problems_cnf_fof.txt}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
TIME_PER_TASK=${TIME_PER_TASK:-68}
PARTITION=${PARTITION:-compute,gpu}
ARRAY_TASK_COUNT=${ARRAY_TASK_COUNT:-1000}

mkdir -p "${OUTPUT:-out}/slurm"
sbatch --input="$PROBLEMS" --cpus-per-task="$CPUS_PER_TASK" --time="$TIME_PER_TASK" --partition="$PARTITION" --array=0-$((ARRAY_TASK_COUNT - 1)) --mail-type=FAIL,REQUEUE,STAGE_OUT --job-name="${STRATEGY_ID:-vampire}" --comment="$(git rev-parse --verify HEAD)" --output="${OUTPUT:-out}/slurm/%A_%a.out" vampire.sh
