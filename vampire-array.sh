#!/usr/bin/env bash

# This script is intended to be run in cluster.ciirc.cvut.cz.

set -euo pipefail

export OUTPUT=${OUTPUT:-out/default}
export VAMPIRE_MEMORY_LIMIT=${VAMPIRE_MEMORY_LIMIT:-8192}
export VAMPIRE_TIME_LIMIT=${VAMPIRE_TIME_LIMIT:-10}
export SOLVE_RUNS_PER_PROBLEM=${SOLVE_RUNS_PER_PROBLEM:-1}

# sbatch parameters
OUTPUT_SLURM=${OUTPUT_SLURM:-$OUTPUT/slurm}
PROBLEMS=${PROBLEMS:-problems_cnf_fof.txt}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
PARTITION=${PARTITION:-compute}
ARRAY_TASK_COUNT=${ARRAY_TASK_COUNT:-1000}
ARRAY=${ARRAY:-0-$((ARRAY_TASK_COUNT - 1))}
TIME_PER_TASK=${TIME_PER_TASK:-$((16748 * (SOLVE_RUNS_PER_PROBLEM + 1) * (VAMPIRE_TIME_LIMIT + 10) / (ARRAY_TASK_COUNT * 60)))}

COMMON_SBATCH_OPTIONS=(
  "--partition=$PARTITION"
  "--cpus-per-task=$CPUS_PER_TASK"
  "--mail-type=FAIL,REQUEUE,STAGE_OUT"
  "--comment=$(git rev-parse --verify HEAD)"
)

echo "TIME_PER_TASK=$TIME_PER_TASK"

mkdir -p "$OUTPUT_SLURM"
ARRAY_JOB_ID=$(sbatch "${COMMON_SBATCH_OPTIONS[@]}" --job-name="$OUTPUT:vampire" --output="$OUTPUT_SLURM/%A_%a.out" --parsable --input="$PROBLEMS" --time="$TIME_PER_TASK" --mem-per-cpu=$((VAMPIRE_MEMORY_LIMIT + 128)) --array="$ARRAY" vampire.sh "$@")
export ARRAY_JOB_ID
sbatch "${COMMON_SBATCH_OPTIONS[@]}" --job-name="$OUTPUT:aggregate" --output="$OUTPUT_SLURM/%j.out" --dependency=afterok:"$ARRAY_JOB_ID" aggregate.sh
