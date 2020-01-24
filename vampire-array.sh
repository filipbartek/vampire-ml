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
PROBLEMS_COUNT=$(wc -l < "$PROBLEMS")
TIME_PER_TASK=${TIME_PER_TASK:-$((PROBLEMS_COUNT * (SOLVE_RUNS_PER_PROBLEM + 1) * (VAMPIRE_TIME_LIMIT + 10) / (ARRAY_TASK_COUNT * 60)))}

COMMON_SBATCH_OPTIONS=(
  "--partition=$PARTITION"
  "--mail-type=FAIL,REQUEUE,STAGE_OUT"
  "--comment=$(git rev-parse --verify HEAD)"
)

echo "PROBLEMS_COUNT=$PROBLEMS_COUNT"
echo "TIME_PER_TASK=$TIME_PER_TASK"

mkdir -p "$OUTPUT_SLURM"
ARRAY_JOB_ID=$(sbatch "${COMMON_SBATCH_OPTIONS[@]}" --cpus-per-task="$CPUS_PER_TASK" --job-name="$OUTPUT:vampire" --output="$OUTPUT_SLURM/%A_%a.out" --parsable --input="$PROBLEMS" --time="$TIME_PER_TASK" --mem-per-cpu=$((VAMPIRE_MEMORY_LIMIT + 128)) --array="$ARRAY" vampire.sh "$@")

BATCHES_DIR="$OUTPUT/batches/$ARRAY_JOB_ID"
mkdir -p "$BATCHES_DIR"
git rev-parse --verify HEAD > "$BATCHES_DIR/git-commit-sha.txt"
env | sort > "$BATCHES_DIR/env.txt"
echo "$@" > "$BATCHES_DIR/parameters.txt"

export ARRAY_JOB_ID
sbatch "${COMMON_SBATCH_OPTIONS[@]}" --job-name="$OUTPUT:aggregate" --output="$OUTPUT_SLURM/%j.out" --dependency=afterok:"$ARRAY_JOB_ID" aggregate.sh
