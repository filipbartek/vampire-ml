#!/usr/bin/env bash

# This script is intended to be run in cluster.ciirc.cvut.cz.

# Usage example: PROBLEMS=problems_selected_aggregated.txt OUTPUT=out/sp-random-predicate sbatch fit-array.sh --random-predicate-precedence

set -euo pipefail

export OUTPUT=${OUTPUT:-out}
export VAMPIRE_MEMORY_LIMIT=${VAMPIRE_MEMORY_LIMIT:-8192}
export VAMPIRE_TIME_LIMIT=${VAMPIRE_TIME_LIMIT:-10}
export SOLVE_RUNS_PER_PROBLEM=${SOLVE_RUNS_PER_PROBLEM:-1000}

COMMON_SBATCH_OPTIONS=(
  "--comment=$(git rev-parse --verify HEAD)"
)

OUTPUT_SLURM=${OUTPUT_SLURM:-$OUTPUT/slurm}
mkdir -p "$OUTPUT_SLURM"

PROBLEMS=${PROBLEMS:-problems_selected_aggregated.txt}

if [ -n "${SKIP_MAP-}" ]; then
  echo "Skipping map step (array job)."
  sbatch "${COMMON_SBATCH_OPTIONS[@]}" --job-name="fit:reduce" --output="$OUTPUT_SLURM/%j.out" fit.sh --problem-list "$PROBLEMS" "$@"
else
  PROBLEMS_COUNT=$(wc -l <"$PROBLEMS")
  CPUS_PER_TASK=${CPUS_PER_TASK:-1}
  MAX_ARRAY_SIZE=$(scontrol show config | grep MaxArraySize | awk '{split($0, a, "="); print a[2]}' | sed 's/^ *//g')
  # https://stackoverflow.com/a/10415158/4054250
  ARRAY_TASK_COUNT=${ARRAY_TASK_COUNT:-$((PROBLEMS_COUNT > MAX_ARRAY_SIZE ? MAX_ARRAY_SIZE : PROBLEMS_COUNT))}
  ARRAY=${ARRAY:-0-$((ARRAY_TASK_COUNT - 1))}
  TIME_PER_TASK=${TIME_PER_TASK:-$((PROBLEMS_COUNT * (SOLVE_RUNS_PER_PROBLEM + 1) * (VAMPIRE_TIME_LIMIT + 10) / (ARRAY_TASK_COUNT * 60)))}

  echo "PROBLEMS_COUNT=$PROBLEMS_COUNT"
  echo "TIME_PER_TASK=$TIME_PER_TASK"

  ARRAY_JOB_ID=$(sbatch "${COMMON_SBATCH_OPTIONS[@]}" --cpus-per-task="$CPUS_PER_TASK" --job-name="fit:map" --output="$OUTPUT_SLURM/%A_%a.out" --parsable --input="$PROBLEMS" --time="$TIME_PER_TASK" --mem-per-cpu=$((VAMPIRE_MEMORY_LIMIT + 128)) --array="$ARRAY" fit.sh "$@" --precompute)

  echo Array job ID: "$ARRAY_JOB_ID"

  BATCHES_DIR="$OUTPUT/batches/$ARRAY_JOB_ID"
  mkdir -p "$BATCHES_DIR"
  git rev-parse --verify HEAD >"$BATCHES_DIR/git-commit-sha.txt"
  env | sort >"$BATCHES_DIR/env.txt"
  echo "$@" >"$BATCHES_DIR/parameters.txt"

  export ARRAY_JOB_ID
  sbatch "${COMMON_SBATCH_OPTIONS[@]}" --job-name="fit:reduce" --output="$OUTPUT_SLURM/%j.out" --dependency=afterok:"$ARRAY_JOB_ID" fit.sh --problem-list "$PROBLEMS" "$@"
fi
