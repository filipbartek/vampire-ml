#!/usr/bin/env bash

# This script is intended to be run in cluster.ciirc.cvut.cz.

# Usage examples:
# PROBLEMS=problems/problems_selected_aggregated.txt OUTPUT=out/sp-random-predicate fit-array.sh --random-predicate-precedence
# PROBLEMS=problems/problems_selected_aggregated.txt OUTPUT=out/sp-random-predicate ARRAY=493,3406 fit-array.sh --random-predicate-precedence

set -euo pipefail

. env.sh

JOB_NAME=${JOB_NAME:-fit}

MEM_PER_CPU=${MEM_PER_CPU:-$((VAMPIRE_MEMORY_LIMIT + 1024))}

COMMON_SBATCH_OPTIONS=(
  --parsable
  "--comment=$(git rev-parse --verify HEAD)"
  "--mem-per-cpu=$MEM_PER_CPU"
)

OUTPUT_SLURM=${OUTPUT_SLURM:-slurm}
mkdir -p "$OUTPUT_SLURM"

PROBLEMS=${PROBLEMS:-problems/cnf_fof.txt}

if [ -n "${SKIP_MAP-}" ]; then
  echo "Skipping map step (array job)."
else
  PROBLEMS_ARRAY=${PROBLEMS_ARRAY:-${PROBLEMS_TRAIN:-$PROBLEMS}}
  PROBLEMS_COUNT=$(wc -l <"$PROBLEMS_ARRAY")
  echo "PROBLEMS_COUNT=$PROBLEMS_COUNT"

  MAX_ARRAY_SIZE=$(scontrol show config | grep MaxArraySize | awk '{split($0, a, "="); print a[2]}' | sed 's/^ *//g')
  # https://stackoverflow.com/a/10415158/4054250
  ARRAY_TASK_COUNT=${ARRAY_TASK_COUNT:-$((PROBLEMS_COUNT > MAX_ARRAY_SIZE ? MAX_ARRAY_SIZE : PROBLEMS_COUNT))}
  export PROBLEM_MODULUS=${PROBLEM_MODULUS:-$ARRAY_TASK_COUNT}
  ARRAY=${ARRAY:-0-$((ARRAY_TASK_COUNT - 1))}

  MAP_CPUS_PER_TASK=${MAP_CPUS_PER_TASK:-1}
  echo "MAP_CPUS_PER_TASK=$MAP_CPUS_PER_TASK"

  TIME_PER_TASK=${TIME_PER_TASK:-$((PROBLEMS_COUNT * (SOLVE_RUNS_PER_PROBLEM + 1) * (VAMPIRE_TIME_LIMIT + 10) / (ARRAY_TASK_COUNT * 60)))}
  echo "TIME_PER_TASK=$TIME_PER_TASK"

  MAP_JOB_ID=$(sbatch "${COMMON_SBATCH_OPTIONS[@]}" --job-name="$JOB_NAME:map" --output="$OUTPUT_SLURM/%A_%a.out" --cpus-per-task="$MAP_CPUS_PER_TASK" --input="$PROBLEMS_ARRAY" --time="$TIME_PER_TASK" --array="$ARRAY" fit.sh --precompute-only "$@")
  echo "MAP_JOB_ID=$MAP_JOB_ID"

  MAP_BATCHES_DIR="$OUTPUT/$MAP_JOB_ID"
  mkdir -p "$MAP_BATCHES_DIR"
  git rev-parse --verify HEAD >"$MAP_BATCHES_DIR/git-commit-sha.txt"
  env | sort >"$MAP_BATCHES_DIR/env.txt"
  echo "$@" >"$MAP_BATCHES_DIR/parameters.txt"
fi

if [ -n "${SKIP_REDUCE-}" ]; then
  echo "Skipping reduce step."
else
  REDUCE_CPUS_PER_TASK=${REDUCE_CPUS_PER_TASK:-1}
  echo "REDUCE_CPUS_PER_TASK=$REDUCE_CPUS_PER_TASK"

  if [ -n "${MAP_JOB_ID-}" ]; then
    DEPENDENCY_OPTION="--dependency=afterok:$MAP_JOB_ID"
  fi

  REDUCE_JOB_ID=$(sbatch "${COMMON_SBATCH_OPTIONS[@]}" --job-name="$JOB_NAME:reduce" --output="$OUTPUT_SLURM/%j.out" --cpus-per-task="$REDUCE_CPUS_PER_TASK" ${DEPENDENCY_OPTION-} fit.sh --problem-list "$PROBLEMS" "$@")
  echo "REDUCE_JOB_ID=$REDUCE_JOB_ID"

  REDUCE_BATCHES_DIR="$OUTPUT/$REDUCE_JOB_ID"
  mkdir -p "$REDUCE_BATCHES_DIR"
  git rev-parse --verify HEAD >"$REDUCE_BATCHES_DIR/git-commit-sha.txt"
  env | sort >"$REDUCE_BATCHES_DIR/env.txt"
  echo "$@" >"$REDUCE_BATCHES_DIR/parameters.txt"
fi
