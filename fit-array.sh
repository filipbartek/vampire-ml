#!/usr/bin/env bash

# This script is intended to be run in cluster.ciirc.cvut.cz.

# Usage examples:
# PROBLEMS=problems_selected_aggregated.txt OUTPUT=out/sp-random-predicate fit-array.sh --random-predicate-precedence
# PROBLEMS=problems_selected_aggregated.txt OUTPUT=out/sp-random-predicate ARRAY=493,3406 fit-array.sh --random-predicate-precedence

set -euo pipefail

export OUTPUT=${OUTPUT:-out}
export VAMPIRE_MEMORY_LIMIT=${VAMPIRE_MEMORY_LIMIT:-8192}
export VAMPIRE_TIME_LIMIT=${VAMPIRE_TIME_LIMIT:-10}
export SOLVE_RUNS_PER_PROBLEM=${SOLVE_RUNS_PER_PROBLEM:-1000}

MEM_PER_CPU=${MEM_PER_CPU:-$((VAMPIRE_MEMORY_LIMIT + 192))}

COMMON_SBATCH_OPTIONS=(
  --parsable
  "--comment=$(git rev-parse --verify HEAD)"
  "--mem-per-cpu=$MEM_PER_CPU"
)

OUTPUT_SLURM=${OUTPUT_SLURM:-$OUTPUT/slurm}
mkdir -p "$OUTPUT_SLURM"

PROBLEMS=${PROBLEMS:-problems_selected_aggregated.txt}
PROBLEMS_COUNT=$(wc -l <"$PROBLEMS")
echo "PROBLEMS_COUNT=$PROBLEMS_COUNT"

if [ -n "${SKIP_MAP-}" ]; then
  echo "Skipping map step (array job)."
else
  MAX_ARRAY_SIZE=$(scontrol show config | grep MaxArraySize | awk '{split($0, a, "="); print a[2]}' | sed 's/^ *//g')
  # https://stackoverflow.com/a/10415158/4054250
  ARRAY_TASK_COUNT=${ARRAY_TASK_COUNT:-$((PROBLEMS_COUNT > MAX_ARRAY_SIZE ? MAX_ARRAY_SIZE : PROBLEMS_COUNT))}
  export PROBLEM_MODULUS=${PROBLEM_MODULUS:-$ARRAY_TASK_COUNT}
  ARRAY=${ARRAY:-0-$((ARRAY_TASK_COUNT - 1))}

  MAP_CPUS_PER_TASK=${MAP_CPUS_PER_TASK:-1}
  echo "MAP_CPUS_PER_TASK=$MAP_CPUS_PER_TASK"

  TIME_PER_TASK=${TIME_PER_TASK:-$((PROBLEMS_COUNT * (SOLVE_RUNS_PER_PROBLEM + 1) * (VAMPIRE_TIME_LIMIT + 10) / (ARRAY_TASK_COUNT * 60)))}
  echo "TIME_PER_TASK=$TIME_PER_TASK"

  MAP_JOB_ID=$(sbatch "${COMMON_SBATCH_OPTIONS[@]}" --job-name="fit:map" --output="$OUTPUT_SLURM/%A_%a.out" --error="$OUTPUT_SLURM/%A_%a.err" --cpus-per-task="$MAP_CPUS_PER_TASK" --input="$PROBLEMS" --time="$TIME_PER_TASK" --array="$ARRAY" fit.sh "$@" --precompute)
  echo "MAP_JOB_ID=$MAP_JOB_ID"

  MAP_BATCHES_DIR="$OUTPUT/batches/$MAP_JOB_ID"
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

  REDUCE_JOB_ID=$(sbatch "${COMMON_SBATCH_OPTIONS[@]}" --job-name="fit:reduce" --output="$OUTPUT_SLURM/%j.out" --error="$OUTPUT_SLURM/%j.err" --cpus-per-task="$REDUCE_CPUS_PER_TASK" ${DEPENDENCY_OPTION-} fit.sh --problem-list "$PROBLEMS" "$@")
  echo "REDUCE_JOB_ID=$REDUCE_JOB_ID"

  REDUCE_BATCHES_DIR="$OUTPUT/batches/$REDUCE_JOB_ID"
  mkdir -p "$REDUCE_BATCHES_DIR"
  git rev-parse --verify HEAD >"$REDUCE_BATCHES_DIR/git-commit-sha.txt"
  env | sort >"$REDUCE_BATCHES_DIR/env.txt"
  echo "$@" >"$REDUCE_BATCHES_DIR/parameters.txt"
fi
