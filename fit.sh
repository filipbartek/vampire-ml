#!/usr/bin/env bash

#SBATCH --mem-per-cpu=3128
#SBATCH --time=600
#SBATCH --requeue

# Usage example: OUTPUT=out/sp-random-predicate sbatch --input=problems_selected_aggregated.txt --job-name=fit-predicate fit.sh --random-predicate-precedence

set -euo pipefail

# https://stackoverflow.com/a/949391/4054250
echo Git commit: "$(git rev-parse --verify HEAD)"

env | sort

source env.sh

OUTPUT=${OUTPUT:-out}
if [ -n "${SLURM_ARRAY_JOB_ID-}" ] && [ -n "${SLURM_ARRAY_TASK_ID-}" ]; then
  JOB_ID=${JOB_ID-$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID}
fi
if [ -n "${SLURM_JOB_ID-}" ]; then
  JOB_ID=${JOB_ID-$SLURM_JOB_ID}
fi
if [ -n "${JOB_ID-}" ]; then
  OUTPUT_LOCAL=$OUTPUT/slurm/$JOB_ID
fi
OUTPUT_LOCAL=${OUTPUT_LOCAL:-$OUTPUT/default}

VAMPIRE_TIME_LIMIT=${VAMPIRE_TIME_LIMIT:-10}
VAMPIRE_MEMORY_LIMIT=${VAMPIRE_MEMORY_LIMIT:-8192}
SOLVE_RUNS_PER_PROBLEM=${SOLVE_RUNS_PER_PROBLEM:-1000}
CPUS=${CPUS:-${SLURM_CPUS_PER_TASK:-1}}

XARGS_COMMAND=(
  xargs --verbose
  python -O
  -m vampire_ml fit
  --output "$OUTPUT_LOCAL"
  --solve-runs "$SOLVE_RUNS_PER_PROBLEM"
  --vampire "$VAMPIRE"
  --vampire-options "{time_limit: $VAMPIRE_TIME_LIMIT, memory_limit: $VAMPIRE_MEMORY_LIMIT}"
  --include "$TPTP"
  --problem-base-path "$TPTP_PROBLEMS"
  --timeout $((VAMPIRE_TIME_LIMIT + 10))
  "$@"
)

if [ -n "${SLURM_JOB_ID-}" ]; then OUTPUT_SCRATCH=${OUTPUT_SCRATCH-/lscratch/$USER/slurm-$SLURM_JOB_ID}; fi

# See also https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html#how-to-recover-files-before-a-job-times-out
function finish {
  echo Finish: Removing directory "$OUTPUT_SCRATCH"
  if [ -n "${OUTPUT_SCRATCH-}" ]; then rm -rf "$OUTPUT_SCRATCH"; fi
}
trap finish EXIT INT TERM

if [ -n "${OUTPUT_SCRATCH-}" ]; then XARGS_COMMAND+=(--scratch "$OUTPUT_SCRATCH"); fi

PROBLEM_ID=${PROBLEM_ID:-${SLURM_ARRAY_TASK_ID:-}}
if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ]; then PROBLEM_MODULUS=${PROBLEM_MODULUS:-$((SLURM_ARRAY_TASK_MAX + 1))}; fi
# -1 stands for infinite modulus.
PROBLEM_MODULUS=${PROBLEM_MODULUS:--1}

if [ -n "${PROBLEM_ID:-}" ]; then
  if [ -n "${PROBLEM_MODULUS:-}" ] && [ -1 -ne "${PROBLEM_MODULUS:-}" ]; then
    echo "Processing problems with id $PROBLEM_ID modulo $PROBLEM_MODULUS."
    sed -n "$((PROBLEM_ID + 1))~${PROBLEM_MODULUS}p" | "${XARGS_COMMAND[@]}"
  else
    echo "Processing problem with id $PROBLEM_ID."
    sed -n "$((PROBLEM_ID + 1))p" | "${XARGS_COMMAND[@]}"
  fi
else
  echo "Processing all problems."
  "${XARGS_COMMAND[@]}"
fi
