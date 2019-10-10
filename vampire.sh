#!/usr/bin/env bash

# This script only uses one Slurm task.

# TODO: Calibrate the memory requirement.
#SBATCH --mem-per-cpu=1G
#SBATCH --requeue

set -euo pipefail

# https://stackoverflow.com/a/949391/4054250
echo Git commit: "$(git rev-parse --verify HEAD)"

env | sort

source env.sh

if [ -n "${SLURM_ARRAY_JOB_ID-}" ]; then
  OUTPUT=${OUTPUT:-out/slurm/$SLURM_ARRAY_JOB_ID}
  if [ -n "${SLURM_ARRAY_TASK_ID-}" ]; then JOB_ID=${JOB_ID:-$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID}; fi
fi
if [ -n "${SLURM_JOB_ID-}" ]; then
  OUTPUT=${OUTPUT:-out/slurm/$SLURM_JOB_ID}
  JOB_ID=${JOB_ID:-$SLURM_JOB_ID}
fi
OUTPUT=${OUTPUT:-out/default}

VAMPIRE_MODE=${VAMPIRE_MODE:-vampire}
VAMPIRE_SYMBOL_PRECEDENCE=${VAMPIRE_SYMBOL_PRECEDENCE:-scramble}
STRATEGY_ID=${STRATEGY_ID:-$VAMPIRE_MODE-$VAMPIRE_SYMBOL_PRECEDENCE}
SOLVE_RUNS_PER_PROBLEM=${SOLVE_RUNS_PER_PROBLEM:-1}
CPUS=${CPUS:-${SLURM_CPUS_PER_TASK:-1}}

# TODO: Expose more Vampire options.
XARGS_COMMAND=(
  xargs --verbose
  python -O
  vampire-ml.py vampire
  --output "$OUTPUT"
  --strategy-id "$STRATEGY_ID"
  --solve-runs "$SOLVE_RUNS_PER_PROBLEM"
  --vampire "$VAMPIRE"
  --vampire-options "--include $TPTP --mode $VAMPIRE_MODE --symbol_precedence $VAMPIRE_SYMBOL_PRECEDENCE --time_limit 10"
  --cpus "$CPUS"
  --problem-base-path "$TPTP_PROBLEMS"
  "$@"
)

if [ -n "${VAMPIRE_OPTIONS-}" ]; then XARGS_COMMAND+=(--vampire-options "$VAMPIRE_OPTIONS"); fi

if [ -n "${JOB_ID-}" ]; then XARGS_COMMAND+=(--job-id "$JOB_ID"); fi

if [ -n "${SLURM_JOB_ID-}" ]; then OUTPUT_SCRATCH=${OUTPUT_SCRATCH-/lscratch/$USER/slurm-$SLURM_JOB_ID}; fi

# See also https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html#how-to-recover-files-before-a-job-times-out
function finish {
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
