#!/usr/bin/env bash

set -euo pipefail

source env.sh

: "${JOBS:=${SLURM_CPUS_PER_TASK:-1}}"

# TODO: Make the output directory unique even in case this script is called multiple times in one Slurm job.
if [ -n "${SLURM_JOB_ID-}" ]; then OUTPUT_SCRATCH=/lscratch/$USER/slurm-$SLURM_JOB_ID; fi

# See also https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html#how-to-recover-files-before-a-job-times-out
function finish {
  if [ -n "${OUTPUT_SCRATCH-}" ]; then rm -rf "$OUTPUT_SCRATCH"; fi
}
trap finish EXIT INT TERM

# TODO: Expose more Vampire options.
VAMPIRE_COMMAND=(
  python -O
  vampire-ml.py vampire
  --output "$OUTPUT"
  --strategy-id "${VAMPIRE_MODE:-clausify}"
  --solve-runs "${SOLVE_RUNS_PER_PROBLEM:-1}"
  --vampire "$VAMPIRE"
  --vampire-options "--include $TPTP --mode ${VAMPIRE_MODE:-clausify} --time_limit ${VAMPIRE_TIME_LIMIT:-10} --symbol_precedence ${VAMPIRE_SYMBOL_PRECEDENCE:-scramble}"
  --cpus "$JOBS"
  --problem-base-path "$TPTP_PROBLEMS"
  --no-clobber
  "$@"
)

# TODO: Use array job composite id in case array job is running.
if [ -n "${SLURM_JOB_ID-}" ]; then VAMPIRE_COMMAND+=(--job-id "$SLURM_JOB_ID"); fi
if [ -n "${OUTPUT_SCRATCH-}" ]; then VAMPIRE_COMMAND+=(--scratch "$OUTPUT_SCRATCH"); fi

echo "${VAMPIRE_COMMAND[@]}"
time "${VAMPIRE_COMMAND[@]}"
echo $?
