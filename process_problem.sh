#!/usr/bin/env bash

set -euo pipefail

source env.sh

: "${JOBS:=${SLURM_CPUS_PER_TASK:-1}}"

OUTPUT_TMP=$OUTPUT
# TODO: Make the output directory unique even in case this script is called multiple times in one Slurm job.
if [ -n "${SLURM_JOB_ID-}" ]; then OUTPUT_TMP=/lscratch/$USER/slurm-$SLURM_JOB_ID; fi

OUTPUT_BATCH=$OUTPUT
OUTPUT_RUNS=$OUTPUT_TMP/problems

if [ -n "${BATCH_ID:-}" ]; then OUTPUT_BATCH=$OUTPUT/batches/$BATCH_ID; fi

# See also https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html#how-to-recover-files-before-a-job-times-out
function finish {
  if [ -n "${SLURM_JOB_ID-}" ]; then
    mkdir -p "$OUTPUT"
    cp -rvt "$OUTPUT" "$OUTPUT_RUNS"
    rm -rf "$OUTPUT_TMP"
  fi
}
trap finish EXIT INT TERM

# TODO: Expose more Vampire options.
VAMPIRE_COMMAND=(
  python -O
  vampire-ml.py vampire
  --output_batch "$OUTPUT_BATCH"
  --output_runs "$OUTPUT_RUNS"
  --problem_base_path "$TPTP_PROBLEMS"
  --vampire "$VAMPIRE"
  --probe
  --solve_runs "${SOLVE_RUNS_PER_PROBLEM:-1}"
  --jobs "$JOBS"
  --vampire_options "--include $TPTP"
  --vampire_options_probe "--time_limit ${VAMPIRE_TIME_LIMIT_PROBE:-10}"
  --vampire_options_solve "--time_limit ${VAMPIRE_TIME_LIMIT_SOLVE:-10}"
  "$@"
)

echo "${VAMPIRE_COMMAND[@]}"
time "${VAMPIRE_COMMAND[@]}"
echo $?
