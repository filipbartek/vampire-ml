#!/usr/bin/env bash

set -euo pipefail

source env.sh

: "${JOBS:=${SLURM_CPUS_PER_TASK:-1}}"

OUTPUT_TMP=$OUTPUT
# TODO: Make the output directory unique even in case this script is called multiple times in one Slurm job.
if [ -n "${SLURM_JOB_ID-}" ]; then OUTPUT_TMP=/lscratch/$USER/slurm-$SLURM_JOB_ID; fi

# TODO: Expose more Vampire options.
VAMPIRE_COMMAND=(
  python -O
  vampire-ml.py vampire
  --output "$OUTPUT_TMP"
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

if [ -n "${BATCH_ID:-}" ]; then VAMPIRE_COMMAND+=(--output_batch "batch/$BATCH_ID"); fi

echo "${VAMPIRE_COMMAND[@]}"
time "${VAMPIRE_COMMAND[@]}"
echo $?

if [ -n "${SLURM_JOB_ID-}" ]; then
  mkdir -p "$OUTPUT"
  cp -rvt "$OUTPUT" "$OUTPUT_TMP"/*
  rm -rf "$OUTPUT_TMP"
fi
