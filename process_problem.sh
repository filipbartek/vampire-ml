#!/usr/bin/env bash

set -euo pipefail

source env.sh

: "${JOBS:=${SLURM_CPUS_PER_TASK:-1}}"

# TODO: Expose more Vampire options.
VAMPIRE_COMMAND=(\
./vampire-ml.py vampire\
 --output "$OUTPUT"\
 --problem_base_path "$TPTP_PROBLEMS"\
 --vampire "$VAMPIRE"\
 --probe\
 --solve_runs "${SOLVE_RUNS_PER_PROBLEM:-1}"\
 --jobs "$JOBS"\
 --vampire_options "--include $TPTP"\
 --vampire_options_probe "--time_limit ${VAMPIRE_TIME_LIMIT_PROBE:-1}"\
 --vampire_options_solve "--time_limit ${VAMPIRE_TIME_LIMIT_SOLVE:-10}"\
 "$@"\
)

if [ -n "${BATCH_ID:-}" ]; then VAMPIRE_COMMAND+=(--output_batch "batch/$BATCH_ID"); fi

echo "${VAMPIRE_COMMAND[@]}"
time "${VAMPIRE_COMMAND[@]}"
echo $?
