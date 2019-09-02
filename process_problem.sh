#!/usr/bin/env bash

set -euo pipefail

PROBLEM=$1

if [ -n "${SLURM_ARRAY_JOB_ID-}" ]; then : "${OUTPUT:=slurm-$SLURM_ARRAY_JOB_ID}"; fi
if [ -n "${SLURM_JOB_ID-}" ]; then : "${OUTPUT:=slurm-$SLURM_JOB_ID}"; fi
: "${OUTPUT:=out/default}"
OUTPUT_DIR=$OUTPUT/$PROBLEM

mkdir -p "$OUTPUT_DIR"

env | sort > "$OUTPUT_DIR/env.txt"

source env.sh

# TODO: Expose more Vampire options.
"$VAMPIRE" --include "$TPTP" --mode clausify --time_limit "${VAMPIRE_TIME_LIMIT:-10}" --json_output "$OUTPUT_DIR/vampire.json" "$TPTP_PROBLEMS/$PROBLEM" > "$OUTPUT_DIR/stdout.txt" 2> "$OUTPUT_DIR/stderr.txt"
