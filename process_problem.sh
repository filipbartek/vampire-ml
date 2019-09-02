#!/usr/bin/env bash

set -euo pipefail

PROBLEM=$1

source env.sh

OUTPUT_DIR=$OUTPUT/$PROBLEM

mkdir -p "$OUTPUT_DIR"

env | sort > "$OUTPUT_DIR/env.txt"

# TODO: Expose more Vampire options.
time "$VAMPIRE" --include "$TPTP" --mode clausify --time_limit "${VAMPIRE_TIME_LIMIT:-10}" --json_output "$OUTPUT_DIR/vampire.json" "$TPTP_PROBLEMS/$PROBLEM" > "$OUTPUT_DIR/stdout.txt" 2> "$OUTPUT_DIR/stderr.txt"
echo $? > "$OUTPUT_DIR/$EXITSTATUS_FILENAME"
