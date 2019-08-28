#!/usr/bin/env bash

set -euo pipefail

PROBLEM=$1
OUTPUT_DIR=$2

mkdir -p "$OUTPUT_DIR"

env | sort > "$OUTPUT_DIR/env.txt"

source env.sh

VAMPIRE_COMMAND="$VAMPIRE --include $TPTP --mode clausify --time_limit 10 --json_output $OUTPUT_DIR/vampire.json $TPTP_PROBLEMS/$PROBLEM"

# TODO: Record exit code.
# TODO: Record execution time.

if command -v srun > /dev/null
then
  srun --job-name=vampire --ntasks=1 sh -c "$VAMPIRE_COMMAND" > "$OUTPUT_DIR/stdout.txt" 2> "$OUTPUT_DIR/stderr.txt"
else
  $VAMPIRE_COMMAND > "$OUTPUT_DIR/stdout.txt" 2> "$OUTPUT_DIR/stderr.txt"
fi
