#!/usr/bin/env bash

# This script is intended to be run in cluster.ciirc.cvut.cz.

set -euo pipefail

OUTPUT_BASE=${OUTPUT_BASE:-out}
SOLVE_RUNS_PER_PROBLEM=${SOLVE_RUNS_PER_PROBLEM:-24}

OUTPUT=${OUTPUT_BASE}/sp-frequency ./vampire-array.sh
OUTPUT=${OUTPUT_BASE}/sp-random-predicate SOLVE_RUNS_PER_PROBLEM=$SOLVE_RUNS_PER_PROBLEM ./vampire-array.sh --random-predicate-precedence
OUTPUT=${OUTPUT_BASE}/sp-random-function SOLVE_RUNS_PER_PROBLEM=$SOLVE_RUNS_PER_PROBLEM ./vampire-array.sh --random-function-precedence
OUTPUT=${OUTPUT_BASE}/sp-random-both SOLVE_RUNS_PER_PROBLEM=$SOLVE_RUNS_PER_PROBLEM ./vampire-array.sh --random-predicate-precedence --random-function-precedence
