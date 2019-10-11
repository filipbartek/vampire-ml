#!/usr/bin/env bash

set -euo pipefail

export STRATEGY_ID=${STRATEGY_ID-scramble}
export OUTPUT=${OUTPUT-out/default/$STRATEGY_ID}
export VAMPIRE_MODE=vampire
export TIME_PER_TASK=${TIME_PER_TASK-60}
export VAMPIRE_SYMBOL_PRECEDENCE=scramble
export SOLVE_RUNS_PER_PROBLEM=${SOLVE_RUNS_PER_PROBLEM-16}

./vampire-array.sh "$@"
