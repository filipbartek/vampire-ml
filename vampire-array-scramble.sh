#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT-out/default}
export STRATEGY_ID=${STRATEGY_ID-scramble}
export VAMPIRE_MODE=vampire
export CPUS_PER_TASK=${CPUS_PER_TASK-16}
export TIME_PER_TASK=${TIME_PER_TASK-68}
export VAMPIRE_SYMBOL_PRECEDENCE=scramble
export SOLVE_RUNS_PER_PROBLEM=${SOLVE_RUNS_PER_PROBLEM-16}

./vampire-array.sh "$@"
