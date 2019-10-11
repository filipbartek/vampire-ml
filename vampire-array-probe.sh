#!/usr/bin/env bash

set -euo pipefail

export STRATEGY_ID=${STRATEGY_ID-probe}
export OUTPUT=${OUTPUT-out/default/$STRATEGY_ID}
export VAMPIRE_MODE=clausify
export CPUS_PER_TASK=${CPUS_PER_TASK-4}
export TIME_PER_TASK=${TIME_PER_TASK-8}

./vampire-array.sh "$@"
