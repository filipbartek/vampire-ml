#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-out/default}
export STRATEGY_ID=${STRATEGY_ID:-probe}
export VAMPIRE_MODE=clausify
export CPUS_PER_TASK=${CPUS_PER_TASK:-4}
export TIME_PER_TASK=${TIME_PER_TASK:-8}

./vampire-array.sh
