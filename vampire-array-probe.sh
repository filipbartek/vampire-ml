#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT-out/default-probe}
export VAMPIRE_MODE=clausify
export TIME_PER_TASK=${TIME_PER_TASK-8}

./vampire-array.sh "$@"
