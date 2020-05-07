#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)/predicate}
export VAMPIRE_ML=${VAMPIRE_ML:-$(dirname $0)/..}
export PROBLEMS=${PROBLEMS:-$VAMPIRE_ML/problems/predicate-small-10000.txt}
export PROBLEMS_TRAIN=${PROBLEMS_TRAIN:-$VAMPIRE_ML/problems/predicate-easy-variation.txt}

$(dirname $0)/run.sh --random-predicate-precedence "$@"
