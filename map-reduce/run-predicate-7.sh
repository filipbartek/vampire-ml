#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)/predicate-7}
export VAMPIRE_ML=${VAMPIRE_ML:-$(dirname $0)/..}
export PROBLEMS_TRAIN=${PROBLEMS_TRAIN-$VAMPIRE_ML/problems/predicate-7.txt}

$(dirname $0)/run-predicate.sh "$@"
