#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)/predicate-easy}
export PROBLEMS=${PROBLEMS-$VAMPIRE_ML/problems/predicate-easy-variation.txt}
export PROBLEMS_TRAIN=${PROBLEMS_TRAIN-}

$(dirname $0)/run-predicate.sh --n-splits 5 --train-solve-runs 1000 "$@"
