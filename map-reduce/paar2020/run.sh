#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)}
export REDUCE_CASES=${REDUCE_CASES:-$(dirname $0)/reduce_cases.txt}
export VAMPIRE_ML=${VAMPIRE_ML:-$(dirname $0)/../..}
export PROBLEMS=${PROBLEMS-$VAMPIRE_ML/problems/predicate-small-1024.txt}
export PROBLEMS_TRAIN=${PROBLEMS_TRAIN-$VAMPIRE_ML/problems/predicate-solvable.txt}

$(dirname $0)/../run-predicate.sh --test-size 1000 --train-solve-runs 100 --n-splits 5 --train-score 0 --refit-scorer success.count "$@"
