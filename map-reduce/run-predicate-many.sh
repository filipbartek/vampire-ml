#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)/predicate-many}
export PROBLEMS=${PROBLEMS-$VAMPIRE_ML/problems/predicate-small-1024.txt}
export PROBLEMS_TRAIN=${PROBLEMS_TRAIN-$VAMPIRE_ML/problems/predicate-1.txt}

$(dirname $0)/run-predicate.sh --train-size 100 --test-size 1000 --train-solve-runs 100 "$@"
