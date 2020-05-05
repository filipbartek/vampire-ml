#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)}
export JOB_ID=${JOB_ID:-reduce}
export VAMPIRE_ML=${VAMPIRE_ML:-$(dirname $0)/..}
PROBLEMS=${PROBLEMS:-$VAMPIRE_ML/problems/cnf_fof.txt}

echo OUTPUT=$OUTPUT
echo JOB_ID=$JOB_ID
echo VAMPIRE_ML=$VAMPIRE_ML
echo PROBLEMS=$PROBLEMS

echo Parameters: "$@"

$(realpath $VAMPIRE_ML/fit.sh) --problem-list $PROBLEMS "$@"
