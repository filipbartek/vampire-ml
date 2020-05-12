#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)}/map
export VAMPIRE_ML=${VAMPIRE_ML:-$(dirname $0)/..}
PROBLEMS=${PROBLEMS:-$VAMPIRE_ML/problems/cnf_fof.txt}
MAX_PROCS=${MAX_PROCS:-${MAP_MAX_PROCS:-1}}

echo OUTPUT=$OUTPUT
echo VAMPIRE_ML=$VAMPIRE_ML
echo PROBLEMS=$PROBLEMS
echo MAX_PROCS=$MAX_PROCS

echo Parameters: "$@"

xargs --max-args=1 --max-procs=$MAX_PROCS --process-slot-var=JOB_ID --verbose $(realpath $VAMPIRE_ML/fit.sh) --precompute selection --precompute-only "$@" <$PROBLEMS
