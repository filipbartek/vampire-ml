#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)}/reduce
export VAMPIRE_ML=${VAMPIRE_ML:-$(dirname $0)/..}
export PROBLEMS=${PROBLEMS-$VAMPIRE_ML/problems/cnf_fof.txt}
MAX_PROCS=${MAX_PROCS:-1}

echo OUTPUT=$OUTPUT
echo VAMPIRE_ML=$VAMPIRE_ML
echo PROBLEMS=$PROBLEMS
echo MAX_PROCS=$MAX_PROCS

echo Parameters: "$@"

if [ -n "${REDUCE_CASES-}" ]; then
  CASE_COUNT=$(wc -l <$REDUCE_CASES)
  cat $REDUCE_CASES | xargs --max-args=2 --max-procs=$CASE_COUNT --process-slot-var=JOB_ID --verbose $(realpath $VAMPIRE_ML/fit.sh) "$@"
else
  $(realpath $VAMPIRE_ML/fit.sh) "$@"
fi
