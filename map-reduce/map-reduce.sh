#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)}
export VAMPIRE_ML=${VAMPIRE_ML:-$(dirname $0)/..}
export PROBLEMS=${PROBLEMS-$VAMPIRE_ML/problems/cnf_fof.txt}

echo OUTPUT=$OUTPUT
echo VAMPIRE_ML=$VAMPIRE_ML
echo PROBLEMS=$PROBLEMS

echo Parameters: "$@"

if [ -n "${MAP_MAX_PROCS-}" ]; then $(dirname $0)/map.sh "$@"; fi

$(dirname $0)/reduce.sh "$@"
