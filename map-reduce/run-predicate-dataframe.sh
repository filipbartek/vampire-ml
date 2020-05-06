#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)/predicate-dataframe}
export VAMPIRE_ML=${VAMPIRE_ML:-$(dirname $0)/..}

$(dirname $0)/run-predicate.sh --problems-dataframe --precompute-only "$@"
