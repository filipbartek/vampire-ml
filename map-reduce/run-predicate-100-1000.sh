#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)/predicate-100-1000}

$(dirname $0)/run-predicate.sh --train-size 100 --test-size 1000 "$@"
