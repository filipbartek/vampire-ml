#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)/predicate-all-1000}

$(dirname $0)/run-predicate.sh --train-size 1.0 --test-size 1000 "$@"
