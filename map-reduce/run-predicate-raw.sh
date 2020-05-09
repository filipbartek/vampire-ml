#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)/predicate-raw}

$(dirname $0)/run-predicate.sh --train-size 100 --test-size 1000 --cases default raw_scores default_heuristic random best_encountered heuristics "$@"
