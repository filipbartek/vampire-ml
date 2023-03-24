#!/usr/bin/env bash

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

export VAMPIRE=${VAMPIRE:-$(dirname "$0")/../vampire/build/bin/vampire}
WORKSPACE=${WORKSPACE:-workspace}
PROBLEMS=${PROBLEMS:-$WORKSPACE/problems/train_val.txt}

python -O $(dirname "$0")/collect.py workspace_dir=$WORKSPACE problem.list_file=$PROBLEMS parallel.n_jobs=${N_JOBS:-1} $@
