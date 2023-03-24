#!/usr/bin/env bash

# Required options:
# checkpoint.restore
# problem.*

# Example usage:
# solve.sh checkpoint.restore=outputs/tf_ckpts/empirical/00-00 problem.names=[PUZ001+1]

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

export DGLBACKEND=tensorflow
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:--1}
export VAMPIRE=${VAMPIRE:-$(dirname "$0")/../vampire/build/bin/vampire}

python solve.py parallel.n_jobs=${N_JOBS:-1} $@
