#!/usr/bin/env bash

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

export DGLBACKEND=tensorflow
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:--1}
export VAMPIRE=${VAMPIRE:-$(dirname "$0")/../vampire/build/bin/vampire}
WORKSPACE=${WORKSPACE:-workspace}

# Additional environment variables that influence the training:
# XDG_CACHE_HOME: Some disk caching is done in this directory.

python train.py workspace_dir=$WORKSPACE parallel.n_jobs=${N_JOBS:-1} $@
