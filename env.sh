#!/usr/bin/env bash

# Source this file in bash to populate the environment with useful variables and activate the venv.

export VAMPIRE_ML=${VAMPIRE_ML:-$PWD}
export VAMPIRE_ML_VENV=${VAMPIRE_ML_VENV:-$VAMPIRE_ML/venv}
export VAMPIRE_DIR=${VAMPIRE_DIR:-$VAMPIRE_ML/vampire}
export VAMPIRE=${VAMPIRE:-$VAMPIRE_DIR/build/release/bin/vampire}
export VAMPIRE_MEMORY_LIMIT=${VAMPIRE_MEMORY_LIMIT:-8192}
export VAMPIRE_TIME_LIMIT=${VAMPIRE_TIME_LIMIT:-10}
export SOLVE_RUNS_PER_PROBLEM=${SOLVE_RUNS_PER_PROBLEM:-1000}

export OUTPUT=${OUTPUT:-out}

export TPTP_VERSION=${TPTP_VERSION:-TPTP-v7.3.0}
export TPTP=${TPTP:-$HOME/$TPTP_VERSION}
export TPTP_PROBLEMS=$TPTP/Problems

if [ -n "${MODULESHOME-}" ]; then module load Python; fi
if [ -e "$VAMPIRE_ML_VENV" ]; then . "$VAMPIRE_ML_VENV/bin/activate"; fi
