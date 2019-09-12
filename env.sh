#!/usr/bin/env bash

# Source this file in bash to populate the environment with useful variables and activate the venv.

export VAMPIRE_ML=${VAMPIRE_ML:-$PWD}
export VAMPIRE_ML_VENV=$VAMPIRE_ML/venv
export VAMPIRE_DIR=$VAMPIRE_ML/vampire
export VAMPIRE=$VAMPIRE_DIR/vampire_rel

export TPTP_VERSION=${TPTP_VERSION:-TPTP-v7.3.0}
export TPTP=${TPTP:-$HOME/$TPTP_VERSION}
export TPTP_PROBLEMS=$TPTP/Problems

if [ -n "${MODULESHOME-}" ]; then module load Python; fi
if [ -e "$VAMPIRE_ML_VENV" ]; then . "$VAMPIRE_ML_VENV/bin/activate"; fi
