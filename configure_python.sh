#!/usr/bin/env bash

set -euo pipefail

source env.sh

if [ -n "${MODULESHOME-}" ]; then module load Python; fi
if [ ! -e "$VAMPIRE_ML_VENV" ]; then python3 -m virtualenv "$VAMPIRE_ML_VENV"; fi
source "$VAMPIRE_ML_VENV/bin/activate"
pip install -r requirements.txt
