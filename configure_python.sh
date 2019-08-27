#!/usr/bin/env bash

set -euo pipefail

source env.sh

if [ -v "${MODULESHOME-}" ]; then module load Python; fi
python3 -m virtualenv $VAMPIRE_ML_VENV
source $VAMPIRE_ML_VENV/bin/activate
pip install -r requirements.txt
