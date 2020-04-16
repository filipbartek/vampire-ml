#!/usr/bin/env bash

# To parallelize building of Vampire, set e.g. `MAKEFLAGS=-j4`.

set -euo pipefail

. env.sh

git submodule update --init --recursive
cd "$VAMPIRE_DIR"
if [ -n "${MODULESHOME-}" ]; then module load GCC; fi
make vampire_rel
