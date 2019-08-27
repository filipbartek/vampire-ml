#!/usr/bin/env bash

# To parallelize building of Vampire, set e.g. `MAKEFLAGS=-j4`.

set -euo pipefail

source env.sh

git submodule update --init --recursive
cd vampire
if [ -v "${MODULESHOME-}" ]; then module load GCC; fi
make vampire_rel
