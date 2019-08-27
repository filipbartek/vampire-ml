#!/usr/bin/env bash

# To parallelize building of Vampire, set e.g. `MAKEFLAGS=-j4`.

set -euo pipefail

./configure_python.sh
./configure_vampire.sh
