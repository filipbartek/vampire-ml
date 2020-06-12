#!/usr/bin/env bash

#SBATCH --mem-per-cpu=1G
#SBATCH --time=60

set -euo pipefail

. env.sh

python -O -u proving/dm.py "$@"
