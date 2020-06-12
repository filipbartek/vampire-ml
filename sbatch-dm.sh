#!/usr/bin/env bash

#SBATCH --mem-per-cpu=1G
#SBATCH --time=60

set -euo pipefail

. $(dirname $0)/env.sh

python -O -u proving/dm.py --jobs "$SLURM_CPUS_PER_TASK" "$@"
