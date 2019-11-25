#!/usr/bin/env bash

#SBATCH --mem=4G
#SBATCH --time=60

set -euo pipefail

source env.sh

OUTPUT=${OUTPUT:-out/default}

python -O aggregate.py --output "$OUTPUT/aggregate" "$OUTPUT/batches/**/batch.json"
