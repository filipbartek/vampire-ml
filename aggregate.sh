#!/usr/bin/env bash

#SBATCH --mem=4G
#SBATCH --time=60

set -euo pipefail

source env.sh

OUTPUT=${OUTPUT:-out/default}

if [ -n "${ARRAY_JOB_ID-}" ]; then
  BATCHES_DIR=${BATCHES_DIR:-$OUTPUT/batches/$ARRAY_JOB_ID}
fi
BATCHES_DIR=${BATCHES_DIR:-$OUTPUT/batches}

OUTPUT_AGGREGATE=${OUTPUT_AGGREGATE:-$BATCHES_DIR/aggregate}
BATCHES=${BATCHES:-$BATCHES_DIR/**/batch.json}

python -O aggregate.py --output "$OUTPUT_AGGREGATE" "$BATCHES"
