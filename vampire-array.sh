#!/usr/bin/env bash

# This script is intended to be run in cluster.ciirc.cvut.cz.

set -euo pipefail

export OUTPUT=${OUTPUT:-out/default}
export VAMPIRE_MEMORY_LIMIT=${VAMPIRE_MEMORY_LIMIT:-8192}

# sbatch parameters
OUTPUT_SLURM=${OUTPUT_SLURM:-$OUTPUT/slurm}
PROBLEMS=${PROBLEMS:-problems_cnf_fof.txt}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
TIME_PER_TASK=${TIME_PER_TASK:-60}
PARTITION=${PARTITION:-compute}
ARRAY_TASK_COUNT=${ARRAY_TASK_COUNT:-1000}
ARRAY=${ARRAY:-0-$((ARRAY_TASK_COUNT - 1))}
JOB_NAME=${JOB_NAME:-vampire:$OUTPUT}

mkdir -p "$OUTPUT_SLURM"
sbatch --parsable --input="$PROBLEMS" --cpus-per-task="$CPUS_PER_TASK" --time="$TIME_PER_TASK" --mem-per-cpu=$((VAMPIRE_MEMORY_LIMIT + 128)) --partition="$PARTITION" --array="$ARRAY" --mail-type=FAIL,REQUEUE,STAGE_OUT --job-name="$JOB_NAME" --comment="$(git rev-parse --verify HEAD)" --output="$OUTPUT_SLURM/%A_%a.out" vampire.sh "$@"
