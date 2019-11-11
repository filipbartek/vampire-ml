#!/usr/bin/env bash

set -euo pipefail

WORKSPACE=${WORKSPACE:-out/default}

WORKSPACE_PROBE=$WORKSPACE/probe
PROBE_JOB_ID=$(OUTPUT=$WORKSPACE_PROBE ./vampire-array-probe.sh)
RESULTS_BASE=$WORKSPACE_PROBE OUTPUT_ID=minimal sbatch --time=10 --job-name=stats-probe-minimal --dependency=afterok:"$PROBE_JOB_ID" stats.sh
RESULTS_BASE=$WORKSPACE_PROBE OUTPUT_ID=full sbatch --time=240 --job-name=stats-probe-full --dependency=afterok:"$PROBE_JOB_ID" stats.sh --source stdout symbols clauses

WORKSPACE_SCRAMBLE=$WORKSPACE/scramble
SCRAMBLE_JOB_ID=$(OUTPUT=$WORKSPACE_SCRAMBLE ./vampire-array-scramble.sh)
RESULTS_BASE=$WORKSPACE_SCRAMBLE OUTPUT_ID=stdout sbatch --time=240 --job-name=stats-scramble --dependency=afterok:"$SCRAMBLE_JOB_ID" stats.sh --source stdout
