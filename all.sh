#!/usr/bin/env bash

set -euo pipefail

WORKSPACE=${WORKSPACE:-out/default}

WORKSPACE_PROBE=$WORKSPACE/probe
PROBE_JOB_ID=$(OUTPUT=$WORKSPACE_PROBE ./vampire-array-probe.sh)
RESULTS_BASE=$WORKSPACE_PROBE OUTPUT_ID=minimal sbatch --time=10 --job-name=stats-probe-minimal --dependency=afterok:"$PROBE_JOB_ID" stats.sh
PROBE_STATS_JOB_ID=$(RESULTS_BASE=$WORKSPACE_PROBE OUTPUT_ID=full sbatch --job-name=stats-probe-full --dependency=afterok:"$PROBE_JOB_ID" stats.sh --source stdout symbols clauses)

WORKSPACE_SCRAMBLE=$WORKSPACE/scramble
SCRAMBLE_JOB_ID=$(OUTPUT=$WORKSPACE_SCRAMBLE ./vampire-array-scramble.sh)
RESULTS_BASE=$WORKSPACE_SCRAMBLE OUTPUT_ID=stdout sbatch --job-name=stats-scramble-stdout --dependency=afterok:"$SCRAMBLE_JOB_ID" stats.sh --source stdout
RESULTS_BASE=$WORKSPACE_SCRAMBLE OUTPUT_ID=complete sbatch --job-name=stats-scramble-complete --dependency=afterok:"$SCRAMBLE_JOB_ID":"$PROBE_STATS_JOB_ID" stats.sh --source stdout --input-probe-runs-pickle "$WORKSPACE_PROBE/stats/full/runs.pkl"
