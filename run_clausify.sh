#!/usr/bin/env bash

set -euo pipefail

source env.sh

PROBLEMS=$1
OUTPUT=out_clausify

VAMPIRE_COMMAND="$VAMPIRE --include $TPTP --mode clausify --time_limit 1 --json_output $OUTPUT/{}/vampire.json > $OUTPUT/{}/stdout.txt 2> $OUTPUT/{}/stderr.txt"
xargs --arg-file=$PROBLEMS --max-args=1 --max-procs=$SLURM_NTASKS -I '{}' srun sh -c "$VAMPIRE_COMMAND"
