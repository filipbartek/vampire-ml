#!/usr/bin/env bash

set -euo pipefail

PROBLEM=$1

if command -v srun > /dev/null
then
  srun --ntasks=1 ./process_problem.sh "$PROBLEM"
else
  ./process_problem.sh "$PROBLEM"
fi
