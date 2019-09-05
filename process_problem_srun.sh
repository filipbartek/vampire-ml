#!/usr/bin/env bash

set -euo pipefail

if command -v srun >/dev/null; then
  srun --ntasks=1 ./process_problem.sh "$@"
else
  ./process_problem.sh "$@"
fi
