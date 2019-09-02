#!/usr/bin/env bash

# Example usage:
# ./run_clausify.sh < problems.txt
# srun ./run_clausify.sh < problems.txt
# sbatch --array=0-3 --input=problems.txt run_clausify.sh

# Recommended time: (nproblems / arraysize) * 10s

# TODO: Calibrate the memory requirement.
#SBATCH --mem=1G
#SBATCH --requeue

set -euo pipefail

echo -e "$(env | sort)"

source env.sh

: "${PROBLEM_ID:=${SLURM_ARRAY_TASK_ID:-}}"
if [ -n "${SLURM_ARRAY_TASK_MAX-}" ]; then : "${PROBLEM_MODULUS:=$((SLURM_ARRAY_TASK_MAX+1))}"; fi
# -1 stands for infinite modulus.
: "${PROBLEM_MODULUS:=-1}"

: "${MAX_PROCS:=${SLURM_NTASKS:-1}}"
XARGS_COMMAND=(xargs --verbose "--max-args=1" "--max-procs=$MAX_PROCS" -I '{}' ./process_problem_srun.sh {})


if [ -n "${PROBLEM_ID:-}" ]
then
  if [ -n "${PROBLEM_MODULUS:-}" ] && [ -1 -ne "${PROBLEM_MODULUS:-}" ]
  then
    echo "Processing problems with id $PROBLEM_ID modulo $PROBLEM_MODULUS."
    sed -n "$((PROBLEM_ID+1))~${PROBLEM_MODULUS}p" | "${XARGS_COMMAND[@]}"
  else
    echo "Processing problem with id $PROBLEM_ID."
    sed -n "$((PROBLEM_ID+1))p" | "${XARGS_COMMAND[@]}"
  fi
else
  echo "Processing all problems."
  "${XARGS_COMMAND[@]}"
fi
