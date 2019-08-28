#!/usr/bin/env bash

# Example usage:
# ./run_clausify.sh problems.txt
# srun ./run_clausify.sh problems.txt
# sbatch --array=0-3 run_clausify.sh problems.txt

# TODO: Calibrate the memory requirement.
#SBATCH --mem=1G

set -euo pipefail

echo -e "$(env | sort)"

source env.sh

PROBLEMS=$1

if [ -n "${SLURM_ARRAY_JOB_ID-}" ]; then : "${OUTPUT:=slurm-$SLURM_ARRAY_JOB_ID}"; fi
if [ -n "${SLURM_JOB_ID-}" ]; then : "${OUTPUT:=slurm-$SLURM_JOB_ID}"; fi
: "${OUTPUT:=out/run_clausify}"
: "${PROBLEM_ID:=${SLURM_ARRAY_TASK_ID:-}}"
if [ -n "${SLURM_ARRAY_TASK_MAX-}" ]; then : "${PROBLEM_MODULUS:=$((SLURM_ARRAY_TASK_MAX+1))}"; fi
# -1 stands for infinite modulus.
: "${PROBLEM_MODULUS:=-1}"
: "${MAX_PROCS:=${SLURM_NTASKS:-1}}"

if [ -n "${PROBLEM_ID:-}" ]
then
  if [ -n "${PROBLEM_MODULUS:-}" ] && [ -1 -ne "${PROBLEM_MODULUS:-}" ]
  then
    echo "Processing problems with id $PROBLEM_ID modulo $PROBLEM_MODULUS."
    sed -n "$((PROBLEM_ID+1))~${PROBLEM_MODULUS}p" "$PROBLEMS" | xargs --verbose --max-args=1 --max-procs="$MAX_PROCS" -I '{}' ./run_clausify_once.sh {} $OUTPUT/{}
  else
    echo "Processing problem with id $PROBLEM_ID."
    sed -n "$((PROBLEM_ID+1))p" "$PROBLEMS" | xargs --verbose --max-args=1 --max-procs="$MAX_PROCS" -I '{}' ./run_clausify_once.sh {} $OUTPUT/{}
  fi
else
  echo "Processing all problems."
  xargs --arg-file "$PROBLEMS" --verbose --max-args=1 --max-procs="$MAX_PROCS" -I '{}' ./run_clausify_once.sh {} $OUTPUT/{}
fi
