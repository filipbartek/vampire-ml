#!/usr/bin/env bash

# Example usage:
# ./process_problems.sh < problems.txt
# srun ./process_problems.sh < problems.txt
# sbatch --array=0-999 --input=problems.txt --time=3 process_problems.sh

# Maximum expected time per array job: ceil(nproblems / $((SLURM_ARRAY_TASK_MAX+1)))) * ${VAMPIRE_TIME_LIMIT_PROBE:-1} + ${VAMPIRE_TIME_LIMIT_SOLVE:-10} * ${SOLVE_RUNS_PER_PROBLEM:-1}
# With array of 1000 jobs and 16094 problems, we have at most 17 problems per job.
# With 16 solve runs per problem, each problem takes at most 161s.
# Each job takes at most 2737s = 45m 37s = 45:37
# We can parallelize each job to up to 17 * 16 = 272 threads. We should parallelize to at least 16 threads.

##SBATCH --time=50
##SBATCH --cpus-per-task=16

# TODO: Calibrate the memory requirement.
#SBATCH --mem=1G
#SBATCH --requeue

set -euo pipefail

echo -e "$(env | sort)"

source env.sh

: "${BATCH_ID:=${SLURM_ARRAY_TASK_ID:-}}"
export BATCH_ID
if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ]; then : "${PROBLEM_MODULUS:=$((SLURM_ARRAY_TASK_MAX + 1))}"; fi
# -1 stands for infinite modulus.
: "${PROBLEM_MODULUS:=-1}"

: "${MAX_PROCS:=${SLURM_NTASKS:-1}}"
XARGS_COMMAND=(xargs --verbose "--max-procs=$MAX_PROCS" ./process_problem_srun.sh)

if [ -n "${BATCH_ID:-}" ]; then
  if [ -n "${PROBLEM_MODULUS:-}" ] && [ -1 -ne "${PROBLEM_MODULUS:-}" ]; then
    echo "Processing problems with id $BATCH_ID modulo $PROBLEM_MODULUS."
    sed -n "$((BATCH_ID + 1))~${PROBLEM_MODULUS}p" | "${XARGS_COMMAND[@]}"
  else
    echo "Processing problem with id $BATCH_ID."
    sed -n "$((BATCH_ID + 1))p" | "${XARGS_COMMAND[@]}"
  fi
else
  echo "Processing all problems."
  "${XARGS_COMMAND[@]}"
fi
