#!/usr/bin/env bash

#SBATCH --mem-per-cpu=3128
#SBATCH --time=600

# Usage examples:
# fit.sh < problems.txt
# OUTPUT=out/sp-random-predicate sbatch --input=problems/problems_selected_aggregated.txt --job-name=fit-predicate fit.sh --random-predicate-precedence

set -euo pipefail

. $(dirname $0)/env.sh

# https://scikit-learn.org/stable/modules/computing.html#parallel-numpy-routines-from-numerical-libraries
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=1

if [ -n "${SLURM_ARRAY_JOB_ID-}" ] && [ -n "${SLURM_ARRAY_TASK_ID-}" ]; then
  JOB_ID=${JOB_ID-$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID}
fi
if [ -n "${SLURM_JOB_ID-}" ]; then
  JOB_ID=${JOB_ID-$SLURM_JOB_ID}
fi
if [ -n "${JOB_ID-}" ]; then
  OUTPUT_LOCAL=$OUTPUT/$JOB_ID
fi
OUTPUT_LOCAL=${OUTPUT_LOCAL:-$OUTPUT/default}

if [ -n "${LOCAL_CACHE-}" ]; then export XDG_CACHE_HOME=$OUTPUT_LOCAL/.cache; fi

CPUS=${CPUS:-${SLURM_CPUS_PER_TASK:-1}}

python_call=(python -O -u -m)
if [ -n "${DO_MPROF-}" ]; then
  python_call=(mprof run --python --include-children --interval 1 --exit-code --output "$OUTPUT_LOCAL/mprofile.dat")
fi

export PYTHONPATH=${PYTHONPATH-}:$(dirname $0)

XARGS_COMMAND=(
  xargs
  "${python_call[@]}"
  vampire_ml
  --log-output "$OUTPUT_LOCAL/vampire_ml.log"
  --log-level ${LOG_LEVEL:-INFO}
  --log-config "$VAMPIRE_ML/logging.conf"
  --jobs "$CPUS"
  fit
  --output "$OUTPUT_LOCAL"
  --train-solve-runs "$SOLVE_RUNS_PER_PROBLEM"
  --vampire "$VAMPIRE"
  --vampire-options "{time_limit: $VAMPIRE_TIME_LIMIT, memory_limit: $VAMPIRE_MEMORY_LIMIT}"
  --timeout $((VAMPIRE_TIME_LIMIT + 10))
  "$@"
)

if [ -n "${PROBLEMS-}" ]; then XARGS_COMMAND+=(--problem-list "$PROBLEMS"); fi
if [ -n "${PROBLEMS_TRAIN-}" ]; then XARGS_COMMAND+=(--train-problem-list "$PROBLEMS_TRAIN"); fi

if [ -n "${SLURM_JOB_ID-}" ]; then SCRATCH=${SCRATCH-/lscratch/$USER/slurm-$SLURM_JOB_ID}; fi

# See also https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html#how-to-recover-files-before-a-job-times-out
function finish {
  if [ -n "${SCRATCH-}" ]; then
    echo Finish: Removing directory "$SCRATCH"
    rm -rf "$SCRATCH"
  fi
  if [ -n "${DO_MPROF-}" ]; then
    mprof plot --output "$OUTPUT_LOCAL/mprofile.png" "$OUTPUT_LOCAL/mprofile.dat"
  fi
}
trap finish EXIT INT TERM

PROBLEM_ID=${PROBLEM_ID:-${SLURM_ARRAY_TASK_ID:-}}
if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ]; then PROBLEM_MODULUS=${PROBLEM_MODULUS:-$((SLURM_ARRAY_TASK_MAX + 1))}; fi
# -1 stands for infinite modulus.
PROBLEM_MODULUS=${PROBLEM_MODULUS:--1}

if [ -n "${PROBLEM_ID:-}" ]; then
  if [ -n "${PROBLEM_MODULUS:-}" ] && [ -1 -ne "${PROBLEM_MODULUS:-}" ]; then
    echo "Processing problems with id $PROBLEM_ID modulo $PROBLEM_MODULUS."
    problems=$(sed -n "$((PROBLEM_ID + 1))~${PROBLEM_MODULUS}p")
  else
    echo "Processing problem with id $PROBLEM_ID."
    problems=$(sed -n "$((PROBLEM_ID + 1))p")
  fi
else
  echo "Processing all problems."
  problems=$(cat) || problems=""
fi

problem_count=$(echo "$problems" | wc -l)
echo "problem_count=$problem_count"
problem_first=$(echo "$problems" | head -1)
echo "problem_first=$problem_first"

if [ -n "${SLURM_JOB_ID-}" ]; then
  scontrol update job "$SLURM_JOB_ID" Comment="$problem_first" || true
fi

echo "Storing configuration into $OUTPUT_LOCAL"
mkdir -p "$OUTPUT_LOCAL"
env | sort >"$OUTPUT_LOCAL/env.txt"
echo "$@" >"$OUTPUT_LOCAL/parameters.txt"
echo "$problems" >"$OUTPUT_LOCAL/problems.txt"
echo "${XARGS_COMMAND[@]}" >"$OUTPUT_LOCAL/xargs-command.sh"

echo "$problems" | "${XARGS_COMMAND[@]}" &>$OUTPUT_LOCAL/stdout.txt
