#!/usr/bin/env bash

#SBATCH --mem-per-cpu=3128
#SBATCH --time=600

# Usage examples:
# fit.sh < problems.txt
# OUTPUT=out/sp-random-predicate sbatch --input=problems/problems_selected_aggregated.txt --job-name=fit-predicate fit.sh --random-predicate-precedence

set -euo pipefail

. env.sh

OUTPUT=${OUTPUT:-out}
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

VAMPIRE_TIME_LIMIT=${VAMPIRE_TIME_LIMIT:-10}
VAMPIRE_MEMORY_LIMIT=${VAMPIRE_MEMORY_LIMIT:-3000}
SOLVE_RUNS_PER_PROBLEM=${SOLVE_RUNS_PER_PROBLEM:-1000}
CPUS=${CPUS:-${SLURM_CPUS_PER_TASK:-1}}

python_call=(python -O -u -m)
if [ -n "${DO_MPROF-}" ]; then
  python_call=(mprof run --python --include-children --interval 1 --exit-code --output "$OUTPUT_LOCAL/mprofile.dat")
fi

XARGS_COMMAND=(
  xargs
  "${python_call[@]}"
  vampire_ml
  --log ${LOG_LEVEL:-INFO}
  --jobs "$CPUS"
  fit
  --output "$OUTPUT_LOCAL"
  --train-solve-runs "$SOLVE_RUNS_PER_PROBLEM"
  --vampire "$VAMPIRE"
  --vampire-options "{time_limit: $VAMPIRE_TIME_LIMIT, memory_limit: $VAMPIRE_MEMORY_LIMIT}"
  --include "$TPTP"
  --problem-base-path "$TPTP_PROBLEMS"
  --timeout $((VAMPIRE_TIME_LIMIT + 10))
  "$@"
)

if [ -n "${PROBLEMS_TRAIN-}" ]; then XARGS_COMMAND+=(--problems-train "$PROBLEMS_TRAIN"); fi

if [ -n "${SLURM_JOB_ID-}" ]; then OUTPUT_SCRATCH=${OUTPUT_SCRATCH-/lscratch/$USER/slurm-$SLURM_JOB_ID}; fi

# See also https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html#how-to-recover-files-before-a-job-times-out
function finish {
  if [ -n "${OUTPUT_SCRATCH-}" ]; then
    echo Finish: Removing directory "$OUTPUT_SCRATCH"
    rm -rf "$OUTPUT_SCRATCH"
  fi
  if [ -n "${DO_MPROF-}" ]; then
    mprof plot --output "$OUTPUT_LOCAL/mprofile.png" "$OUTPUT_LOCAL/mprofile.dat"
  fi
}
trap finish EXIT INT TERM

if [ -n "${OUTPUT_SCRATCH-}" ]; then XARGS_COMMAND+=(--scratch "$OUTPUT_SCRATCH"); fi

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

echo "$problems" | "${XARGS_COMMAND[@]}"
