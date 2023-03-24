#!/usr/bin/env bash

# Required environment variables:
# TPTP

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

WORKSPACE=${WORKSPACE:-workspace}
OUT_DIR=${OUT_DIR:-$WORKSPACE/problems}
ALL_SORTED=${ALL_SORTED:-$OUT_DIR/sorted.txt}
ALL_SHUFFLED=${ALL_SHUFFLED:-$OUT_DIR/shuffled.txt}
# In TPTP 8.1.2, there are 17436 FOF and CNF problems in total. 3487 is approximately 20 % of 17436.
TEST_PROBLEMS=${TEST_PROBLEMS:-3487}

# Find all FOF and CNF problems in TPTP.
find $TPTP/Problems -regextype sed -regex ".*/[A-Z]\{3\}[0-9]\{3\}[-+][1-9][0-9]*\(\.[0-9]\{3\}\)*\.p" -printf "%P\n" | sort > $ALL_SORTED
shuf $ALL_SORTED --output $ALL_SHUFFLED
head -n $TEST_PROBLEMS $ALL_SHUFFLED > $OUT_DIR/test.txt
tail -n +$((TEST_PROBLEMS + 1)) $ALL_SHUFFLED > $OUT_DIR/train_val.txt
