#!/usr/bin/env bash

set -euo pipefail

source env.sh

: "${EXITSTATUS_EXPECTED:=0}"
export EXITSTATUS_EXPECTED

# https://github.com/koalaman/shellcheck/wiki/SC2156
find "$OUTPUT" -name "$EXITSTATUS_FILENAME" -exec bash -c 'if [[ $(< "$1") == "$EXITSTATUS_EXPECTED" ]]; then dirname "$1"; fi' _ {} \; | xargs realpath --relative-to "$OUTPUT"
