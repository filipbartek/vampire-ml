#!/usr/bin/env bash

set -euo pipefail

source env.sh

TPTP_TGZ="$TPTP_VERSION.tgz"

# http://www.tptp.org/
wget "http://www.tptp.org/TPTP/Distribution/$TPTP_TGZ" -P /tmp
tar -xzf "/tmp/$TPTP_TGZ" -C "$HOME"

find "$TPTP_PROBLEMS" -regextype sed -regex ".*/[A-Z]\{3\}[0-9]\{3\}[-+][1-9][0-9]*\(\.[0-9]\{3\}\)*\.p" -exec realpath --relative-to "$TPTP_PROBLEMS" {} + | sort > problems_cnf_fof.txt
