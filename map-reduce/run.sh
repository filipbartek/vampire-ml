#!/usr/bin/env bash

set -euo pipefail

export OUTPUT=${OUTPUT:-$(dirname $0)}/$(date +"%Y-%m-%d_%H-%M-%S")

mkdir -p $OUTPUT

nohup nice $(realpath $(dirname $0))/map-reduce.sh "$@" &>$OUTPUT/stdout.txt &

# https://stackoverflow.com/a/17389526/4054250
pid=$!
echo "pid=$pid"
echo $pid >$OUTPUT/pid.txt

# We echo to get rid of leading whitespace.
pgid=$(echo $(ps p $pid o pgid=))
echo "pgid=$pgid"
echo $pgid >$OUTPUT/pgid.txt

echo "kill -- -$pgid" >$OUTPUT/kill.sh
chmod a+x $OUTPUT/kill.sh
