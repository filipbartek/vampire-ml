#!/usr/bin/env python3.7

import glob
import itertools
import sys

import run_database


def call(namespace):
    result_paths = list(itertools.chain(*(glob.iglob(pattern, recursive=True) for pattern in namespace.result)))
    mbr = run_database.MultiBatchResult(result_paths)
    df = mbr.runs_data_frame
    if namespace.output_pickle is not None:
        df.to_pickle(namespace.output_pickle)
    if namespace.output_csv is not None:
        if namespace.output_csv == '-':
            df.to_csv(sys.stdout)
        else:
            df.to_csv(namespace.output_csv)


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('result', type=str, nargs='+', help='glob pattern of result of a prove or probe run')
    parser.add_argument('--output-csv', help='output CSV runs document')
    parser.add_argument('--output-pickle')
