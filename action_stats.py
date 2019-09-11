#!/usr/bin/env python3.7

import glob
import itertools
import sys

import run_database


def call(namespace):
    result_paths = list(itertools.chain(*(glob.iglob(pattern, recursive=True) for pattern in namespace.result)))
    mbr = run_database.MultiBatchResult(result_paths)
    fields = namespace.fields
    if fields is not None:
        fields = tuple(fields)
    excluded_sources = []
    if not namespace.source_stdout:
        excluded_sources.append('stdout')
    if not namespace.source_vampire_json:
        excluded_sources.append('vampire_json')
    df = mbr.get_runs_data_frame(fields, tuple(excluded_sources))
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
    parser.add_argument('--fields')
    parser.add_argument('--source-stdout', action='store_true')
    parser.add_argument('--source-vampire-json', action='store_true')
