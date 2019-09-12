#!/usr/bin/env python3.7

import glob
import itertools
import os
import sys

import pandas as pd

import file_path_list
import run_database


def call(namespace):
    if namespace.input_pickle is None:
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
    else:
        df = pd.read_pickle(namespace.input_pickle)

    os.makedirs(namespace.output, exist_ok=True)

    if len(namespace.problem_list) >= 1:
        problem_paths, _ = file_path_list.compose(namespace.problem_list)
        assert set(problem_paths) >= set(df.problem_path)
        problems_unprocessed = set(problem_paths) - set(df.problem_path)
        with open(os.path.join(namespace.output, 'problems_unprocessed.txt'), 'w') as f:
            f.writelines(f'{path}\n' for path in problems_unprocessed)

    # Distribution of combinations of status and exit code
    print(df.fillna({'exit_code': 0}).groupby(['status', 'exit_code']).size())

    df.to_pickle(os.path.join(namespace.output, 'stats.pkl'))
    df.to_csv(os.path.join(namespace.output, 'stats.csv'))
    with open(os.path.join(namespace.output, 'problems.txt'), 'w') as f:
        f.writelines(f'{path}\n' for path in df.problem_path)
    with open(os.path.join(namespace.output, 'problems_successful.txt'), 'w') as f:
        f.writelines(f'{path}\n' for path in df[df.exit_code == 0].problem_path)


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('result', type=str, nargs='*', help='glob pattern of result of a prove or probe run')
    parser.add_argument('--fields', help='names of fields to extract separated by commas')
    parser.add_argument('--source-stdout', action='store_true', help='include data from stdout.txt files')
    parser.add_argument('--source-vampire-json', action='store_true', help='include data from vampire.json files')
    parser.add_argument('--problem-list', action='append', default=[], help='input file with a list of problem paths')
    parser.add_argument('--input-pickle', help='load a previously saved stats pickle')
    parser.add_argument('--output', '-o', required=True, help='output directory')
