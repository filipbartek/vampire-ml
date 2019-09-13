#!/usr/bin/env python3.7

import glob
import itertools
import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import file_path_list
import run_database


def call(namespace):
    if namespace.input_pickle is None:
        result_paths = list(itertools.chain(*(glob.iglob(pattern, recursive=True) for pattern in namespace.result)))
        if len(result_paths) == 0:
            logging.error('No results were given.', namespace.result)
            return
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

    # We save the dataframe before we replace the NaNs in category columns with 'NA'.
    if namespace.output is not None:
        os.makedirs(namespace.output, exist_ok=True)
        df.to_pickle(os.path.join(namespace.output, 'stats.pkl'))
        df.to_csv(os.path.join(namespace.output, 'stats.csv'))

    print(df.info())

    # Replace NaN with 'NA' in category columns. This allows getting more useful statistics.
    for field in df.select_dtypes(['category']):
        series = df[field]
        assert 'NA' not in series.cat.categories
        series.cat.add_categories('NA', inplace=True)
        series.fillna('NA', inplace=True)
    assert 'status' in df and 'exit_code' in df

    # Distributions of some combinations of category fields
    print(df.groupby(['status', 'exit_code']).size())
    if 'termination_reason' in df and 'termination_phase' in df:
        print(df.groupby(['status', 'exit_code', 'termination_reason', 'termination_phase']).size())

    numeric_fields = {
        'time_elapsed_process': {
            'title': 'Time elapsed (process)',
            'xlabel': 'Time elapsed [seconds]',
            'ylabel': 'Runs'
        },
        'memory_used': {
            'title': 'Memory used',
            'xlabel': 'Memory used [kilobytes]',
            'ylabel': 'Runs'
        },
        'saturation_iterations': {
            'title': 'Main saturation loop iterations',
            'xlabel': 'Saturation iterations',
            'ylabel': 'Runs'
        },
        'predicates_count': {
            'title': 'Number of predicate symbols',
            'xlabel': 'Predicates',
            'ylabel': 'Runs'
        },
        'functions_count': {
            'title': 'Number of function symbols',
            'xlabel': 'Functions',
            'ylabel': 'Runs'
        },
        'clauses_count': {
            'title': 'Number of clauses',
            'xlabel': 'Clauses',
            'ylabel': 'Runs'
        }
    }

    # Distributions of numeric fields
    for field, properties in numeric_fields.items():
        if field in df:
            print(df[field].describe())
            df.hist(field, bins=100)
            plt.title(properties['title'] + ' in all runs')
            plt.xlabel(properties['xlabel'])
            plt.ylabel(properties['ylabel'])
            if namespace.output is not None:
                plt.savefig(os.path.join(namespace.output, f'hist_{field}_all.svg'))
            df[df.exit_code == 0].hist(field, bins=100)
            plt.title(properties['title'] + ' in successful runs')
            plt.xlabel(properties['xlabel'])
            plt.ylabel(properties['ylabel'])
            if namespace.output is not None:
                plt.savefig(os.path.join(namespace.output, f'hist_{field}_successful.svg'))
    if namespace.gui:
        plt.show()

    # Save problem lists
    if namespace.output is not None:
        os.makedirs(namespace.output, exist_ok=True)
        if len(namespace.problem_list) >= 1:
            problem_paths, _ = file_path_list.compose(namespace.problem_list)
            assert set(problem_paths) >= set(df.problem_path)
            problems_unprocessed = set(problem_paths) - set(df.problem_path)
            with open(os.path.join(namespace.output, 'problems_unprocessed.txt'), 'w') as f:
                f.writelines(f'{path}\n' for path in problems_unprocessed)
        with open(os.path.join(namespace.output, 'problems.txt'), 'w') as f:
            f.writelines(f'{path}\n' for path in df.problem_path)
        with open(os.path.join(namespace.output, 'problems_successful.txt'), 'w') as f:
            f.writelines(f'{path}\n' for path in df[df.exit_code == 0].problem_path)


def add_arguments(parser):
    parser.set_defaults(action=call)
    # TODO: Allow initializing directly by glob pattern of result or run configuration files.
    parser.add_argument('result', type=str, nargs='*', help='glob pattern of result of a prove or probe run')
    parser.add_argument('--fields', help='names of fields to extract separated by commas')
    parser.add_argument('--source-stdout', action='store_true', help='include data from stdout.txt files')
    parser.add_argument('--source-vampire-json', action='store_true', help='include data from vampire.json files')
    parser.add_argument('--problem-list', action='append', default=[], help='input file with a list of problem paths')
    parser.add_argument('--input-pickle', help='load a previously saved stats pickle')
    parser.add_argument('--output', '-o', help='output directory')
    parser.add_argument('--gui', action='store_true', help='open windows with histograms')
