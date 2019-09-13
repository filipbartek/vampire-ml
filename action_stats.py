#!/usr/bin/env python3.7

import glob
import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import file_path_list
import run_database

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


def call(namespace):
    if namespace.input_pickle is None:
        result_paths = itertools.chain(*(glob.iglob(pattern, recursive=True) for pattern in namespace.result))
        runs = (run_database.Run(result_path) for result_path in result_paths)
        fields = namespace.fields
        if fields is not None:
            fields = tuple(fields)
        excluded_sources = []
        if not namespace.source_stdout:
            excluded_sources.append('stdout')
        if not namespace.source_vampire_json:
            excluded_sources.append('vampire_json')
        df = run_database.Run.get_data_frame(runs, None, fields, excluded_sources)
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

    sns.set()

    numeric_fields_present = [field for field in numeric_fields.keys() if field in df]
    df_successful = df[df.exit_code == 0]

    # Distributions of numeric fields
    for field, properties in numeric_fields.items():
        if field in df:
            print(df[field].describe())
            distplot(df[field], properties['title'] + ' in all runs', properties['xlabel'], properties['ylabel'],
                     namespace.output, f'hist_{field}_all')
            distplot(df_successful[field], properties['title'] + ' in successful runs', properties['xlabel'],
                     properties['ylabel'], namespace.output, f'hist_{field}_successful')

    hue_field = 'exit_code'
    if 'termination_reason' in df:
        hue_field = 'termination_reason'

    g = sns.pairplot(df.dropna(subset=numeric_fields_present), vars=numeric_fields_present, diag_kind='hist',
                     hue=hue_field)
    g.fig.suptitle('All runs')
    if namespace.output is not None:
        plt.savefig(os.path.join(namespace.output, f'pairs_all.svg'))

    g = sns.pairplot(df_successful.dropna(subset=numeric_fields_present), vars=numeric_fields_present, diag_kind='hist',
                     hue=hue_field)
    g.fig.suptitle('Successful runs')
    if namespace.output is not None:
        plt.savefig(os.path.join(namespace.output, f'pairs_successful.svg'))

    if namespace.gui:
        plt.show()

    # Save problem lists
    if namespace.output is not None:
        os.makedirs(namespace.output, exist_ok=True)
        if len(namespace.problem_list) >= 1:
            problem_paths, problem_base_path = file_path_list.compose(namespace.problem_list, None,
                                                                      namespace.problem_base_path)
            problem_abs_paths = set(
                os.path.abspath(os.path.join(problem_base_path, problem_path)) for problem_path in problem_paths)
            solved_problem_abs_paths = set(map(os.path.abspath, df.problem_path))
            assert problem_abs_paths >= solved_problem_abs_paths
            with open(os.path.join(namespace.output, 'problems_unprocessed.txt'), 'w') as f:
                for problem_path in problem_paths:
                    problem_abs_path = os.path.abspath(os.path.join(problem_base_path, problem_path))
                    if problem_abs_path not in solved_problem_abs_paths:
                        f.write(f'{problem_abs_path}\n')
        with open(os.path.join(namespace.output, 'problems.txt'), 'w') as f:
            f.writelines(f'{path}\n' for path in df.problem_path)
        with open(os.path.join(namespace.output, 'problems_successful.txt'), 'w') as f:
            f.writelines(f'{path}\n' for path in df[df.exit_code == 0].problem_path)


def distplot(series, title, xlabel, ylabel, output_directory, output_name):
    output_path = None
    if output_directory is not None and output_name is not None:
        output_path = os.path.join(output_directory, output_name)
    plt.figure()
    sns.distplot(series.dropna(), kde=False, rug=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if output_path is not None:
        plt.savefig(f'{output_path}.svg')
    if series.nunique() > 1:
        plt.figure()
        sns.distplot(series.dropna(), rug=True)
        plt.title(title)
        plt.xlabel(xlabel)
        if output_path is not None:
            plt.savefig(f'{output_path}_kde.svg')


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('result', type=str, nargs='*', help='glob pattern of result of a prove or probe run')
    parser.add_argument('--fields', help='names of fields to extract separated by commas')
    parser.add_argument('--source-stdout', action='store_true', help='include data from stdout.txt files')
    parser.add_argument('--source-vampire-json', action='store_true', help='include data from vampire.json files')
    parser.add_argument('--problem-list', action='append', default=[], help='input file with a list of problem paths')
    parser.add_argument('--problem-base-path', type=str, help='the problem paths are relative to the base path')
    parser.add_argument('--input-pickle', help='load a previously saved stats pickle')
    parser.add_argument('--output', '-o', help='output directory')
    parser.add_argument('--gui', action='store_true', help='open windows with histograms')
