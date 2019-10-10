#!/usr/bin/env python3.7

import glob
import itertools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
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


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('result', type=str, nargs='*', help='glob pattern of result of a prove or probe run')
    parser.add_argument('--fields', help='names of fields to extract separated by commas')
    parser.add_argument('--source', nargs='+', action='append', choices=['stdout', 'symbols', 'clauses'], default=[],
                        help='data source file (output from Vampire runs)')
    parser.add_argument('--problem-list', action='append', default=[], help='input file with a list of problem paths')
    parser.add_argument('--problem-base-path', type=str, help='the problem paths are relative to the base path')
    parser.add_argument('--input-runs-pickle', help='load a previously saved runs pickle')
    parser.add_argument('--input-probe-runs-pickle', help='merge a previously saved probe runs pickle')
    parser.add_argument('--solve-runs-per-problem', type=int, default=1,
                        help='minimum number of runs for a problem to be considered interesting')
    parser.add_argument('--output', '-o', help='output directory')
    parser.add_argument('--plot-format', action='append', nargs='+',
                        help='Output plot formats recognized by `pyplot.savefig`. Recommended values: svg, png')
    parser.add_argument('--gui', action='store_true', help='open windows with histograms')
    parser.add_argument('--only-save-runs', action='store_true', help='do nothing except collecting and saving runs')


def call(namespace):
    if namespace.input_runs_pickle is None:
        runs_df = generate_runs_df(namespace.result, namespace.fields, list(itertools.chain(*namespace.source)),
                                   namespace.problem_base_path)
    else:
        runs_df = pd.read_pickle(namespace.input_runs_pickle)

    # We save the dataframe before we replace the NaNs in category columns with 'NA'.
    save_df(runs_df, 'runs', namespace.output)

    if namespace.only_save_runs:
        return

    print('Runs info:')
    runs_df.info()

    problem_paths = get_problem_paths(runs_df.problem_path, namespace.problem_list, namespace.problem_base_path)

    termination_fieldnames = [f for f in ['status', 'exit_code', 'termination_reason', 'termination_phase'] if
                              f in runs_df]

    problem_first_runs = pd.DataFrame(index=problem_paths)
    problem_first_runs.index.name = 'problem_path'
    problem_first_runs = problem_first_runs.join(runs_df.groupby('problem_path').first()).sort_values(
        termination_fieldnames + ['time_elapsed_process'])
    save_df(problem_first_runs, 'problem_first_runs', namespace.output)

    fill_category_na(runs_df)
    fill_category_na(problem_first_runs)

    # Distributions of some combinations of category fields
    print('Run termination distribution:', runs_df.groupby(termination_fieldnames).size(), sep='\n')
    print('Problem first run termination distribution:', problem_first_runs.groupby(termination_fieldnames).size(),
          sep='\n')

    problems_df = generate_problems_df(problem_paths, runs_df, namespace.input_probe_runs_pickle)
    print('Problems info:')
    problems_df.info()
    save_df(problems_df, 'problems', namespace.output)

    problems_interesting_df = problems_df[
        (problems_df.n_completed >= namespace.solve_runs_per_problem) & (problems_df.n_exit_0 >= 1) & (
                problems_df.n_exit_0 + problems_df.n_exit_1 == problems_df.n_total)]
    print(f'Number of interesting problems: {len(problems_interesting_df)}')
    # TODO: Sort the rows by more criteria, for example time_elapsed mean.
    if ('saturation_iterations', 'variation') in problems_interesting_df:
        problems_interesting_df = problems_interesting_df.sort_values(('saturation_iterations', 'variation'),
                                                                      ascending=False)
    save_df(problems_interesting_df, 'problems_interesting', namespace.output)

    if namespace.plot_format is not None:
        plot_formats = list(itertools.chain(*namespace.plot_format))
        plot(runs_df, namespace.output, namespace.gui, plot_formats)


def generate_runs_df(results, fields, sources, problem_base_path):
    result_paths = itertools.chain(*(glob.iglob(pattern, recursive=True) for pattern in results))
    runs = (run_database.Run(result_path, problem_base_path) for result_path in result_paths)
    if fields is not None:
        fields = tuple(fields)
    runs_df = run_database.Run.get_data_frame(runs, None, fields, sources)
    return runs_df


def save_df(df, name, output):
    if output is not None:
        os.makedirs(output, exist_ok=True)
        df.to_pickle(os.path.join(output, f'{name}.pkl'))
        df.to_csv(os.path.join(output, f'{name}.csv'))


def fill_category_na(df, value='NA'):
    # Replace NaN with 'NA' in category columns. This allows getting more useful statistics.
    for field in df.select_dtypes(['category']):
        series = df[field]
        assert value not in series.cat.categories
        series.cat.add_categories(value, inplace=True)
        series.fillna(value, inplace=True)


def get_problem_paths(initial_problem_paths, additional_problem_lists, problem_base_path):
    additional_problem_paths, _ = file_path_list.compose(additional_problem_lists, None, problem_base_path)
    problem_paths = list(dict.fromkeys(itertools.chain(initial_problem_paths, additional_problem_paths)))
    return problem_paths


def generate_problems_df(problem_abs_paths, runs_df, input_probe_runs_pickle):
    problems_df = pd.DataFrame(index=problem_abs_paths)
    problems_df.index.name = 'problem_path'
    problem_groups = runs_df.groupby(['problem_path'])
    problems_df = problems_df.join(problem_groups.size().astype(pd.UInt64Dtype()).to_frame('n_total'))
    problems_df = problems_df.join(
        runs_df[runs_df.status == 'completed'].groupby(['problem_path']).size().astype(pd.UInt64Dtype()).to_frame(
            'n_completed'))
    problems_df = problems_df.join(
        runs_df[runs_df.exit_code == 0].groupby(['problem_path']).size().astype(pd.UInt64Dtype()).to_frame('n_exit_0'))
    problems_df = problems_df.join(
        runs_df[runs_df.exit_code == 1].groupby(['problem_path']).size().astype(pd.UInt64Dtype()).to_frame('n_exit_1'))
    if 'termination_reason' in runs_df:
        problems_df = problems_df.join(
            runs_df[runs_df.termination_reason == 'Refutation'].groupby(['problem_path']).size().astype(
                pd.UInt64Dtype()).to_frame(
                'n_refutation'))
        problems_df = problems_df.join(
            runs_df[runs_df.termination_reason == 'Satisfiable'].groupby(['problem_path']).size().astype(
                pd.UInt64Dtype()).to_frame(
                'n_satisfiable'))
        problems_df = problems_df.join(
            runs_df[runs_df.termination_reason == 'Time limit'].groupby(['problem_path']).size().astype(
                pd.UInt64Dtype()).to_frame(
                'n_time_limit'))
    problems_df.fillna(
        {'n_total': 0, 'n_completed': 0, 'n_exit_0': 0, 'n_exit_1': 0, 'n_refutation': 0, 'n_satisfiable': 0,
         'n_time_limit': 0},
        inplace=True)
    problems_df = problems_df.join(problem_groups.agg([np.mean, np.std, scipy.stats.variation, np.min, np.max]))
    # Merge probe run results into `problems_df`
    if input_probe_runs_pickle is not None:
        probe_runs_df = pd.read_pickle(input_probe_runs_pickle)
        problems_df = problems_df.join(probe_runs_df[['problem_path', 'status', 'exit_code', 'termination_reason',
                                                      'termination_phase', 'time_elapsed_process', 'predicates_count',
                                                      'functions_count', 'clauses_count']].set_index('problem_path'),
                                       rsuffix='probe')
    return problems_df


def plot(runs_df, output, gui, formats):
    sns.set()

    numeric_fields_present = [field for field in numeric_fields.keys() if field in runs_df]
    df_successful = runs_df[runs_df.exit_code == 0]

    # Distributions of numeric fields
    for field, properties in numeric_fields.items():
        if field in runs_df:
            print(runs_df[field].describe())
            distplot(runs_df[field], properties['title'] + ' in all runs', properties['xlabel'], properties['ylabel'],
                     output, f'hist_{field}_all', formats)
            distplot(df_successful[field], properties['title'] + ' in successful runs', properties['xlabel'],
                     properties['ylabel'], output, f'hist_{field}_successful', formats)

    hue_field = 'exit_code'
    if 'termination_reason' in runs_df:
        hue_field = 'termination_reason'

    g = sns.pairplot(runs_df.dropna(subset=numeric_fields_present), vars=numeric_fields_present, diag_kind='hist',
                     hue=hue_field)
    g.fig.suptitle('All runs')
    if output is not None:
        savefig(os.path.join(output, f'pairs_all'), formats)

    g = sns.pairplot(df_successful.dropna(subset=numeric_fields_present), vars=numeric_fields_present, diag_kind='hist',
                     hue=hue_field)
    g.fig.suptitle('Successful runs')
    if output is not None:
        savefig(os.path.join(output, f'pairs_successful'), formats)

    if gui:
        plt.show()


def distplot(series, title, xlabel, ylabel, output_directory, output_name, formats):
    if series.count() == 0:
        logging.warning(f'Skipping distplot {output_name} because there is no valid data.')
        return
    output_path = None
    if output_directory is not None and output_name is not None:
        output_path = os.path.join(output_directory, output_name)
    plt.figure()
    sns.distplot(series.dropna(), kde=False, rug=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if output_path is not None:
        savefig(output_path, formats)
    if series.nunique() > 1:
        plt.figure()
        sns.distplot(series.dropna(), rug=True)
        plt.title(title)
        plt.xlabel(xlabel)
        if output_path is not None:
            savefig(f'{output_path}_kde', formats)


def savefig(name, formats):
    for format in formats:
        plt.savefig(f'{name}.{format}')
