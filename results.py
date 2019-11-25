#!/usr/bin/env python3.7

import os

import numpy as np
import pandas as pd
import scipy.stats


def save_df(df, base_name, output_dir):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df.to_pickle(os.path.join(output_dir, f'{base_name}.pkl'))
        df.to_csv(os.path.join(output_dir, f'{base_name}.csv'))


def save_terminations(solve_runs_df, output_batch):
    solve_runs_df = fill_category_na(solve_runs_df)
    termination_fieldnames = ['status', 'exit_code', 'termination_reason', 'termination_phase']

    # Distribution of run terminations
    terminations = solve_runs_df.groupby(termination_fieldnames).size()
    print('Distribution of solve run terminations:', terminations, sep='\n')
    print(terminations, file=open(os.path.join(output_batch, 'runs_solve_terminations.txt'), 'w'))


def save_problems(solve_runs_df, clausify_runs_df, output_batch, problem_paths=None, solve_runs=0):
    problems_df = generate_problems_df(solve_runs_df, clausify_runs_df, problem_paths)
    save_df(problems_df, 'problems', output_batch)

    problems_interesting_df = problems_df[
        (problems_df.n_completed >= solve_runs) & (problems_df.n_exit_0 >= 1) & (
                problems_df.n_exit_0 + problems_df.n_exit_1 == problems_df.n_total)]
    print(f'Number of interesting problems: {len(problems_interesting_df)}')
    # TODO: Sort the rows by more criteria, for example time_elapsed mean.
    if ('saturation_iterations', 'variation') in problems_interesting_df and len(problems_interesting_df.index) >= 2:
        problems_interesting_df.sort_values(('saturation_iterations', 'variation'), ascending=False, inplace=True)
    save_df(problems_interesting_df, 'problems_interesting', output_batch)


def fill_category_na(df, value='NA', inplace=False):
    if not inplace:
        df = df.copy()
    # Replace NaN with 'NA' in category columns. This allows getting more useful statistics.
    for field in df.select_dtypes(['category']):
        series = df[field]
        assert value not in series.cat.categories
        series.cat.add_categories(value, inplace=True)
        series.fillna(value, inplace=True)
    return df


def generate_problems_df(runs_df, probe_runs_df=None, problem_paths=None):
    if problem_paths is None:
        problem_paths = runs_df.problem_path.drop_duplicates()
    # TODO: Ensure duplicate index values are not introduced.
    problems_df = pd.DataFrame(index=problem_paths)
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
    agg_fields = ['time_elapsed_process', 'time_elapsed_vampire', 'memory_used', 'saturation_iterations']
    agg_functions = [np.mean, np.std, scipy.stats.variation, np.min, np.max]
    problems_df = problems_df.join(problem_groups.agg({field_name: agg_functions for field_name in agg_fields}))
    # Merge probe run results into `problems_df`
    if probe_runs_df is not None:
        problems_df = problems_df.join(
            probe_runs_df[['problem_path', 'predicates_count', 'functions_count', 'clauses_count']].drop_duplicates(
                'problem_path').set_index('problem_path'),
            rsuffix='probe')
    return problems_df