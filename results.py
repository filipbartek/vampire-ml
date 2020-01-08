#!/usr/bin/env python3.7

import logging
import os

import numpy as np
import pandas as pd
import scipy.stats


def save_all(df_solve, df_clausify, output):
    save_df(df_solve, 'runs_solve', output)
    save_terminations(df_solve, output)
    save_df(df_clausify, 'runs_clausify', output)
    save_problems(df_solve, df_clausify, output)


def save_df(df, base_name, output_dir):
    if df is None:
        return
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df.to_pickle(os.path.join(output_dir, f'{base_name}.pkl'))
        df.to_csv(os.path.join(output_dir, f'{base_name}.csv'))
        logging.info(f'DataFrame of length {len(df.index)} saved: {os.path.join(output_dir, base_name)}.{{pkl,csv}}')


def save_terminations(solve_runs_df, output_batch):
    solve_runs_df = fill_category_na(solve_runs_df)
    termination_fieldnames = ['status', 'exit_code', 'termination_reason', 'termination_phase']

    # Distribution of run terminations
    terminations = solve_runs_df.groupby(termination_fieldnames).size()
    print('Distribution of solve run terminations:', terminations, sep='\n')
    print(terminations, file=open(os.path.join(output_batch, 'runs_solve_terminations.txt'), 'w'))


def save_problems(solve_runs_df, clausify_runs_df, output_batch, problem_paths=None, problem_base_path=None,
                  solve_runs=0):
    problems_df = generate_problems_df(solve_runs_df, clausify_runs_df, problem_paths, problem_base_path)
    save_df(problems_df, 'problems', output_batch)

    problems_interesting_df = problems_df[
        (problems_df.n_completed >= solve_runs) & (problems_df.n_exit_0 >= 1) & (
                problems_df.n_exit_0 + problems_df.n_exit_1 == problems_df.n_total)]
    print(f'Number of interesting problems: {len(problems_interesting_df)}')
    # TODO: Sort the rows by more criteria, for example time_elapsed mean.
    if ('saturation_iterations', 'variation') in problems_interesting_df and len(problems_interesting_df.index) >= 2:
        problems_interesting_df = problems_interesting_df.sort_values(('saturation_iterations', 'variation'),
                                                                      ascending=False)
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


def generate_problems_df(runs_df, probe_runs_df=None, problem_paths=None, problem_base_path=None):
    if problem_paths is None:
        problem_paths = runs_df.problem_path
    elif problem_base_path is not None:
        problem_paths = [os.path.relpath(problem_path, problem_base_path) for problem_path in problem_paths]
    problem_paths = pd.Index(problem_paths).drop_duplicates()
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

    def variation(a):
        return scipy.stats.variation(a.astype(np.float), nan_policy='omit')

    agg_functions = [np.mean, np.std, variation, np.min, np.max]
    # Aggregate memory across all runs
    problems_df = problems_df.join(runs_df.groupby(['problem_path']).agg(
        {field_name: agg_functions for field_name in ['memory_used']}))
    # Aggregate time measurements across successful runs
    # TODO: Ensure that NANs don't spoil the aggregation.
    problems_df = problems_df.join(runs_df[runs_df.exit_code == 0].groupby(['problem_path']).agg(
        {field_name: agg_functions for field_name in
         ['time_elapsed_process', 'time_elapsed_vampire', 'saturation_iterations']}))
    # Merge probe run results into `problems_df`
    if probe_runs_df is not None:
        problems_df = problems_df.join(
            probe_runs_df[['problem_path', 'predicates_count', 'functions_count', 'clauses_count']].drop_duplicates(
                'problem_path').set_index('problem_path'),
            rsuffix='probe')
    return problems_df
