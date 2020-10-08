#!/usr/bin/env python3

import logging
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.preprocessing


def save_all(df_solve, df_clausify, output, df_custom=None):
    if df_solve.saturation_iterations.isnull().sum() < len(df_solve):
        scaler = sklearn.preprocessing.StandardScaler()
        df_solve['saturation_iterations_normalized'] = scaler.fit_transform(df_solve[['saturation_iterations']].astype(np.float))
        if df_custom is not None:
            df_custom['saturation_iterations_normalized'] = scaler.transform(df_custom[['saturation_iterations']].astype(np.float))

    save_df(df_solve, 'runs_solve', output, index=False)
    save_df(df_clausify, 'runs_clausify', output, index=False)
    if df_solve is not None:
        save_terminations(df_solve, os.path.join(output, 'runs_solve_terminations.txt'))
        save_problems(df_solve, df_clausify, output, custom_runs_df=df_custom)
    if df_custom is not None:
        save_df(df_custom, 'runs_custom', output, index=False)
        save_terminations(df_custom, os.path.join(output, 'runs_custom_terminations.txt'))


def save_df(df, base_name, output_dir=None, index=True):
    if df is None:
        return
    assert base_name is not None
    path_common = base_name
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path_common = os.path.join(output_dir, path_common)
    if len(os.path.dirname(path_common)) > 0:
        os.makedirs(os.path.dirname(path_common), exist_ok=True)
    df.to_pickle(f'{path_common}.pkl')
    with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
        df.to_csv(f'{path_common}.csv', index=index, header=df.columns.values)
    logging.info(f'DataFrame of length {len(df.index)} saved: {path_common}.{{pkl,csv}}')


def save_terminations(solve_runs_df, output_batch):
    solve_runs_df = fill_category_na(solve_runs_df)
    termination_fieldnames = [col for col in ['name', 'status', 'exit_code', 'termination_reason', 'termination_phase']
                              if col in solve_runs_df.columns]

    # Distribution of run terminations
    terminations = solve_runs_df.groupby(termination_fieldnames).size()
    print('Distribution of run terminations:', terminations, sep='\n')
    os.makedirs(os.path.dirname(output_batch), exist_ok=True)
    with pd.option_context('display.max_rows', None):
        print(terminations, file=open(output_batch, 'w'))


def save_problems(solve_runs_df, clausify_runs_df, output_batch, problem_paths=None, problem_base_path=None,
                  custom_runs_df=None):
    problems_df = generate_problems_df(solve_runs_df, clausify_runs_df, problem_paths, problem_base_path,
                                       custom_runs_df)
    save_df(problems_df, 'problems', output_batch)
    return problems_df


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


def generate_problems_df(runs_df, probe_runs_df=None, problem_paths=None, problem_base_path=None, custom_runs_df=None):
    if problem_paths is None:
        problem_paths = runs_df.problem_path
    elif problem_base_path is not None:
        problem_paths = [os.path.relpath(problem_path, problem_base_path) for problem_path in problem_paths]
    problem_paths = pd.Index(problem_paths).drop_duplicates()
    problems_df = pd.DataFrame(index=problem_paths)
    problems_df.index.name = 'problem_path'
    # Merge probe run results into `problems_df`
    if probe_runs_df is not None:
        problems_df = problems_df.join(
            probe_runs_df[['problem_path', 'predicates_count', 'functions_count', 'clauses_count']].drop_duplicates(
                'problem_path').set_index('problem_path'),
            rsuffix='probe')
    # Random solve run stats
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
        if (a == 0).all():
            # We need to handle this special case explicitly because `scipy.stats.variation` raises an exception on it.
            return 0
        res = scipy.stats.variation(a.astype(np.float), nan_policy='omit')
        if isinstance(res, np.ma.core.MaskedConstant):
            # The input array contains all nans.
            return np.nan
        return res

    agg_functions = [np.mean, np.std, variation, np.min, np.max]
    # Aggregate time measurements across successful runs
    problems_df = problems_df.join(runs_df[runs_df.exit_code == 0].groupby(['problem_path']).agg(
        {field_name: agg_functions for field_name in
         ['time_elapsed_process', 'time_elapsed_vampire', 'saturation_iterations']}))
    # Count unique numbers of saturation iterations across successful runs
    problems_df = problems_df.join(
        runs_df[runs_df.exit_code == 0].groupby(['problem_path']).agg({'saturation_iterations': ['nunique']}).astype(
            pd.UInt64Dtype()))
    # Aggregate memory across all runs
    problems_df = problems_df.join(runs_df.groupby(['problem_path']).agg({'memory_used': agg_functions}))
    if custom_runs_df is not None:
        for name, value in custom_runs_df.groupby(['name']):
            value = value.set_index('problem_path')
            value = value[['exit_code', 'saturation_iterations']]
            # https://stackoverflow.com/a/40225796/4054250
            value.columns = pd.MultiIndex.from_product([[name], value.columns])
            problems_df = problems_df.join(value, rsuffix=name)
    problems_df.sort_index(inplace=True)
    return problems_df
