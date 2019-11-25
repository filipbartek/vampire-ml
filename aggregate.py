#!/usr/bin/env python3.7

import argparse
import os

import pandas as pd
from pandas.api.types import union_categoricals

import file_path_list
import results


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('batch', nargs='+', help='glob pattern of batch configuration files')
    parser.add_argument('--output', '-o', required=True, type=str, help='output directory')


def call(namespace):
    batches = list()
    result_paths, _ = file_path_list.compose(glob_patterns=namespace.batch)
    for result_path in result_paths:
        result_dir = os.path.dirname(result_path)
        batches.append({'path': result_dir,
                        'solve': pd_read_pickle_robust(os.path.join(result_dir, 'runs_solve.pkl')),
                        'clausify': pd_read_pickle_robust(os.path.join(result_dir, 'runs_clausify.pkl'))})
    result_dirs = [os.path.dirname(result_path) for result_path in result_paths]
    df_solve = concat_pickles(result_dirs, 'runs_solve.pkl')
    df_clausify = concat_pickles(result_dirs, 'runs_clausify.pkl')
    results.save_all(df_solve, df_clausify, namespace.output)


def concat_pickles(result_dirs, pickle_base_name):
    dfs = list()
    for result_dir in result_dirs:
        try:
            dfs.append(pd.read_pickle(os.path.join(result_dir, pickle_base_name)).assign(batch=result_dir))
        except FileNotFoundError:
            pass
    return concat_dfs(dfs)


def concat_dfs(dfs):
    dfs = list(dfs)
    # https://stackoverflow.com/a/57809778/4054250
    for col in dfs[0].select_dtypes(['category']):
        uc = union_categoricals([df[col] for df in dfs])
        for df in dfs:
            df[col] = pd.Categorical(df[col], categories=uc.categories)
    return pd.concat(dfs)


def pd_read_pickle_robust(path):
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    namespace = parser.parse_args()
    call(namespace)
