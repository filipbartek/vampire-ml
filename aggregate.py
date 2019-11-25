#!/usr/bin/env python3.7

import argparse
import os

import numpy as np
import pandas as pd

import file_path_list
import results
import vampire


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('batch', nargs='+', help='glob pattern of batch configuration files')
    parser.add_argument('--output', '-o', required=True, type=str, help='output directory')


def call(namespace):
    result_paths, _ = file_path_list.compose(glob_patterns=namespace.batch)
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
    dfs = [df.astype({col: np.object for col in df.select_dtypes(['category'])}) for df in dfs]
    if len(dfs) == 0:
        return None
    return pd.concat(dfs).astype({col: pd.CategoricalDtype() for col, field in vampire.Run.fields.items() if
                                  isinstance(field.dtype, pd.CategoricalDtype)})


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
