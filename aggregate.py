#!/usr/bin/env python3.7

import argparse
import logging
import os

import pandas as pd

import vampyre
from utils import file_path_list
from vampire_ml import results


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('batch', nargs='+',
                        help='glob pattern of batch configuration files, for example "workspace/batches/*/batch.json"')
    parser.add_argument('--output', '-o', required=True, type=str, help='output directory')


def call(namespace):
    result_paths, _ = file_path_list.compose(glob_patterns=namespace.batch)
    result_dirs = [os.path.dirname(result_path) for result_path in result_paths]
    df_solve = concat_pickles(result_dirs, 'runs_solve.pkl')
    df_clausify = concat_pickles(result_dirs, 'runs_clausify.pkl')
    df_custom = concat_pickles(result_dirs, 'runs_custom.pkl')
    # TODO: Drop rows with duplicate index. Ensure that they have the same content.
    results.save_all(df_solve, df_clausify, namespace.output, df_custom)


def concat_pickles(result_dirs, pickle_base_name):
    dfs = list()
    for result_dir in result_dirs:
        try:
            dfs.append(pd.read_pickle(os.path.join(result_dir, pickle_base_name)).assign(batch=result_dir))
        except FileNotFoundError:
            pass
    return vampyre.vampire.Execution.concat_dfs(dfs)


def pd_read_pickle_robust(path):
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    namespace = parser.parse_args()
    call(namespace)
