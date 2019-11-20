#!/usr/bin/env python3.7

import argparse
import os

import pandas as pd

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

    df_solve = pd.concat(
        (batch['solve'].assign(batch=batch['path']) for batch in batches if batch['solve'] is not None))
    results.save_df(df_solve, 'runs_solve', namespace.output)

    df_clausify = pd.concat(
        (batch['clausify'].assign(batch=batch['path']) for batch in batches if batch['clausify'] is not None))
    results.save_df(df_clausify, 'runs_clausify', namespace.output)

    results.save_problems(df_solve, df_clausify, namespace.output)


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
