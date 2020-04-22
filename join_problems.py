#!/usr/bin/env python3

import argparse
import itertools
import logging
import os

import numpy as np
import pandas as pd
import yaml

import vampire_ml.results


def arg_to_columns(arg):
    if arg is None:
        return None
    return list(itertools.chain(*arg))


def str_to_column(s):
    value = yaml.safe_load(s)
    if isinstance(value, list):
        value = tuple(value)
    return value


def filter_columns(df, filters):
    for column, op, expected in filters:
        assert op in ['<=', '>=', '==']
        column = str_to_column(column)
        actual = df[column]
        flags = None
        if op == '<=':
            flags = actual <= float(expected)
        elif op == '>=':
            flags = actual >= float(expected)
        elif op == '==':
            flags = actual == expected
        assert flags is not None
        logging.debug(f'{column} {op} {expected}: {flags.sum()}/{len(flags)}')
        df = df[flags]
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problems', nargs=2, action='append', metavar=('NAME', 'PATH'),
                        help='pairs of dataset names and dataframe pickle paths')
    parser.add_argument('--output', required=True, metavar='DIRECTORY', help='output directory')
    parser.add_argument('--columns_common', nargs='+', action='append', type=str_to_column, metavar='COLUMN')
    parser.add_argument('--columns_individual', nargs='+', action='append', type=str_to_column, metavar='COLUMN')
    parser.add_argument('--filters', nargs=3, action='append', metavar=('COLUMN', 'COMPARISON', 'VALUE'),
                        help='triplets "column_name {<=,>=,==} expected_value"')
    parser.add_argument('--sort_column', type=str_to_column, metavar='COLUMN')
    parser.add_argument('--sort_asc', action='store_true')
    parser.add_argument('--count', type=int, metavar='N', help='number of top records to take from each input dataset')
    namespace = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    columns_individual = arg_to_columns(namespace.columns_individual)
    columns_common = arg_to_columns(namespace.columns_common)

    problems_common = None
    problems = dict()
    for i, (name, file_path) in enumerate(namespace.problems):
        df = pd.read_pickle(file_path)
        assert df.index.name == 'problem_path'
        if columns_common is not None and i == 0:
            problems_common = df[columns_common]
        df = filter_columns(df, namespace.filters)
        if namespace.sort_column is not None:
            df.sort_values(namespace.sort_column, ascending=namespace.sort_asc, inplace=True)
        if len(columns_individual) > 0:
            df = df[columns_individual]
        if namespace.count is not None:
            assert namespace.count >= 0
            df = df[:namespace.count]
        problems[name] = df
    problems_aggregated = pd.concat(problems.values(), axis=1, keys=problems.keys(), sort=True)
    problems_aggregated.index.name = 'problem_path'
    if problems_common is not None:
        problems_aggregated = problems_common.join(problems_aggregated, how='right')
    vampire_ml.results.save_df(problems_aggregated, 'problems', output_dir=namespace.output)
    np.savetxt(os.path.join(namespace.output, 'problems.txt'), problems_aggregated.index.values, fmt='%s')
