#!/usr/bin/env python3

import argparse
import itertools
import logging
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from proving import config
from proving import file_path_list
from proving import tptp
from proving import utils
from proving.memory import memory
from proving.solver import Solver
from vampire_ml.results import save_df

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('problem', nargs='*')
    parser.add_argument('--problem-list', action='append', default=[])
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--file-properties', action='store_true')
    parser.add_argument('--content-properties', action='store_true')
    parser.add_argument('--clausify', action='store_true')
    parser.add_argument('--random-precedences', type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(threadName)s %(levelname)s - %(message)s')

    with joblib.parallel_backend('loky', n_jobs=args.jobs):
        problems, _ = list(file_path_list.compose(args.problem_list, args.problem))
        log.info('Problems available: %s', len(problems))
        solver = Solver(timeout=20)
        seeds = None
        if args.random_precedences is not None:
            seeds = range(args.random_precedences)
        df = get_problems_dataframe(problems, args.file_properties, args.content_properties, solver, args.clausify,
                                    seeds)
        log.info('Problem records collected: %s', len(df))

        print(df.dtypes)

        # https://stackoverflow.com/a/46672301/4054250
        # with pd.option_context('display.max_columns', None, 'display.float_format', '{:.2f}'.format):
        # https://stackoverflow.com/a/33375383/4054250
        with pd.option_context('display.max_columns', None):
            print(df.describe())

        if args.output is not None:
            save_df(df, os.path.join(args.output, 'problems'))

            # Interesting numeric: size, rating, predicates, functors, variables
            # Interesting categorical: domain, form, source, status, spc

            cols_num = ('rating', 'size', 'predicates', 'functors', 'variables', 'formulae', 'clauses', 'atoms')
            cols_num = [col for col in cols_num if col in df]
            cols_log_scale = {'size', 'predicates', 'functors', 'variables', 'formulae', 'clauses', 'atoms'} & set(
                cols_num)
            cols_cat = ('domain', 'form', 'status')
            cols_cat = [col for col in cols_cat if col in df]

            plt.figure()
            sns.pairplot(data=df, vars=cols_num, hue='form', markers='.')
            plt.savefig(os.path.join(args.output, f'pair.png'), bbox_inches='tight')
            plt.close()

            os.makedirs(os.path.join(args.output, 'dist'), exist_ok=True)
            for col in cols_num:
                plt.figure()
                ax = sns.distplot(df[col].astype(float), kde=False)
                if col in cols_log_scale:
                    ax.set(yscale='log')
                plt.savefig(os.path.join(args.output, 'dist', f'{col}.png'), bbox_inches='tight')
                plt.close()

            os.makedirs(os.path.join(args.output, 'cat'), exist_ok=True)
            for col_cat, col_num in itertools.product(cols_cat, cols_num):
                plt.figure()
                g = sns.catplot(x=col_num, y=col_cat, data=df.astype({col_num: float}), kind='box')
                if col_num in cols_log_scale:
                    g.set(xscale='log')
                plt.savefig(os.path.join(args.output, 'cat', f'{col_cat}_{col_num}.png'),
                            bbox_inches='tight')
                plt.close()


path_property_getters = {'isfile': os.path.isfile,
                         'size': os.path.getsize}


@memory.cache
def get_problems_dataframe(problems, file_properties, content_properties, solver=None, clausify=False, seeds=None):
    records = Parallel(verbose=1000)(
        delayed(process_problem)(problem, file_properties, content_properties, solver, clausify, seeds) for problem in
        problems)
    dtypes = utils.join_dicts((tptp.property_types, {'size': pd.UInt32Dtype()}))
    keys = set(itertools.chain.from_iterable(record.keys() for record in records))
    dtypes = {k: v for k, v in dtypes.items() if k in keys}
    return utils.dataframe_from_records(records, 'problem', dtypes=dtypes)


def process_problem(problem, file_properties=True, content_properties=True, solver=None, clausify=False, seeds=None):
    record = {'problem': problem}
    if file_properties:
        path = config.full_problem_path(problem)
        record['path'] = path
        record['size'] = os.path.getsize(path)
    record.update(tptp.problem_properties(problem, header_properties=content_properties))
    if solver is not None:
        if clausify:
            record.update(problem_clausify_properties(problem, solver))
        if seeds is not None:
            random_precedence_types = ('predicate', 'function')
            record.update(problem_solving_random_precedence_properties(problem, solver, random_precedence_types, seeds))
    return record


@memory.cache
def problem_clausify_properties(problem, solver, get_symbols=True, get_clauses=True):
    record = {}
    clausify_result = solver.clausify(problem, get_symbols=get_symbols, get_clauses=get_clauses)
    record[f'clausify_returncode'] = clausify_result.returncode
    if clausify_result.returncode == 0:
        if get_symbols:
            for symbol_type in ('predicate', 'function'):
                record[f'clausify_{symbol_type}_count'] = len(clausify_result.symbols_of_type(symbol_type))
        if get_clauses:
            record['clausify_clauses'] = len(clausify_result.clauses)
    return record


@memory.cache
def problem_solving_random_precedence_properties(problem, solver, random_precedence_types, seeds):
    record = {}
    try:
        precedence_costs, symbols_by_type = solver.costs_of_random_precedences(problem, random_precedence_types,
                                                                               seeds)
        if len(precedence_costs) == 0:
            return record
        costs = np.asarray(tuple(zip(*precedence_costs))[1], dtype=np.float)
        record['random_precedence_costs_nan_count'] = np.count_nonzero(np.isnan(costs))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            record['random_precedence_costs_mean'] = np.nanmean(costs)
            record['random_precedence_costs_std'] = np.nanstd(costs)
            record['random_precedence_costs_min'] = np.nanmin(costs)
            record['random_precedence_costs_max'] = np.nanmax(costs)
    except RuntimeError:
        pass
    return record


if __name__ == '__main__':
    main()
