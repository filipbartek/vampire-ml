#!/usr/bin/env python3.7

import argparse
import itertools
import logging
import os
import sys
from itertools import chain

import numpy as np
import pandas as pd
import sklearn
import sklearn.impute
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import yaml
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVR, SVR
from tqdm import tqdm

import vampyre
from utils import file_path_list
from utils import memory
from vampire_ml.results import save_df
from vampire_ml.sklearn_extensions import MeanRegression
from vampire_ml.sklearn_extensions import QuantileImputer
from vampire_ml.sklearn_extensions import StableShuffleSplit
from vampire_ml.sklearn_extensions import StableStandardScaler
from vampire_ml.train import BestPrecedenceGenerator
from vampire_ml.train import GreedyPrecedenceGenerator
from vampire_ml.train import PreferenceMatrixPredictor
from vampire_ml.train import PreferenceMatrixTransformer
from vampire_ml.train import RunGenerator
from vampire_ml.train import ScorerPercentile
from vampire_ml.train import ScorerSaturationIterations
from vampire_ml.train import ScorerSuccessRate


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('problem', type=str, nargs='*', help='glob pattern of a problem path')
    parser.add_argument('--problem-list', action='append', default=[], help='input file with a list of problem paths')
    parser.add_argument('--problem-base-path', type=str, help='the problem paths are relative to the base path')
    parser.add_argument('--include', help='path prefix for the include TPTP directive')
    # Naming convention: `sbatch --output`
    parser.add_argument('--output', '-o', required=True, type=str, help='main output directory')
    parser.add_argument('--scratch', help='temporary output directory')
    parser.add_argument('--solve-runs', type=int, default=1000,
                        help='Number of Vampire executions per problem. '
                             'Each of the executions uses random predicate and function precedence.')
    parser.add_argument('--vampire', type=str, default='vampire', help='Vampire command')
    # https://stackoverflow.com/a/20493276/4054250
    parser.add_argument('--vampire-options', type=yaml.safe_load, action='append', nargs='+',
                        help='Options passed to Vampire. '
                             'Run `vampire --show_options on --show_experimental_options on` to print the options '
                             'supported by Vampire. '
                             'Format: YAML dictionary. '
                             'For example, "{time_limit: 10}" translates into '
                             '"--include $TPTP --time_limit 10".'
                             'Recommended options: include, time_limit.')
    parser.add_argument('--timeout', type=float, default=20,
                        help='Time in seconds after which each Vampire call is terminated.')
    parser.add_argument('--random-predicate-precedence', action='store_true')
    parser.add_argument('--random-function-precedence', action='store_true')
    parser.add_argument('--run-policy', choices=['none', 'interrupted', 'failed', 'all'], default='interrupted',
                        help='Which Vampire run configurations should be executed?')
    parser.add_argument('--clear-cache-joblib', action='store_true')
    parser.add_argument('--n-splits', type=int, default=1)
    parser.add_argument('--train-size', type=split_size)
    parser.add_argument('--test-size', type=split_size)
    parser.add_argument('--precompute', action='store_true')
    parser.add_argument('--problems-train', action='append')


def split_size(s):
    try:
        return int(s)
    except ValueError:
        pass
    return float(s)


def call(namespace):
    # SWV567-1.014.p has clause depth of more than the default recursion limit of 1000,
    # making `json.load()` raise `RecursionError`.
    sys.setrecursionlimit(2000)

    if namespace.clear_cache_joblib:
        memory.clear()

    problem_paths, problem_base_path = file_path_list.compose(namespace.problem_list, namespace.problem,
                                                              namespace.problem_base_path)
    problem_paths_train, _ = file_path_list.compose(namespace.problems_train, base_path=problem_base_path)

    # Default Vampire options:
    vampire_options = {
        'encode': 'on',
        'statistics': 'full',
        'time_statistics': 'on',
        'proof': 'off',
        'literal_comparison_mode': 'predicate',
        'symbol_precedence': 'frequency',
        'saturation_algorithm': 'discount',
        'age_weight_ratio': '10',
        'avatar': 'off'
    }
    for value in chain(*namespace.vampire_options):
        vampire_options.update(value)

    # TODO: Refine the conditions under which we skip execution. Shall we retry runs that terminated with code -11?
    never_load = False
    never_run = False
    result_is_ok_to_load = None
    if namespace.run_policy == 'none':
        never_run = True
    if namespace.run_policy == 'interrupted':
        result_is_ok_to_load = lambda result: result.status == 'completed' and result.exit_code in [0, 1]
    if namespace.run_policy == 'failed':
        result_is_ok_to_load = lambda result: result.status == 'completed' and result.exit_code == 0
    if namespace.run_policy == 'all':
        never_load = True

    with vampyre.vampire.workspace_context(program=namespace.vampire,
                                           problem_dir=problem_base_path, include_dir=namespace.include,
                                           scratch_dir=namespace.scratch,
                                           never_load=never_load, never_run=never_run,
                                           result_is_ok_to_load=result_is_ok_to_load):
        problems = [vampyre.vampire.Problem(os.path.relpath(problem_path, problem_base_path),
                                            vampire_options=vampire_options,
                                            timeout=namespace.timeout) for problem_path in problem_paths]
        run_generator = RunGenerator(namespace.solve_runs,
                                     namespace.random_predicate_precedence,
                                     namespace.random_function_precedence)
        score_scaler_steps = {
            'imputer': QuantileImputer(copy=False, quantile=1),
            'log': FunctionTransformer(func=np.log),
            'standardizer': StableStandardScaler(copy=False)
        }
        score_scaler = sklearn.pipeline.Pipeline(list(score_scaler_steps.items()))
        problem_preference_matrix_transformer = PreferenceMatrixTransformer(run_generator,
                                                                            sklearn.base.clone(score_scaler),
                                                                            LassoCV(copy_X=False))
        if namespace.precompute:
            logging.info('Omitting cross-problem training.')
            isolated_param_grid = [
                {},
                {'score_scaler__imputer__divide_by_success_rate': [False]},
                {'score_scaler__imputer__factor': [1, 2, 10]},
                {'score_scaler__log': ['passthrough']},
                {'score_scaler__standardizer': ['passthrough']}
            ]
            for params in tqdm(ParameterGrid(isolated_param_grid), desc='Precomputing', unit='combination'):
                estimator = sklearn.base.clone(problem_preference_matrix_transformer)
                estimator.set_params(**params)
                list(estimator.transform(problems))
        else:
            reg_linear = LinearRegression(copy_X=False)
            reg_lasso = LassoCV(copy_X=False)
            reg_svr_linear = LinearSVR(loss='squared_epsilon_insensitive', dual=False, random_state=0)
            reg_svr = SVR()
            preference_predictor = PreferenceMatrixPredictor(problem_preference_matrix_transformer,
                                                             reg_lasso,
                                                             batch_size=1000000)
            precedence_estimator = sklearn.pipeline.Pipeline([
                ('preference', preference_predictor),
                ('precedence', GreedyPrecedenceGenerator())
            ])
            precedence_estimator = sklearn.pipeline.Pipeline([
                ('precedence', precedence_estimator)
            ])
            # TODO: Use a run generator with fewer runs per problem.
            scoring_run_generator = run_generator
            scorers = {
                'success_rate': ScorerSuccessRate(),
                'iterations': ScorerSaturationIterations(scoring_run_generator, sklearn.base.clone(score_scaler)),
                'percentile.strict': ScorerPercentile(scoring_run_generator, kind='strict'),
                'percentile.rank': ScorerPercentile(scoring_run_generator, kind='rank'),
                'percentile.weak': ScorerPercentile(scoring_run_generator, kind='weak')
            }
            cv = StableShuffleSplit(n_splits=namespace.n_splits, train_size=namespace.train_size,
                                    test_size=namespace.test_size, random_state=0)
            param_grid = [
                {},
                {
                    'precedence__preference__batch_size': [1000],
                    'precedence__preference__pair_value': [reg_linear, reg_lasso, reg_svr_linear, reg_svr]
                },
                {'precedence__preference__pair_value': [reg_linear, reg_lasso, reg_svr_linear]},
                {'precedence__preference__weighted': [False]},
                {'precedence__preference__problem_matrix__score_scaler__quantile__divide_by_success_rate': [False]},
                {'precedence__preference__problem_matrix__score_scaler__quantile__factor': [1, 2, 10]},
                {'precedence__preference__problem_matrix__score_scaler__log': ['passthrough']},
                {'precedence__preference__problem_matrix__score_scaler__normalize': ['passthrough']},
                {'precedence': [FunctionTransformer(func=transform_problems_to_none),
                                BestPrecedenceGenerator(scoring_run_generator)]},
                {
                    'precedence__preference__pair_value': [reg_svr_linear],
                    'precedence__preference__pair_value__C': [0.1, 0.5, 1.0, 2.0]
                },
                {
                    'precedence__preference__batch_size': [1000],
                    'precedence__preference__pair_value': [reg_svr],
                    'precedence__preference__pair_value__C': [0.1, 0.5, 1.0, 2.0]
                },
                {'precedence__preference__problem_matrix__score_predictor': [MeanRegression()]}
            ]
            gs = GridSearchCV(precedence_estimator, param_grid, scoring=scorers, cv=cv, refit=False, verbose=5,
                              error_score='raise')
            problem_paths_train = set(problem_paths_train)
            groups = None
            if len(problem_paths_train) > 0:
                groups = np.fromiter((p in problem_paths_train for p in problem_paths), dtype=np.bool,
                                     count=len(problem_paths))
            # TODO: Parallelize.
            gs.fit(problems, groups=groups)
            df = pd.DataFrame(gs.cv_results_)
            save_df(df, 'fit_cv_results', output_dir=namespace.output, index=False)
            with pd.option_context('display.max_seq_items', None, 'display.max_columns', None,
                                   'display.expand_frame_repr',
                                   False):
                columns = ['params'] + [f'mean_test_{key}' for key in scorers.keys()]
                print(df[columns])


def transform_problems_to_none(problems):
    return itertools.repeat(None, len(problems))


if __name__ == '__main__':
    np.seterr(all='raise', under='warn')

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    # TODO: Allow loading a trained model.
    namespace = parser.parse_args()

    np.random.seed(0)

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    namespace.action(namespace)
