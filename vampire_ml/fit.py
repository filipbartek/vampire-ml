#!/usr/bin/env python3.7

import argparse
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
from sklearn.model_selection import GridSearchCV

import vampyre
from utils import file_path_list
from vampire_ml.sklearn_extensions import EstimatorDict
from vampire_ml.sklearn_extensions import QuantileImputer
from vampire_ml.sklearn_extensions import StableShuffleSplit
from vampire_ml.train import IsolatedProblemToPreferencesTransformer
from vampire_ml.train import JointProblemToPreferencesTransformer
from vampire_ml.train import PreferenceToPrecedenceTransformer
from vampire_ml.train import ProblemToResultsTransformer
from vampire_ml.train import ScorerMeanScore
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


def call(namespace):
    # SWV567-1.014.p has clause depth of more than the default recursion limit of 1000,
    # making `json.load()` raise `RecursionError`.
    sys.setrecursionlimit(2000)

    problem_paths, problem_base_path = file_path_list.compose(namespace.problem_list, namespace.problem,
                                                              namespace.problem_base_path)

    vampire_options = dict()
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

    with vampyre.vampire.workspace_context(path=namespace.output, program=namespace.vampire,
                                           problem_dir=problem_base_path, include_dir=namespace.include,
                                           scratch_dir=namespace.scratch,
                                           never_load=never_load, never_run=never_run,
                                           result_is_ok_to_load=result_is_ok_to_load):
        problems = [vampyre.vampire.Problem(os.path.relpath(problem_path, problem_base_path),
                                            base_options=vampire_options,
                                            timeout=namespace.timeout) for problem_path in problem_paths]
        problem_to_results_transformer = ProblemToResultsTransformer(namespace.solve_runs,
                                                                     namespace.random_predicate_precedence,
                                                                     namespace.random_function_precedence,
                                                                     progress=True)
        # TODO: Expose the parameters properly.
        target_transformer = get_y_pipeline()
        precedence_transformer = IsolatedProblemToPreferencesTransformer(problem_to_results_transformer,
                                                                         target_transformer,
                                                                         LassoCV(copy_X=False))
        # TODO: Try MLPRegressor.
        preference_regressors = EstimatorDict(predicate=LassoCV(copy_X=False), function=LassoCV(copy_X=False))
        preference_learner = JointProblemToPreferencesTransformer(precedence_transformer, preference_regressors,
                                                                  batch_size=1000000, progress=True)
        precedence_estimator = sklearn.pipeline.Pipeline([
            ('problem_to_preference', preference_learner),
            ('preference_to_precedence', PreferenceToPrecedenceTransformer())
        ])

        scorers = {
            'success_rate': ScorerSuccessRate(progress=True),
            'saturation_iterations': ScorerMeanScore(problem_to_results_transformer, target_transformer, progress=True)
        }

        param_grid = [{
            'problem_to_preference__batch_size': [1000]
        }]
        cv = StableShuffleSplit(n_splits=1, train_size=10, test_size=10, random_state=0)
        gs = GridSearchCV(precedence_estimator, param_grid, scoring=scorers, cv=cv, refit=False, verbose=1000,
                          error_score='raise')
        gs.fit(problems)
        with pd.option_context('display.max_seq_items', None, 'display.max_columns', None, 'display.expand_frame_repr',
                               False):
            print(pd.DataFrame(gs.cv_results_))


def get_y_pipeline(failure_penalty_quantile=1, failure_penalty_factor=1, failure_penalty_divide_by_success_rate=True,
                   log_scale=True, normalize=True):
    y_pipeline_steps = list()
    if failure_penalty_quantile is not None:
        y_pipeline_steps.append(
            ('quantile', QuantileImputer(copy=False, quantile=failure_penalty_quantile, factor=failure_penalty_factor,
                                         divide_by_success_rate=failure_penalty_divide_by_success_rate)))
    if log_scale:
        y_pipeline_steps.append(('log', sklearn.preprocessing.FunctionTransformer(func=np.log)))
    if normalize:
        y_pipeline_steps.append(('normalize', sklearn.preprocessing.StandardScaler(copy=False)))
    if len(y_pipeline_steps) == 0:
        y_pipeline_steps.append(('passthrough', 'passthrough'))
    return sklearn.pipeline.Pipeline(y_pipeline_steps)


if __name__ == '__main__':
    np.seterr(all='raise', under='warn')

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    # TODO: Allow loading a trained model.
    namespace = parser.parse_args()

    np.random.seed(0)

    logging.getLogger('matplotlib').setLevel(logging.INFO)

    namespace.action(namespace)
