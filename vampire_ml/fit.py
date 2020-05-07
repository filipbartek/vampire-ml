#!/usr/bin/env python3

import argparse
import itertools
import logging
import os
import sys
import warnings
from itertools import chain

import numpy as np
import pandas as pd
import sklearn
import sklearn.impute
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVR, SVR

import utils
import vampyre
from utils import ProgressBar
from utils import file_path_list
from utils import memory
from vampire_ml import results
from vampire_ml.precedence_generators import BestPrecedenceGenerator
from vampire_ml.precedence_generators import GreedyPrecedenceGenerator
from vampire_ml.precedence_generators import RandomPrecedenceGenerator
from vampire_ml.results import save_df
from vampire_ml.scorers import ScorerExplainer
from vampire_ml.scorers import ScorerOrdering
from vampire_ml.scorers import ScorerPercentile
from vampire_ml.scorers import ScorerPrediction
from vampire_ml.scorers import ScorerSaturationIterations
from vampire_ml.scorers import ScorerSuccess
from vampire_ml.scorers import ScorerSuccessRelative
from vampire_ml.sklearn_extensions import FrozenLinearModel
from vampire_ml.sklearn_extensions import QuantileImputer
from vampire_ml.sklearn_extensions import StableShuffleSplit
from vampire_ml.sklearn_extensions import StableStandardScaler
from vampire_ml.train import PreferenceMatrixPredictor
from vampire_ml.train import PreferenceMatrixTransformer
from vampire_ml.train import RunGenerator

cases_all = ['preference_estimation', 'pair_value_regressors', 'pair_value_svr', 'unweighted',
             'default_heuristic', 'random', 'best_encountered', 'default', 'heuristics']

cases_default = ['pair_value_regressors', 'default_heuristic', 'random', 'best_encountered', 'default', 'unweighted',
                 'heuristics']


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('problem', type=str, nargs='*', help='glob pattern of a problem path')
    parser.add_argument('--problem-list', action='append', default=[], help='input file with a list of problem paths')
    # Naming convention: `sbatch --output`
    parser.add_argument('--output', '-o', required=True, type=str, help='main output directory')
    parser.add_argument('--train-solve-runs', type=int, default=1000,
                        help='Number of Vampire executions per problem. '
                             'Each of the executions uses random predicate and function precedence.')
    parser.add_argument('--test-solve-runs', type=int, default=10)
    parser.add_argument('--vampire', type=str, default='vampire', help='Vampire command')
    # https://stackoverflow.com/a/20493276/4054250
    parser.add_argument('--vampire-options', type=yaml.safe_load, action='append', nargs=1, default=list(),
                        help='Options passed to Vampire. '
                             'Run `vampire --show_options on --show_experimental_options on` to print the options '
                             'supported by Vampire. '
                             'Format: YAML dictionary. '
                             'For example, "{time_limit: 10}" translates into '
                             '"--time_limit 10".'
                             'Recommended options: time_limit.')
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
    parser.add_argument('--precompute-exhaustive', action='store_true')
    parser.add_argument('--precompute-only', action='store_true')
    parser.add_argument('--precompute', choices=['all', 'splits'])
    parser.add_argument('--problems-train', action='append')
    parser.add_argument('--learn-max-symbols', type=int, default=200)
    parser.add_argument('--predict-max-symbols', type=int, default=1024)
    parser.add_argument('--progress', type=int, default=1)
    parser.add_argument('--progress-mininterval', type=float, default=1)
    parser.add_argument('--progress-postfix', type=int, default=1)
    parser.add_argument('--progress-postfix-refresh', type=int, default=0)
    parser.add_argument('--cases', nargs='+', choices=cases_all)
    parser.add_argument('--problems-dataframe', action='store_true')


def split_size(s):
    try:
        return int(s)
    except ValueError:
        pass
    return float(s)


def instantiate_problems(problem_paths, problem_base_path, vampire_options, timeout, ind=None, modulus=None):
    if ind is not None:
        if modulus is not None:
            ind = ind % modulus
            problem_paths = itertools.islice(problem_paths, ind, None, modulus)
        else:
            problem_paths = itertools.islice(problem_paths, ind, ind + 1)
    for problem_path in problem_paths:
        yield vampyre.vampire.Problem(os.path.relpath(problem_path, problem_base_path), vampire_options=vampire_options,
                                      timeout=timeout)


def decorate_param_grid(param_grid, prefix):
    res = list()
    for d in param_grid:
        d2 = dict()
        for k, v in d.items():
            d2[prefix + k] = v
        res.append(d2)
    return res


def augment_param_grid(param_grid, new_param):
    for d in param_grid:
        d.update(new_param)
    return param_grid


def generate_dataframes(run_generator_test, problem):
    clausify_df = problem.get_clausify_execution().get_dataframe(
        field_names_obligatory=vampyre.vampire.Execution.field_names_clausify)
    solve_dfs = None
    try:
        executions = run_generator_test.get_executions(problem, progress_bar=False)
        solve_dfs = [
            execution.get_dataframe(field_names_obligatory=vampyre.vampire.Execution.field_names_solve) for
            execution in executions]
    except RuntimeError:
        logging.debug(f'Failed to generate runs on problem {problem}.', exc_info=True)
    return clausify_df, solve_dfs


def frozen_model(symbol_type, values):
    column_names = vampyre.vampire.Problem.get_symbol_pair_embedding_column_names(symbol_type)
    coef = np.zeros(len(column_names))
    for column, value in values.items():
        coef[np.where(column_names == column)[0][0]] = value
    return FrozenLinearModel(coef)


def call(namespace):
    # SWV567-1.014.p has clause depth of more than the default recursion limit of 1000,
    # making `json.load()` raise `RecursionError`.
    sys.setrecursionlimit(2000)

    if namespace.clear_cache_joblib:
        memory.recompute = True

    utils.progress_bar.enabled = namespace.progress
    utils.progress_bar.mininterval = namespace.progress_mininterval
    utils.progress_bar.postfix_enabled = namespace.progress_postfix
    utils.progress_bar.postfix_refresh = namespace.progress_postfix_refresh

    problem_base_path = None
    try:
        problem_base_path = os.environ['TPTP_PROBLEMS']
        logging.info(f'Problem base path set to $TPTP_PROBLEMS: {problem_base_path}')
    except KeyError:
        try:
            problem_base_path = os.path.join(os.environ['TPTP'], 'Problems')
            logging.info(f'Problem base path set to $TPTP/Problems: {problem_base_path}')
        except KeyError:
            warnings.warn('Set $TPTP_PROBLEMS to the problem base path.')

    include_path = None
    try:
        include_path = os.environ['TPTP']
        logging.info(f'Include path set to $TPTP: {include_path}')
    except KeyError:
        warnings.warn('Set $TPTP to the path prefix for the include TPTP directive.')

    problem_paths, problem_base_path = file_path_list.compose(namespace.problem_list, namespace.problem,
                                                              problem_base_path)
    problem_paths_train, _ = file_path_list.compose(namespace.problems_train, base_path=problem_base_path)
    if len(problem_paths_train) == 0:
        logging.info('Falling back: training on all problems.')
        problem_paths_train = problem_paths

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

    symbol_types = []
    if namespace.random_predicate_precedence:
        symbol_types.append('predicate')
    if namespace.random_function_precedence:
        symbol_types.append('function')

    with vampyre.vampire.workspace_context(program=namespace.vampire,
                                           problem_dir=problem_base_path, include_dir=include_path,
                                           never_load=never_load, never_run=never_run,
                                           result_is_ok_to_load=result_is_ok_to_load):
        problems = np.asarray(
            list(instantiate_problems(problem_paths, problem_base_path, vampire_options, namespace.timeout)))
        run_generator_train = RunGenerator(namespace.train_solve_runs,
                                           namespace.random_predicate_precedence,
                                           namespace.random_function_precedence)
        run_generator_test = None
        if namespace.test_solve_runs > 0:
            run_generator_test = RunGenerator(namespace.test_solve_runs,
                                              namespace.random_predicate_precedence,
                                              namespace.random_function_precedence)
        if namespace.problems_dataframe and run_generator_test is not None:
            dfs = Parallel()(delayed(generate_dataframes)(run_generator_test, problem) for problem in
                             ProgressBar(problems, desc='Generating problems dataframe', unit='problem'))
            clausify_dfs, solve_df_lists = zip(*dfs)
            solve_dfs = list(itertools.chain(*solve_df_lists))
            solve_runs_df = vampyre.vampire.Execution.concat_dfs(solve_dfs)
            clausify_runs_df = vampyre.vampire.Execution.concat_dfs(clausify_dfs)
            results.save_all(solve_runs_df, clausify_runs_df, namespace.output)
        if namespace.cases is not None:
            cases = set(namespace.cases)
        else:
            cases = set(cases_default)
        score_scaler_steps = {
            'imputer': QuantileImputer(copy=False, quantile=1),
            'log': FunctionTransformer(func=np.log),
            'standardizer': StableStandardScaler(copy=False)
        }
        score_scaler_continuous = sklearn.pipeline.Pipeline(list(score_scaler_steps.items()))
        score_scaler_binary = FunctionTransformer(func=np.isnan)
        problem_preference_matrix_transformer = PreferenceMatrixTransformer(run_generator_train,
                                                                            score_scaler_continuous,
                                                                            LassoCV(copy_X=False),
                                                                            max_symbols=namespace.learn_max_symbols)
        problem_preference_matrix_transformer_param_grid = list()
        if 'preference_estimation' in cases:
            problem_preference_matrix_transformer_param_grid.extend([
                {
                    'score_scaler': [score_scaler_continuous],
                    'score_predictor': [LassoCV(copy_X=False)],
                    'score_scaler__standardizer': [score_scaler_steps['standardizer'], 'passthrough'],
                    'score_scaler__log': [score_scaler_steps['log'], 'passthrough'],
                    'score_scaler__imputer__divide_by_success_rate': [False, True],
                    'score_scaler__imputer__factor': [1, 2, 10]
                },
                {
                    'score_scaler': [score_scaler_continuous],
                    'score_predictor': [LassoCV(copy_X=False)],
                    'score_scaler__imputer': ['passthrough'],
                    'score_scaler__log': [score_scaler_steps['log'], 'passthrough'],
                    'score_scaler__standardizer': [score_scaler_steps['standardizer'], 'passthrough']
                },
                {
                    'scorer_scaler': [score_scaler_binary],
                    'score_predictor': [LogisticRegression(), RidgeClassifier(), LogisticRegressionCV(),
                                        RidgeClassifierCV()]
                }
            ])
        if namespace.precompute_exhaustive:
            preference_predictor = problem_preference_matrix_transformer
            preference_predictor_param_grid = problem_preference_matrix_transformer_param_grid
            cv = StableShuffleSplit(n_splits=1, train_size=0, test_size=1.0, random_state=0)
        else:
            reg_linear = LinearRegression(copy_X=False)
            reg_lasso = LassoCV(copy_X=False)
            reg_svr_linear = LinearSVR(loss='squared_epsilon_insensitive', dual=False, random_state=0)
            reg_svr = SVR()
            reg_gbr = GradientBoostingRegressor(random_state=0)
            preference_predictor = PreferenceMatrixPredictor(problem_preference_matrix_transformer,
                                                             reg_lasso,
                                                             batch_size=1000000,
                                                             max_symbols=namespace.predict_max_symbols,
                                                             random_state=0)
            preference_predictor_param_grid = decorate_param_grid(problem_preference_matrix_transformer_param_grid,
                                                                  'problem_matrix__')
            if 'pair_value_regressors' in cases:
                preference_predictor_param_grid.extend(
                    [{'batch_size': [1000], 'pair_value': [reg_linear, reg_lasso, reg_svr_linear, reg_svr]},
                     {'pair_value': [reg_linear, reg_lasso, reg_svr_linear, reg_gbr]}])
            if 'heuristics' in cases:
                if len(symbol_types) == 1:
                    symbol_type = symbol_types[0]
                    preference_predictor_param_grid.extend([{'batch_size': [0], 'pair_value': [
                        # Ascending order by usage count; mimics `vampire --symbol_precedence reverse_frequency`
                        frozen_model(symbol_type, {'l.usageCnt': 1, 'r.usageCnt': -1}),
                        # Descending order by usage count; mimics `vampire --symbol_precedence frequency`
                        frozen_model(symbol_type, {'l.usageCnt': -1, 'r.usageCnt': 1}),
                        # Descending order by arity
                        frozen_model(symbol_type, {'l.arity': -1, 'r.arity': 1})]}])
            if 'unweighted' in cases:
                preference_predictor_param_grid.extend([{'weighted': [False]}])
            if 'pair_value_svr' in cases:
                preference_predictor_param_grid.extend([
                    {'pair_value': [reg_svr_linear], 'pair_value__C': [0.1, 0.5, 1.0, 2.0]},
                    {'batch_size': [1000], 'pair_value': [reg_svr], 'pair_value__C': [0.1, 0.5, 1.0, 2.0]}
                ])
            cv = StableShuffleSplit(n_splits=namespace.n_splits, train_size=namespace.train_size,
                                    test_size=namespace.test_size, random_state=0)
        precedence_estimator = sklearn.pipeline.Pipeline([
            ('preference', preference_predictor),
            ('precedence', GreedyPrecedenceGenerator())
        ])
        precedence_estimator = sklearn.pipeline.Pipeline([
            ('precedence', precedence_estimator)
        ])
        param_grid = list()
        precedence_default_heuristic = None
        if 'default_heuristic' in cases:
            precedence_default_heuristic = FunctionTransformer(func=transform_problems_to_empty_dicts)
            param_grid.extend([{'precedence': [precedence_default_heuristic]}])
        if 'random' in cases:
            param_grid.extend([{'precedence': [
                RandomPrecedenceGenerator(random_predicates=namespace.random_predicate_precedence,
                                          random_functions=namespace.random_function_precedence)]}])
        if 'best_encountered' in cases:
            if run_generator_test is not None:
                param_grid.extend([{'precedence': [BestPrecedenceGenerator(run_generator_test)]}])
            else:
                warnings.warn('Increase number of test runs to allow case best_encountered.')
        if 'default' in cases:
            param_grid.extend([{}])
        param_grid.extend(decorate_param_grid(preference_predictor_param_grid, 'precedence__preference__'))
        scorers = {
            'success.rate': ScorerSuccess(aggregate=np.mean),
            'success.count': ScorerSuccess(aggregate=np.sum),
            'prediction.rate': ScorerPrediction(aggregate=np.mean),
            'prediction.count': ScorerPrediction(aggregate=np.sum),
            'explainer': ScorerExplainer()
        }
        if precedence_default_heuristic is not None:
            scorers.update({
                'success_relative.better': ScorerSuccessRelative(baseline_estimator=precedence_default_heuristic,
                                                                 mode='better'),
                'success_relative.worse': ScorerSuccessRelative(baseline_estimator=precedence_default_heuristic,
                                                                mode='worse')
            })
        if run_generator_test is not None:
            scorers.update({
                'iterations': ScorerSaturationIterations(run_generator_test,
                                                         sklearn.base.clone(score_scaler_continuous)),
                'percentile.strict': ScorerPercentile(run_generator_test, kind='strict'),
                'percentile.rank': ScorerPercentile(run_generator_test, kind='rank'),
                'percentile.weak': ScorerPercentile(run_generator_test, kind='weak')
            })
            for symbol_type in symbol_types:
                for measure in ['saturation_iterations', 'success']:
                    for comparison in ['strict', 'weak']:
                        scorers[f'ordering.{symbol_type}.{measure}.{comparison}'] = ScorerOrdering(run_generator_test,
                                                                                                   symbol_type,
                                                                                                   measure=measure,
                                                                                                   comparison=comparison)
        groups = None
        if len(problem_paths_train) > 0:
            problem_paths_train_set = set(problem_paths_train)
            groups = np.fromiter((p in problem_paths_train_set for p in problem_paths), dtype=np.bool,
                                 count=len(problem_paths))
        logging.info('Number of splits: %s', namespace.n_splits)
        logging.info('Train problems per split: %s / %s', cv.train_samples(np.count_nonzero(groups)),
                     np.count_nonzero(groups))
        logging.info('Test problems per split: %s / %s', cv.test_samples(len(problems)), len(problems))
        if namespace.precompute is not None:
            logging.info('Precomputing preference matrices with default preference matrix estimation.')
            if namespace.precompute == 'all':
                if groups is not None:
                    problems_train = problems[groups]
                else:
                    problems_train = problems
                logging.info('Precomputing %s train problems.', len(problems_train))
                problem_preference_matrix_transformer.transform(problems_train)
                if run_generator_test is not None:
                    logging.info('Precomputing %s test problems.', len(problems))
                    run_generator_test.transform(problems)
            if namespace.precompute == 'splits':
                # Note: Calling `split` preserves `random_state`.
                for train, test in cv.split(problems, groups=groups):
                    problem_preference_matrix_transformer.transform(problems[train])
                    if run_generator_test is not None:
                        run_generator_test.transform(problems[test])
        gs = GridSearchCV(precedence_estimator, param_grid, scoring=scorers, cv=cv, refit=False, verbose=5,
                          error_score='raise', return_train_score=True)
        if namespace.precompute_exhaustive:
            # Precompute data for train set
            problems_train = problems[groups]
            fit_gs(gs, problems_train, scorers, output=namespace.output, name='precompute_train')

            if run_generator_test is not None:
                # Precompute data for test set
                problem_preference_matrix_transformer.run_generator = run_generator_test
                fit_gs(gs, problems, scorers, output=namespace.output, name='precompute_test')
            return
        if namespace.precompute_only:
            return
        fit_gs(gs, problems, scorers, groups=groups, output=namespace.output, name='fit_cv_results')
        df = scorers['explainer'].get_dataframe()
        save_df(df, 'feature_weights', output_dir=namespace.output, index=True)


def fit_gs(gs, problems, scorers, groups=None, output=None, name=None):
    if len(problems) == 0:
        logging.info(f'{name}: Skipping fitting on an empty set of problems.')
        return
    gs.fit(problems, groups=groups)
    df = pd.DataFrame(gs.cv_results_)
    if name is not None:
        save_df(df, name, output_dir=output, index=False)
    with pd.option_context('display.max_seq_items', None, 'display.max_columns', None,
                           'display.expand_frame_repr', False):
        columns = ['params'] + [f'mean_test_{key}' for key in scorers.keys()]
        print(df[columns])


def transform_problems_to_empty_dicts(problems):
    return itertools.repeat(dict(), len(problems))


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
