#!/usr/bin/env python3.7

import itertools
import json
import logging
import os
import sys
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import sklearn
import sklearn.impute
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import yaml

import vampyre
from utils import file_path_list
from utils import makedirs_open
from . import precedence
from . import results
from .sklearn_extensions import MeanRegression, QuantileImputer, Flattener


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('problem', type=str, nargs='*', help='glob pattern of a problem path')
    parser.add_argument('--problem-list', action='append', default=[], help='input file with a list of problem paths')
    parser.add_argument('--problem-base-path', type=str, help='the problem paths are relative to the base path')
    parser.add_argument('--include', help='path prefix for the include TPTP directive')
    # Naming convention: `sbatch --output`
    parser.add_argument('--output', '-o', required=True, type=str, help='main output directory')
    parser.add_argument('--batch-id', default='default',
                        help='Identifier of this batch of Vampire runs. '
                             'Disambiguates name of the output directory with the batch configuration. '
                             'Useful if multiple batches share the output directory.')
    parser.add_argument('--scratch', help='temporary output directory')
    parser.add_argument('--solve-runs', type=int, default=1,
                        help='Number of Vampire executions per problem. '
                             'Each of the executions uses random predicate and function precedence.')
    # TODO: Add support for `vampire --random_seed`.
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
    # TODO: Fix or remove.
    parser.add_argument('--no-clausify', action='store_true', help='Omit clausify runs. Compatible with stock Vampire.')
    parser.add_argument('--random-predicate-precedence', action='store_true')
    parser.add_argument('--random-function-precedence', action='store_true')
    parser.add_argument('--learn-max-symbols', default=1024, type=int,
                        help='Maximum signature size with which learning is enabled.')
    # TODO: Clean up.
    parser.add_argument('--run-policy', choices=['none', 'interrupted', 'failed', 'all'], default='interrupted',
                        help='Which Vampire run configurations should be executed?')


def execution_base_score(execution):
    if execution['exit_code'] == 0:
        return execution['saturation_iterations']
    return np.nan


def pairs_hit(precedences):
    precedences = np.asarray(precedences)
    m = precedences.shape[0]
    n = precedences.shape[1]
    assert precedences.shape == (m, n)
    res = np.empty((m, n, n), np.bool)
    precedence_inverse = np.empty(n, precedences.dtype)
    for i, precedence in enumerate(precedences):
        # https://stackoverflow.com/a/25535723/4054250
        precedence_inverse[precedence] = np.arange(n)
        res[i] = np.tri(n, k=-1, dtype=np.bool).transpose()[precedence_inverse, :][:, precedence_inverse]
    return res


def preprocess_scores(scores_base, failure_penalty_quantile, failure_penalty_factor, log_scale, normalize):
    scores = scores_base.copy()
    y_pipeline_steps = list()
    quantile_imputer = None
    if failure_penalty_quantile is not None:
        quantile_imputer = QuantileImputer(copy=False, quantile=failure_penalty_quantile, factor=failure_penalty_factor)
        y_pipeline_steps.append(('quantile', quantile_imputer))
    if log_scale:
        y_pipeline_steps.append(('log', sklearn.preprocessing.FunctionTransformer(func=np.log)))
    if normalize:
        y_pipeline_steps.append(('normalize', sklearn.preprocessing.StandardScaler(copy=False)))
    y_pipeline_steps.append(('passthrough', 'passthrough'))
    # TODO: Expose the y pipeline so that it can be reused for prediction.
    y_pipeline = sklearn.pipeline.Pipeline(y_pipeline_steps)
    scores = y_pipeline.fit_transform(scores)[:, 0]
    json.dump({
        'count': len(scores),
        'nan_count': np.count_nonzero(np.isnan(scores)),
        'mean': np.nanmean(scores),
        'std': np.nanstd(scores),
        'max': np.nanmax(scores)
    }, sys.stderr, indent=4)
    if quantile_imputer is not None:
        logging.info({'failure_score': quantile_imputer.fill_value})
    return scores


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

    assert namespace.output is not None
    assert namespace.batch_id is not None
    # Multiple jobs are expected to populate a common output directory.
    output_batch = os.path.join(namespace.output, 'batches', namespace.batch_id)
    problems_txt_base_name = 'problems.txt'

    # We assume that `makedirs_open` is thread-safe.
    with makedirs_open(output_batch, 'batch.json', 'w') as output_json_file:
        json.dump({
            'output_root': os.path.relpath(namespace.output, output_batch),
            'problem_base_path': problem_base_path,
            'problems': problems_txt_base_name,
            'solve_runs_per_problem': namespace.solve_runs,
            'batch_id': namespace.batch_id,
            'scratch': namespace.scratch,
            'cwd': os.getcwd(),
            'vampire': namespace.vampire,
            'vampire_options': vampire_options,
            'timeout': namespace.timeout,
            'no_clausify': namespace.no_clausify,
            'include': namespace.include
        }, output_json_file, indent=4)
    with makedirs_open(output_batch, problems_txt_base_name, 'w') as problems_file:
        problems_file.write('\n'.join(problem_paths))
        problems_file.write('\n')

    workspace = vampyre.vampire.Workspace(path=namespace.output, program=namespace.vampire,
                                          problem_dir=problem_base_path, include_dir=namespace.include,
                                          scratch_dir=namespace.scratch,
                                          never_load=never_load, never_run=never_run,
                                          result_is_ok_to_load=result_is_ok_to_load)
    solve_dfs = []
    clausify_dfs = []
    custom_dfs = {'default': list()}
    try:
        for problem_path in problem_paths:
            problem = vampyre.vampire.Problem(os.path.relpath(problem_path, problem_base_path), workspace,
                                              base_options=vampire_options, timeout=namespace.timeout)
            try:
                clausify_dfs.append(problem.get_clausify_execution().get_dataframe(
                    field_names_obligatory=vampyre.vampire.Execution.field_names_clausify))
                executions = problem.solve_with_random_precedences(solve_count=namespace.solve_runs,
                                                                   random_predicates=namespace.random_predicate_precedence,
                                                                   random_functions=namespace.random_function_precedence)
                executions = list(executions)
                solve_dfs.extend(
                    execution.get_dataframe(field_names_obligatory=vampyre.vampire.Execution.field_names_solve) for
                    execution in executions)
                custom_points = dict()
                # Run with the default settings without randomized precedences.
                execution = problem.get_execution()
                logging.info({'default': str(execution.result)})
                if execution.result.exit_code == 0:
                    custom_points['default'] = execution['saturation_iterations']
                else:
                    custom_points['default'] = None
                custom_dfs['default'].append(
                    execution.get_dataframe(field_names_obligatory=vampyre.vampire.Execution.field_names_solve))
                saturation_iterations = [execution['saturation_iterations'] for execution in executions if
                                         execution['exit_code'] == 0]
                if max(len(problem.get_predicates()), len(problem.get_functions())) > namespace.learn_max_symbols:
                    logging.info(
                        f'Precedence learning skipped because signature is too large. Predicates: {len(problem.get_predicates())}. Functions: {len(problem.get_functions())}. Maximum: {namespace.learn_max_symbols}.')
                else:
                    scores_base = np.fromiter(map(execution_base_score, executions), dtype=np.float,
                                              count=len(executions)).reshape(-1, 1)
                    json.dump({
                        'count': len(scores_base),
                        'nan_count': np.count_nonzero(np.isnan(scores_base)),
                        'mean': np.nanmean(scores_base),
                        'std': np.nanstd(scores_base),
                        'max': np.nanmax(scores_base)
                    }, sys.stderr, indent=4)
                    symbol_type_preprocessed_data = dict()
                    for precedence_option in executions[0].configuration.precedences.keys():
                        precedences = [execution.configuration.precedences[precedence_option] for execution in
                                       executions]
                        flattener = Flattener()
                        pipeline = sklearn.pipeline.Pipeline([
                            ('pairs_hit', sklearn.preprocessing.FunctionTransformer(func=pairs_hit)),
                            ('flatten', flattener)
                        ])
                        x = pipeline.fit_transform(precedences)
                        symbol_type_preprocessed_data[precedence_option] = {
                            'x': x,
                            'flattener': flattener
                        }
                    params = itertools.chain(
                        itertools.product([False, True], [True], [1], [1, 2, 10],
                                          ['mean', 'linear_regression', 'lasso_cv'], [1.0]),
                        itertools.product([False, True], [True], [1], [1, 2, 10],
                                          ['lasso'], [0.01, 0.1, 0.2, 1.0])
                    )
                    for log_scale, normalize, failure_penalty_quantile, failure_penalty_factor, pair_scoring, alpha in params:
                        name = f'ltot_{log_scale}_{normalize}_{failure_penalty_quantile}_{failure_penalty_factor}_{pair_scoring}_{alpha}'
                        # TODO: Fit score preprocessing only on the training set.
                        y = preprocess_scores(scores_base, failure_penalty_quantile, failure_penalty_factor,
                                              log_scale, normalize)
                        good_precedences = dict()
                        stats = dict()
                        for precedence_option, data in symbol_type_preprocessed_data.items():
                            x = data['x']
                            flattener = data['flattener']
                            # TODO: Disable copying of x in transformers.
                            reg = {
                                'mean': MeanRegression(),
                                'linear_regression': sklearn.linear_model.LinearRegression(copy_X=False),
                                'lasso': sklearn.linear_model.Lasso(alpha=alpha, copy_X=False),
                                'lasso_cv': sklearn.linear_model.LassoCV(copy_X=False)
                            }[pair_scoring]
                            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,
                                                                                                        shuffle=False)
                            reg.fit(x_train, y_train)
                            pair_scores = flattener.inverse_transform(reg.coef_)[0]
                            symbol_type = {
                                'predicate_precedence': 'predicate',
                                'function_precedence': 'function'
                            }[precedence_option]
                            perm = precedence.learn_ltot(pair_scores, symbol_type=symbol_type)
                            with np.printoptions(threshold=16, edgeitems=8):
                                json.dump({
                                    'name': name,
                                    'symbol_type': symbol_type,
                                    'n_nan_pairs': np.count_nonzero(np.isnan(pair_scores)),
                                    'instance': str(reg),
                                    'train': reg.score(x_train, y_train),
                                    'test': reg.score(x_test, y_test),
                                    'greedy_precedence': str(perm)
                                }, sys.stderr, indent=4)
                            stats[(symbol_type, 'train')] = reg.score(x_train, y_train)
                            stats[(symbol_type, 'test')] = reg.score(x_test, y_test)
                            stats[(symbol_type, 'nonzero_coefs')] = np.count_nonzero(reg.coef_)
                            if pair_scoring in ['lasso', 'lasso_cv']:
                                stats[(symbol_type, 'n_iter')] = reg.n_iter_
                                if pair_scoring == 'lasso_cv':
                                    stats[(symbol_type, 'alpha')] = reg.alpha_
                                    stats[(symbol_type, 'alphas.min')] = reg.alphas_.min()
                                    stats[(symbol_type, 'alphas.max')] = reg.alphas_.max()
                            good_precedences[precedence_option] = perm
                            try:
                                output_dir = os.path.join(output_batch, 'ltot', name)
                                precedence.plot_preference_heatmap(pair_scores, perm,
                                                                   symbol_type, problem,
                                                                   output_file=os.path.join(output_dir, 'preferences',
                                                                                            f'{problem.name()}_{symbol_type}'))
                            except ValueError:
                                logging.debug('Preference heatmap plotting failed.', exc_info=True)
                        execution = problem.get_execution(precedences=good_precedences)
                        logging.info({'name': name, 'status': execution['status'], 'exit_code': execution['exit_code'],
                                      'saturation_iterations': execution['saturation_iterations']})
                        if execution.result.exit_code == 0:
                            custom_points[name] = execution['saturation_iterations']
                        else:
                            custom_points[name] = None
                        if name not in custom_dfs:
                            custom_dfs[name] = list()
                        df = execution.get_dataframe(field_names_obligatory=vampyre.vampire.Execution.field_names_solve)
                        df['log_scale'] = log_scale
                        df['normalize'] = normalize
                        df['failure_penalty_quantile'] = failure_penalty_quantile
                        df['failure_penalty_factor'] = failure_penalty_factor
                        df['pair_scoring'] = pair_scoring
                        df['alpha'] = alpha
                        for key, value in stats.items():
                            df[key] = value
                        custom_dfs[name].append(df)
                try:
                    plot_saturation_iterations_distribution(saturation_iterations, problem, len(executions),
                                                            custom_points, output_dir=os.path.join(output_batch,
                                                                                                   'saturation_iterations_distribution'))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    logging.debug(
                        f'Plotting of histogram of saturation iterations failed. Unique measurements: {len(np.unique(saturation_iterations))}.',
                        exc_info=True)
            except (RuntimeError, FileNotFoundError, json.decoder.JSONDecodeError):
                logging.debug('Solving with random precedences failed.', exc_info=True)
    finally:
        logging.info(workspace.cache_info)
        solve_runs_df = vampyre.vampire.Execution.concat_dfs(solve_dfs)
        clausify_runs_df = vampyre.vampire.Execution.concat_dfs(clausify_dfs)
        custom_dfs_joint = {name: vampyre.vampire.Execution.concat_dfs(value) for (name, value) in custom_dfs.items()}
        df_custom = vampyre.vampire.Execution.concat_dfs(
            value.assign(name=name) for (name, value) in custom_dfs_joint.items() if value is not None)
        results.save_all(solve_runs_df, clausify_runs_df, output_batch, df_custom)


def plot_saturation_iterations_distribution(saturation_iterations, problem, execution_count, custom_points=None,
                                            output_dir=None):
    plt.figure()
    sns.distplot(saturation_iterations, rug=True, norm_hist=True, kde=False, fit=scipy.stats.fisk)
    plt.title(
        f'Distribution of saturation iteration counts in successful solve runs ({len(saturation_iterations)}/{execution_count}) on problem {problem}')
    plt.ylabel('Density')
    plt.xlabel('Number of saturation iterations')
    if custom_points is not None:
        for i, (name, x) in enumerate(custom_points.items()):
            if x is not None:
                plt.axvline(x, 0, 1, label=name, color=f'C{i % 10}')
    plt.legend()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{problem.name()}.svg'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()
