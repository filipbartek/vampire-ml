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
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, RidgeCV
from sklearn.svm import LinearSVR

import vampyre
from utils import file_path_list
from utils import makedirs_open
from utils import dict_to_name
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
    parser.add_argument('--learn-max-pair-values', type=int, default=int(1e9))
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


def get_y_pipeline(failure_penalty_quantile, failure_penalty_factor, log_scale, normalize):
    y_pipeline_steps = list()
    if failure_penalty_quantile is not None:
        y_pipeline_steps.append(
            ('quantile', QuantileImputer(copy=False, quantile=failure_penalty_quantile, factor=failure_penalty_factor)))
    if log_scale:
        y_pipeline_steps.append(('log', sklearn.preprocessing.FunctionTransformer(func=np.log)))
    if normalize:
        y_pipeline_steps.append(('normalize', sklearn.preprocessing.StandardScaler(copy=False)))
    if len(y_pipeline_steps) == 0:
        y_pipeline_steps.append(('passthrough', 'passthrough'))
    return sklearn.pipeline.Pipeline(y_pipeline_steps)


def preprocess_scores(y_train, y_test, failure_penalty_quantile, failure_penalty_factor, log_scale, normalize):
    y_pipeline = get_y_pipeline(failure_penalty_quantile, failure_penalty_factor, log_scale, normalize)
    y_train = y_pipeline.fit_transform(y_train.copy())[:, 0]
    y_test = y_pipeline.transform(y_test.copy())[:, 0]
    logging.info('scores_preprocessed: ' + json.dumps({
        'y_train': {
            'count': len(y_train),
            'nan_count': np.count_nonzero(np.isnan(y_train)),
            'mean': np.nanmean(y_train),
            'std': np.nanstd(y_train),
            'min': np.nanmin(y_train),
            'max': np.nanmax(y_train)
        },
        'y_test': {
            'count': len(y_test),
            'nan_count': np.count_nonzero(np.isnan(y_test)),
            'mean': np.nanmean(y_test),
            'std': np.nanstd(y_test),
            'min': np.nanmin(y_test),
            'max': np.nanmax(y_test)
        }
    }, indent=4))
    try:
        logging.debug({'failure_score': y_pipeline['quantile'].fill_value})
    except KeyError:
        pass
    return y_train, y_test, y_pipeline


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
                pair_values = max(len(problem.get_predicates()), len(problem.get_functions())) ** 2 * len(executions)
                logging.info(json.dumps({
                    'n_executions': len(executions),
                    'n_predicates': len(problem.get_predicates()),
                    'n_functions': len(problem.get_functions()),
                    'n_pair_values': pair_values,
                    'max_pair_values': namespace.learn_max_pair_values
                }, indent=4))
                if pair_values > namespace.learn_max_pair_values:
                    logging.info('Precedence learning skipped because the training data is too large.')
                else:
                    scores_base = np.fromiter(map(execution_base_score, executions), dtype=np.float,
                                              count=len(executions)).reshape(-1, 1)
                    logging.info('scores_base: ' + json.dumps({
                        'count': len(scores_base),
                        'nan_count': np.count_nonzero(np.isnan(scores_base)),
                        'mean': np.nanmean(scores_base),
                        'std': np.nanstd(scores_base),
                        'max': np.nanmax(scores_base)
                    }, indent=4))
                    precedence_options = executions[0].configuration.precedences.keys()
                    precedences_by_symbol_type = {
                        precedence_option: [execution.configuration.precedences[precedence_option] for execution in
                                            executions] for precedence_option in precedence_options}
                    splitting = sklearn.model_selection.train_test_split(scores_base,
                                                                         *precedences_by_symbol_type.values(),
                                                                         shuffle=False)
                    assert len(splitting) == 2 * (len(precedence_options) + 1)
                    y_base_train = splitting[0]
                    y_base_test = splitting[1]
                    symbol_type_preprocessed_data = dict()
                    for i, precedence_option in enumerate(precedence_options):
                        x_train = splitting[2 * (i + 1)]
                        x_test = splitting[2 * (i + 1) + 1]
                        flattener = Flattener()
                        pipeline = sklearn.pipeline.Pipeline([
                            ('pairs_hit', sklearn.preprocessing.FunctionTransformer(func=pairs_hit)),
                            ('flatten', flattener)
                        ])
                        x_train = pipeline.fit_transform(x_train)
                        x_test = pipeline.transform(x_test)
                        symbol_type_preprocessed_data[precedence_option] = {
                            'x_train': x_train,
                            'x_test': x_test,
                            'flattener': flattener
                        }
                    params = itertools.product([False, True], [True], [1], [1, 2, 10], [
                        (MeanRegression(), None),
                        (LinearRegression(copy_X=False), None),
                        (LassoCV(copy_X=False), None),
                        (Lasso(alpha=0.01, copy_X=False), {'alpha': 0.01}),
                        (RidgeCV(), None),
                        (LinearSVR(C=0.1), {'C': 0.1})
                    ])
                    for log_scale, normalize, failure_penalty_quantile, failure_penalty_factor, (reg, reg_params) in params:
                        reg_type_name = type(reg).__name__
                        reg_name = f'{reg_type_name}_{dict_to_name(reg_params)}'
                        logging.info(json.dumps({
                            'log_scale': log_scale,
                            'normalize': normalize,
                            'failure_penalty_quantile': failure_penalty_quantile,
                            'failure_penalty_factor': failure_penalty_factor,
                            'reg': {
                                'name': reg_name,
                                'type_name': reg_type_name,
                                'params': reg_params,
                                'instance': str(reg)
                            }
                        }, indent=4))
                        name = f'ltot-{log_scale}-{normalize}-{failure_penalty_quantile}-{failure_penalty_factor}-{reg_name}'
                        logging.info(f'Processing {name}')
                        y_train, y_test, y_pipeline = preprocess_scores(y_base_train, y_base_test,
                                                                        failure_penalty_quantile,
                                                                        failure_penalty_factor, log_scale, normalize)
                        good_precedences = dict()
                        stats = dict()
                        for precedence_option, data in symbol_type_preprocessed_data.items():
                            x_train = data['x_train']
                            x_test = data['x_test']
                            flattener = data['flattener']
                            reg.fit(x_train, y_train)
                            pair_scores = flattener.inverse_transform(reg.coef_)[0]
                            symbol_type = {
                                'predicate_precedence': 'predicate',
                                'function_precedence': 'function'
                            }[precedence_option]
                            perm = precedence.learn_ltot(pair_scores, symbol_type=symbol_type)
                            cur_stats = {
                                'pair_scores.n_nan': np.count_nonzero(np.isnan(pair_scores)),
                                'reg.score.train': reg.score(x_train, y_train),
                                'reg.score.test': reg.score(x_test, y_test),
                                'reg.coefs.n': len(reg.coef_),
                                'reg.coefs.n_nonzero': np.count_nonzero(reg.coef_)
                            }
                            if isinstance(reg, LassoCV) or isinstance(reg, RidgeCV):
                                cur_stats['reg.alpha'] = reg.alpha_
                            if isinstance(reg, Lasso) or isinstance(reg, LassoCV) or isinstance(reg, LinearSVR):
                                cur_stats['reg.n_iter'] = int(reg.n_iter_)
                            if isinstance(reg, LassoCV):
                                cur_stats['reg.alphas.min'] = reg.alphas_.min()
                                cur_stats['reg.alphas.max'] = reg.alphas_.max()
                                plot_mse_path(reg, os.path.join(output_batch, 'mse_path', str(problem), symbol_type,
                                                                f'{name}.svg'))
                            with np.printoptions(threshold=16, edgeitems=8):
                                logging.info(json.dumps({
                                    'symbol_type': symbol_type,
                                    'stats': cur_stats,
                                    'greedy_precedence': str(perm)
                                }, indent=4))
                            for key, value in cur_stats.items():
                                stats[(symbol_type, key)] = value
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
                        logging.info(json.dumps({
                            'status': execution['status'],
                            'exit_code': execution['exit_code'],
                            'saturation_iterations': execution['saturation_iterations'],
                            'score': y_pipeline.transform([[execution_base_score(execution)]])[0, 0]
                        }, indent=4))
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
                        df['reg_type_name'] = reg_type_name
                        if reg_params is not None:
                            for key, value in reg_params.items():
                                df[('reg', key)] = value
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


def plot_mse_path(reg, output_file=None):
    plt.figure()
    plt.title(output_file)
    plt.errorbar(x=reg.alphas_, y=reg.mse_path_.mean(axis=1), yerr=reg.mse_path_.std(axis=1), label='Alphas grid')
    plt.axvline(reg.alpha_, 0, 1, label=f'Final alpha ({reg.alpha_})', color='C1')
    plt.xscale('log')
    plt.ylabel('MSE')
    plt.xlabel('alpha')
    plt.legend()
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, bbox_inches='tight')
        logging.debug(f'Plot saved: {output_file}')
    else:
        plt.show()
