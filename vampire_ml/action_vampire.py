#!/usr/bin/env python3.7

import json
import logging
import os
import sys
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

import vampyre
from utils import file_path_list
from utils import makedirs_open
from . import precedence
from . import results


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


def learn_ltot(executions, problem=None, output_dir=None):
    good_precedences = dict()
    for precedence_option in executions[0].configuration.precedences.keys():
        precedence_scores = ((execution.configuration.precedences[precedence_option], execution.score)
                             for execution in executions)
        symbol_type = {'predicate_precedence': 'predicate', 'function_precedence': 'function'}[precedence_option]
        good_precedences[precedence_option] = precedence.learn_precedence(precedence_scores, symbol_type, problem,
                                                                          output_dir=output_dir)
    return good_precedences


def learn_ltot_successful(executions, problem=None, output_dir=None):
    return learn_ltot([execution for execution in executions if execution.result.exit_code == 0], problem, output_dir)


def learn_ltot_lexicographic(executions, problem=None, output_dir=None):
    good_precedences = dict()
    for precedence_option in executions[0].configuration.precedences.keys():
        precedence_scores = ((execution.configuration.precedences[precedence_option],
                              (execution['exit_code'] == 0, execution['saturation_iterations']))
                             for execution in executions)
        symbol_type = {'predicate_precedence': 'predicate', 'function_precedence': 'function'}[precedence_option]
        good_precedences[precedence_option] = precedence.learn_precedence_lex(precedence_scores, symbol_type, problem,
                                                                              output_dir=output_dir)
    return good_precedences


def learn_best(executions, problem=None, output_dir=None):
    best_score = None
    best_execution = None
    for execution in executions:
        if best_score is None or execution.score < best_score:
            best_score = execution.score
            best_execution = execution
    return best_execution.configuration.precedences


precedence_learners = {
    'ltot': learn_ltot,
    'ltot_successful': learn_ltot_successful,
    'ltot_lex': learn_ltot_lexicographic,
    'best': learn_best
}


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

    # TODO: Test scratch dir functionality.
    workspace = vampyre.vampire.Workspace(path=namespace.output, program=namespace.vampire,
                                          problem_dir=problem_base_path, include_dir=namespace.include,
                                          scratch_dir=namespace.scratch,
                                          never_load=never_load, never_run=never_run,
                                          result_is_ok_to_load=result_is_ok_to_load)
    solve_dfs = []
    clausify_dfs = []
    learned_dfs = {name: [] for name in precedence_learners.keys()}
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
                _, _, saturation_iterations = assign_scores(executions)
                solve_dfs.extend(
                    execution.get_dataframe(field_names_obligatory=vampyre.vampire.Execution.field_names_solve) for
                    execution in executions)
                custom_points = dict()
                if max(len(problem.get_predicates()), len(problem.get_functions())) > namespace.learn_max_symbols:
                    logging.debug('Precedence learning skipped because signature is too large.')
                else:
                    for name, learn_precedence in precedence_learners.items():
                        try:
                            precedences = learn_precedence(executions, problem=problem,
                                                           output_dir=os.path.join(output_batch, name))
                        except IndexError:
                            logging.debug('Learning failed.', exc_info=True)
                            continue
                        # TODO: Try to improve the permutation by two-point swaps (hillclimbing).
                        execution = problem.get_execution(precedences=precedences)
                        logging.info({name: str(execution.result)})
                        if execution.result.exit_code == 0:
                            custom_points[name] = execution['saturation_iterations']
                        learned_dfs[name].append(
                            execution.get_dataframe(field_names_obligatory=vampyre.vampire.Execution.field_names_solve))
                try:
                    plot_saturation_iterations_distribution(saturation_iterations, problem, custom_points,
                                                            output_dir=os.path.join(output_batch,
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
        results.save_all(solve_runs_df, clausify_runs_df, output_batch)
        for name, dfs in learned_dfs.items():
            df = vampyre.vampire.Execution.concat_dfs(dfs)
            results.save_all(df, None, os.path.join(output_batch, name))


def run_score(exit_code, saturation_iterations, saturation_iterations_min, saturation_iterations_max):
    if exit_code == 0:
        if saturation_iterations is None:
            raise RuntimeError('A result is missing the number of saturation loop iterations.')
        if saturation_iterations_min == saturation_iterations == saturation_iterations_max:
            return 0
        return np.interp(saturation_iterations, [saturation_iterations_min, saturation_iterations_max], [0, 1])
    else:
        return 2


def assign_score(run, saturation_iterations_min, saturation_iterations_max):
    run.score = run_score(run.result.exit_code, run['saturation_iterations'], saturation_iterations_min,
                          saturation_iterations_max)


def assign_scores(runs):
    saturation_iterations = np.asarray(
        [run['saturation_iterations'] for run in runs if
         run.result.exit_code == 0 and run['saturation_iterations'] is not None])
    saturation_iterations_min = None
    saturation_iterations_max = None
    if len(saturation_iterations) >= 1:
        saturation_iterations_min = saturation_iterations.min()
        saturation_iterations_max = saturation_iterations.max()
    for run in runs:
        assign_score(run, saturation_iterations_min, saturation_iterations_max)
    return saturation_iterations_min, saturation_iterations_max, saturation_iterations


def plot_saturation_iterations_distribution(saturation_iterations, problem, custom_points=None, output_dir=None):
    plt.figure()
    # TODO: Fit lognorm or log-logistic similar.
    sns.distplot(saturation_iterations, rug=True, norm_hist=True)
    plt.title(f'Distribution of saturation iteration counts in successful solve runs on problem {problem}')
    plt.ylabel('Density')
    plt.xlabel('Number of saturation iterations')
    if custom_points is not None:
        for i, (name, x) in enumerate(custom_points.items()):
            plt.axvline(x, 0, 1, label=name, color=f'C{i % 10}')
    plt.legend()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{problem.name()}.svg'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()
