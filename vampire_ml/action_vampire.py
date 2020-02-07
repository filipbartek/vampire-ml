#!/usr/bin/env python3.7

import datetime
import json
import logging
import os
import sys
from collections import Counter
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from tqdm import tqdm

import vampyre
from utils import file_path_list
from utils import makedirs_open, truncate
from . import precedence
from . import results


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('problem', type=str, nargs='*', help='glob pattern of a problem path')
    parser.add_argument('--problem-list', action='append', default=[], help='input file with a list of problem paths')
    parser.add_argument('--problem-base-path', type=str, help='the problem paths are relative to the base path')
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
                             'For example, "{include: $TPTP, time_limit: 10}" translates into '
                             '"--include $TPTP --time_limit 10".'
                             'Recommended options: include, time_limit.')
    parser.add_argument('--timeout', type=float, default=20,
                        help='Time in seconds after which each Vampire call is terminated.')
    parser.add_argument('--no-clausify', action='store_true', help='Omit clausify runs. Compatible with stock Vampire.')
    parser.add_argument('--random-predicate-precedence', action='store_true')
    parser.add_argument('--random-function-precedence', action='store_true')
    parser.add_argument('--learn-max-symbols', default=1024, type=int,
                        help='Maximum signature size with which learning is enabled.')
    parser.add_argument('--run-policy', choices=['none', 'interrupted', 'failed', 'all'], default='interrupted',
                        help='Which Vampire run configurations should be executed?')


def call(namespace):
    # SWV567-1.014.p has clause depth of more than the default recursion limit of 1000,
    # making `json.load()` raise `RecursionError`.
    sys.setrecursionlimit(2000)

    if namespace.no_clausify and namespace.solve_runs > 1:
        logging.warning('Clausification disabled and while running more than 1 solve run per problem.')

    problem_paths, problem_base_path = file_path_list.compose(namespace.problem_list, namespace.problem,
                                                              namespace.problem_base_path)

    vampire_options = dict()
    for value in chain(*namespace.vampire_options):
        vampire_options.update(value)

    assert namespace.output is not None
    assert namespace.batch_id is not None
    # Multiple jobs are expected to populate a common output directory.
    output_batch = os.path.join(namespace.output, 'batches', namespace.batch_id)
    output_problems = os.path.join(namespace.output, 'problems')

    logs_path = os.path.join(output_batch, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    problems_txt_base_name = 'problems.txt'

    # We assume that `makedirs_open` is thread-safe.
    with makedirs_open(output_batch, 'batch.json', 'w') as output_json_file:
        json.dump({
            'output_root': os.path.relpath(namespace.output, output_batch),
            'output_problems': os.path.relpath(output_problems, output_batch),
            'problem_base_path': problem_base_path,
            'problems': problems_txt_base_name,
            'logs': os.path.relpath(logs_path, output_batch),
            'solve_runs_per_problem': namespace.solve_runs,
            'batch_id': namespace.batch_id,
            'scratch': namespace.scratch,
            'cwd': os.getcwd(),
            'vampire': namespace.vampire,
            'vampire_options': vampire_options,
            'timeout': namespace.timeout,
            'no_clausify': namespace.no_clausify
        }, output_json_file, indent=4)
    with makedirs_open(output_batch, problems_txt_base_name, 'w') as problems_file:
        problems_file.write('\n'.join(problem_paths))
        problems_file.write('\n')

    symbol_types = list()
    if namespace.random_predicate_precedence:
        symbol_types.append('predicate')
    if namespace.random_function_precedence:
        symbol_types.append('function')
    assert set(symbol_types) <= set(vampyre.SymbolPrecedence.symbol_types.keys())
    if namespace.solve_runs > 1 and len(symbol_types) == 0:
        logging.warning('Requesting multiple runs per problem with identical parameters.')

    clausify_run_table = None
    if not namespace.no_clausify:
        clausify_run_table = vampyre.RunTable(vampyre.Run.field_names_clausify)
    solve_run_table = vampyre.RunTable(vampyre.Run.field_names_solve)
    learned_run_table = vampyre.RunTable(vampyre.Run.field_names_solve)
    summary_writer = tf.summary.create_file_writer(logs_path)
    base_configuration = vampyre.Run(namespace.vampire, base_options=vampire_options,
                                     timeout=namespace.timeout, output_dir=output_problems,
                                     problem_base_path=problem_base_path, scratch_dir=namespace.scratch,
                                     run_policy=namespace.run_policy)
    try:
        with tqdm(desc='Running Vampire',
                  total=len(problem_paths) * ((not namespace.no_clausify) + namespace.solve_runs),
                  unit='run') as t:
            stats = {'hits': Counter(), 'clausify': Counter(), 'solve': Counter()}
            solve_i = 0
            for problem_path in problem_paths:
                problem_name = os.path.relpath(problem_path, problem_base_path)
                logger = logging.getLogger(f'vampire-ml.problem.{problem_name}')
                problem_configuration = base_configuration.spawn(problem_path=problem_path, output_dir_rel=problem_name)
                clausify_run = None
                if clausify_run_table is not None:
                    clausify_run = problem_configuration.spawn(output_dir_rel='clausify',
                                                               base_options={'mode': 'clausify'})
                    t.set_postfix({'hits': stats['hits'], 'current_run': clausify_run})
                    # TODO: Refine the conditions under which we skip execution. Shall we retry runs that terminated with code -11?
                    loaded = clausify_run.load_or_run()
                    clausify_run_table.add_run(clausify_run)
                    stats['hits'][loaded] += 1
                    stats['clausify'][(clausify_run.status, clausify_run.exit_code)] += 1
                    t.set_postfix({'hits': stats['hits'], 'current_run': clausify_run})
                    t.update()
                if clausify_run is not None and clausify_run.exit_code != 0:
                    t.update(namespace.solve_runs)
                    logger.debug('Skipping solve runs because clausify run failed.')
                    continue
                problem_solve_runs = list()
                # TODO: Parallelize.
                # TODO: Consider exhausting all permutations if they fit in `namespace.solve_runs`. Watch out for imbalance in distribution when learning from all problems.
                for solve_run_i in range(namespace.solve_runs):
                    # TODO: Run two runs for each permutation: forward and reversed.
                    precedences = None
                    if clausify_run is not None:
                        assert clausify_run.exit_code == 0
                        precedences = {symbol_type: clausify_run.random_precedence(symbol_type, seed=solve_run_i)
                                       for symbol_type in symbol_types}
                    solve_run = problem_configuration.spawn(precedences=precedences,
                                                            output_dir_rel=os.path.join('solve', str(solve_run_i)),
                                                            base_options={'mode': 'vampire'})
                    t.set_postfix({'hits': stats['hits'], 'current_run': solve_run})
                    loaded = solve_run.load_or_run()
                    problem_solve_runs.append(solve_run)
                    stats['hits'][loaded] += 1
                    stats['solve'][(solve_run.status, solve_run.exit_code)] += 1
                    t.set_postfix({'hits': stats['hits'], 'current_run': solve_run})
                    t.update()
                    with summary_writer.as_default():
                        tf.summary.text('stats', str(stats), step=solve_i)
                    solve_i += 1
                    # TODO: Consider unloading immediately.
                saturation_iterations_min, saturation_iterations_max, saturation_iterations = assign_scores(
                    problem_solve_runs)
                solve_run_table.extend(problem_solve_runs)
                if saturation_iterations_min is None or saturation_iterations_max is None:
                    logger.debug(
                        'Precedence learning skipped because there are no successful solve runs to learn from.')
                    continue
                try:
                    plot_saturation_iterations_distribution(saturation_iterations, problem_name,
                                                            problem_configuration.output_dir)
                except (ValueError, np.linalg.LinAlgError):
                    logger.debug(
                        f'Plotting of histogram of saturation iterations failed. Unique measurements: {len(np.unique(saturation_iterations))}.',
                        exc_info=True)
                if clausify_run is None:
                    logger.debug('Precedence learning skipped because probing clausification was not performed.')
                    continue
                if len(symbol_types) == 0:
                    logger.debug('Precedence learning skipped because no symbol type was randomized.')
                    continue
                symbol_counts = {symbol_type: clausify_run.get_symbol_count(symbol_type) for symbol_type in
                                 symbol_types}
                if max(symbol_counts.values()) > namespace.learn_max_symbols:
                    logger.debug('Precedence learning skipped because signature is too large.')
                    continue
                good_permutations = precedence.learn(problem_solve_runs, symbol_counts)
                # TODO: Try to improve the permutation by two-point swaps (hillclimbing).
                if clausify_run is not None:
                    assert clausify_run.exit_code == 0
                    precedences = {
                        symbol_type: vampyre.SymbolPrecedence(symbol_type, good_permutations[symbol_type][0]) for
                        symbol_type in symbol_types}
                    solve_run = problem_configuration.spawn(precedences=precedences,
                                                            output_dir_rel=os.path.join('learned'))
                    solve_run.load_or_run()
                    assign_score(solve_run, saturation_iterations_min, saturation_iterations_max)
                    learned_run_table.add_run(solve_run)
                for symbol_type, (permutation, preference_matrix) in good_permutations.items():
                    symbols = clausify_run.get_symbols(symbol_type)
                    try:
                        plot_preference_heatmap(preference_matrix, permutation, symbol_type, symbols, solve_run)
                    except ValueError:
                        logger.debug('Preference heatmap plotting failed.', exc_info=True)
    finally:
        solve_runs_df = solve_run_table.get_data_frame()
        clausify_runs_df = None
        if clausify_run_table is not None:
            clausify_runs_df = clausify_run_table.get_data_frame()
        results.save_all(solve_runs_df, clausify_runs_df, output_batch)
        results.save_df(learned_run_table.get_data_frame(), 'runs_learned', output_batch)


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
    run.score = run_score(run.exit_code, run.saturation_iterations(), saturation_iterations_min,
                          saturation_iterations_max)


def assign_scores(runs):
    saturation_iterations = np.asarray(
        [run.saturation_iterations() for run in runs if run.exit_code == 0 and run.saturation_iterations() is not None])
    saturation_iterations_min = None
    saturation_iterations_max = None
    if len(saturation_iterations) >= 1:
        saturation_iterations_min = saturation_iterations.min()
        saturation_iterations_max = saturation_iterations.max()
    for run in runs:
        assign_score(run, saturation_iterations_min, saturation_iterations_max)
    return saturation_iterations_min, saturation_iterations_max, saturation_iterations


def plot_saturation_iterations_distribution(saturation_iterations, problem_name, output_dir):
    plt.figure()
    sns.distplot(saturation_iterations, rug=True)
    plt.title(f'Distribution of saturation iteration counts in successful solve runs on problem {problem_name}')
    plt.ylabel('Density')
    plt.xlabel('Number of saturation iterations')
    plt.savefig(os.path.join(output_dir, 'solve', 'saturation_iterations.svg'), bbox_inches='tight')
    plt.close()


def plot_preference_heatmap(v, permutation, symbol_type, symbols, solve_run):
    n = len(symbols)
    assert v.shape == (n, n)
    assert len(permutation) == n
    assert np.array_equal(v[permutation, :][:, permutation], v[:, permutation][permutation, :])
    # Vampire forces '=' to be the first predicate.
    assert symbol_type != 'predicate' or (symbols.name[0] == '=' and permutation[0] == 0)
    v_permuted = v[permutation, :][:, permutation]
    tick_labels = False
    if n <= 32:
        tick_labels = [f'{truncate(symbols.name[i], 16)}' for i in permutation]
    file_type = 'svg'
    if n > 64:
        # For large n, the svg file size is too large.
        file_type = 'png'
    # We mask the diagonal because the values on the diagonal don't have a sensible interpretation.
    plt.figure()
    sns.heatmap(v_permuted, xticklabels=tick_labels, yticklabels=tick_labels,
                mask=np.eye(v_permuted.shape[0], dtype=np.bool), square=True)
    plt.title(
        f'Expected pairwise preferences of {symbol_type} symbols in problem {os.path.relpath(solve_run.problem_path, solve_run.problem_base_path)}')
    plt.ylabel('Early symbol')
    plt.xlabel('Late symbol')
    plt.savefig(os.path.join(solve_run.output_dir, f'{symbol_type}_precedence_preferences.{file_type}'),
                bbox_inches='tight')
    plt.close()


def construct_good_permutation(v):
    """Find good permutation greedily."""
    assert len(v.shape) == 2
    assert v.shape[0] == v.shape[1]
    n = v.shape[0]
    # s[i] = total score for row i - total score for column i
    # Symbol i should be picked as the first greedily if it maximizes s[i].
    s = v.sum(axis=1).flatten() - v.sum(axis=0).flatten()
    perm = np.zeros(n, dtype=np.uint)
    # https://papers.nips.cc/paper/1431-learning-to-order-things.pdf
    for i in range(n):
        cur = np.argmax(s)
        perm[i] = cur
        s += v[cur, :] - v[:, cur]
        s[cur] = np.NINF
    return perm
