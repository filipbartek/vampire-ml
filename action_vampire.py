#!/usr/bin/env python3.7

import datetime
import json
import logging
import os
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf
import yaml
from tqdm import tqdm

import file_path_list
import vampire
from utils import save_df, fill_category_na, makedirs_open


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
                             '"--input $TPTP --time_limit 10".'
                             'Recommended options: time_limit, mode, include.')
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--no-clausify', action='store_true', help='Omit clausify runs. Compatible with stock Vampire.')


def call(namespace):
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

    clausify_run_table = None
    if not namespace.no_clausify:
        clausify_run_table = vampire.RunTable()
    solve_run_table = vampire.RunTable()
    summary_writer = tf.summary.create_file_writer(logs_path)
    base_configuration = vampire.Run(namespace.vampire, base_options=vampire_options,
                                     timeout=namespace.timeout, output_dir=output_problems,
                                     scratch_dir=namespace.scratch)
    with tqdm(desc='Running Vampire', total=len(problem_paths) * ((not namespace.no_clausify) + namespace.solve_runs),
              unit='run') as t:
        stats = {'hits': Counter(), 'clausify': Counter(), 'solve': Counter()}
        solve_i = 0
        for problem_path in problem_paths:
            problem_name = os.path.relpath(problem_path, problem_base_path)
            problem_configuration = base_configuration.spawn(problem_path=problem_path, output_dir_rel=problem_name)
            clausify_run = None
            if clausify_run_table is not None:
                clausify_run = problem_configuration.spawn(output_dir_rel='clausify', base_options={'mode': 'clausify'})
                # TODO: Refine the conditions under which we skip execution. Shall we retry runs that terminated with code -11?
                loaded = clausify_run.load_or_run()
                clausify_run_table.add_run(clausify_run)
                stats['hits'][loaded] += 1
                stats['clausify'][(clausify_run.status, clausify_run.exit_code)] += 1
                t.set_postfix_str(stats['hits'])
                t.update()
            if clausify_run is None or clausify_run.exit_code == 0:
                # TODO: Parallelize.
                for solve_run_i in range(namespace.solve_runs):
                    precedences = None
                    if clausify_run is not None:
                        assert clausify_run.exit_code == 0
                        precedences = {'predicate': clausify_run.random_precedence('predicate', solve_run_i),
                                       'function': clausify_run.random_precedence('function', solve_run_i)}
                    # TODO: As an alterantive to setting precedences explicitly, add support for `vampire --random_seed`.
                    solve_execution = problem_configuration.spawn(precedences=precedences,
                                                                  output_dir_rel=os.path.join('solve',
                                                                                              str(solve_run_i)),
                                                                  base_options={'mode': 'vampire'})
                    loaded = solve_execution.load_or_run()
                    solve_run_table.add_run(solve_execution, clausify_run)
                    stats['hits'][loaded] += 1
                    stats['solve'][(solve_execution.status, solve_execution.exit_code)] += 1
                    t.set_postfix_str(stats['hits'])
                    t.update()
                    with summary_writer.as_default():
                        tf.summary.text('stats', str(stats), step=solve_i)
                    solve_i += 1

    solve_runs_df = solve_run_table.get_data_frame()
    save_df(solve_runs_df, 'runs_solve', output_batch)

    print('Solve runs info:')
    solve_runs_df.info()

    fill_category_na(solve_runs_df)

    termination_fieldnames = ['status', 'exit_code', 'termination_reason', 'termination_phase']

    # Distribution of run terminations
    terminations = solve_runs_df.groupby(termination_fieldnames).size()
    print('Distribution of solve run terminations:', terminations, sep='\n')
    print(terminations, file=open(os.path.join(output_batch, 'runs_solve_terminations.txt'), 'w'))

    clausify_runs_df = None
    if clausify_run_table is not None:
        clausify_runs_df = clausify_run_table.get_data_frame()
        save_df(clausify_runs_df, 'runs_clausify', output_batch)
        print('Clausify runs info:')
        clausify_runs_df.info()

    problems_df = generate_problems_df(problem_paths, solve_runs_df, clausify_runs_df)
    print('Problems info:')
    problems_df.info()
    save_df(problems_df, 'problems', output_batch)

    problems_interesting_df = problems_df[
        (problems_df.n_completed >= namespace.solve_runs) & (problems_df.n_exit_0 >= 1) & (
                problems_df.n_exit_0 + problems_df.n_exit_1 == problems_df.n_total)]
    print(f'Number of interesting problems: {len(problems_interesting_df)}')
    # TODO: Sort the rows by more criteria, for example time_elapsed mean.
    if ('saturation_iterations', 'variation') in problems_interesting_df:
        problems_interesting_df = problems_interesting_df.sort_values(('saturation_iterations', 'variation'),
                                                                      ascending=False)
    save_df(problems_interesting_df, 'problems_interesting', output_batch)


def generate_problems_df(problem_paths, runs_df, probe_runs_df=None):
    problems_df = pd.DataFrame(index=problem_paths)
    problems_df.index.name = 'problem_path'
    problem_groups = runs_df.groupby(['problem_path'])
    problems_df = problems_df.join(problem_groups.size().astype(pd.UInt64Dtype()).to_frame('n_total'))
    problems_df = problems_df.join(
        runs_df[runs_df.status == 'completed'].groupby(['problem_path']).size().astype(pd.UInt64Dtype()).to_frame(
            'n_completed'))
    problems_df = problems_df.join(
        runs_df[runs_df.exit_code == 0].groupby(['problem_path']).size().astype(pd.UInt64Dtype()).to_frame('n_exit_0'))
    problems_df = problems_df.join(
        runs_df[runs_df.exit_code == 1].groupby(['problem_path']).size().astype(pd.UInt64Dtype()).to_frame('n_exit_1'))
    if 'termination_reason' in runs_df:
        problems_df = problems_df.join(
            runs_df[runs_df.termination_reason == 'Refutation'].groupby(['problem_path']).size().astype(
                pd.UInt64Dtype()).to_frame(
                'n_refutation'))
        problems_df = problems_df.join(
            runs_df[runs_df.termination_reason == 'Satisfiable'].groupby(['problem_path']).size().astype(
                pd.UInt64Dtype()).to_frame(
                'n_satisfiable'))
        problems_df = problems_df.join(
            runs_df[runs_df.termination_reason == 'Time limit'].groupby(['problem_path']).size().astype(
                pd.UInt64Dtype()).to_frame(
                'n_time_limit'))
    problems_df.fillna(
        {'n_total': 0, 'n_completed': 0, 'n_exit_0': 0, 'n_exit_1': 0, 'n_refutation': 0, 'n_satisfiable': 0,
         'n_time_limit': 0},
        inplace=True)
    agg_fields = ['time_elapsed_process', 'time_elapsed_vampire', 'memory_used', 'saturation_iterations']
    agg_functions = [np.mean, np.std, scipy.stats.variation, np.min, np.max]
    problems_df = problems_df.join(problem_groups.agg({field_name: agg_functions for field_name in agg_fields}))
    # Merge probe run results into `problems_df`
    if probe_runs_df is not None:
        problems_df = problems_df.join(
            probe_runs_df[['problem_path', 'predicates_count', 'functions_count', 'clauses_count']].set_index(
                'problem_path'),
            rsuffix='probe')
    return problems_df
