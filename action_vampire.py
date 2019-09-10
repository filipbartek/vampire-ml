#!/usr/bin/env python3.7

import collections
import itertools
import json
import logging
import os

from tqdm import tqdm

import file_path_list
from batch import Batch
from lazy_csv_writer import LazyCsvWriter

default_vampire_options = {
    'probe': {
        '--encode': 'on',
        '--time_limit': '10',
        '--mode': 'clausify'
    },
    'solve': {
        '--encode': 'on',
        '--time_limit': '10',
        '--statistics': 'full',
        '--time_statistics': 'on',
        '--proof': 'off',
        '--symbol_precedence': 'scramble'
    }
}


def compose_vampire_options(options_common, options_specific, name):
    vampire_options = list(
        itertools.chain(*(s.split() for s in options_common), *(s.split() for s in options_specific)))
    for option, value in default_vampire_options[name].items():
        if option not in vampire_options:
            logging.info(f'Defaulting {name}: {option} {value}')
            vampire_options.extend([option, value])
    return vampire_options


def join_output_path(start, path):
    if start is None and path is None:
        raise RuntimeError('Both of the paths are null.')
    if start is None:
        return path
    if path is None:
        return start
    return os.path.join(start, path)


def call(namespace):
    vampire_options_probe = compose_vampire_options(namespace.vampire_options, namespace.vampire_options_probe, 'probe')
    vampire_options_solve = compose_vampire_options(namespace.vampire_options, namespace.vampire_options_solve, 'solve')
    problem_paths, problem_base_path = file_path_list.compose(namespace.problem_list, namespace.problem,
                                                              namespace.problem_base_path)
    try:
        output_batch = join_output_path(namespace.output, namespace.output_batch)
    except RuntimeError:
        logging.error('At least one of the options --output and --output_batch must be specified.')
        return
    try:
        output_runs = join_output_path(namespace.output, namespace.output_runs)
    except RuntimeError:
        logging.error('At least one of the options --output and --output_runs must be specified.')
        return
    output_runs_relative = os.path.abspath(output_runs)
    if (namespace.output_batch is None or not os.path.isabs(namespace.output_batch)) and (
            namespace.output_runs is None or not os.path.isabs(namespace.output_runs)):
        output_runs_relative = os.path.relpath(output_runs, output_batch)

    csv_file_path = os.path.join(output_batch, 'runs.csv')
    problems_path = os.path.join(output_batch, 'problems.txt')
    problems_successful_path = os.path.join(output_batch, 'problems_successful.txt')

    os.makedirs(output_batch, exist_ok=True)
    with open(os.path.join(output_batch, 'batch.json'), 'w') as output_json_file:
        json.dump({
            'run_output_base_path': output_runs_relative,
            'runs_csv': os.path.relpath(csv_file_path, output_batch),
            'problem_base_path': problem_base_path,
            'problems': os.path.relpath(problems_path, output_batch),
            'problems_successful': os.path.relpath(problems_successful_path, output_batch),
            'solve_runs_per_problem': namespace.solve_runs,
            'jobs': namespace.jobs,
            'vampire': namespace.vampire,
            'vampire_options_probe': vampire_options_probe,
            'vampire_options_solve': vampire_options_solve
        }, output_json_file, indent=4)
    with open(problems_path, 'w') as problems_file:
        problems_file.write('\n'.join(problem_paths))
        problems_file.write('\n')
    batch = Batch(namespace.vampire, vampire_options_probe, vampire_options_solve, output_runs, namespace.solve_runs,
                  namespace.timeout_probe, namespace.timeout_solve, namespace.jobs)
    problems_successful = set()
    with open(csv_file_path, 'w') as csv_file, open(problems_successful_path, 'w') as problems_successful_file:
        csv_writer = LazyCsvWriter(csv_file)
        with tqdm(desc='Running Vampire', total=len(problem_paths), unit='problems') as t:
            solve_runs = collections.Counter()
            stats = collections.Counter()
            t.set_postfix_str(stats)
            for result in batch.generate_results(problem_paths, namespace.probe, problem_base_path):
                csv_writer.writerow({
                    'output_path': os.path.relpath(result['paths']['output'], output_runs),
                    'problem_path': os.path.relpath(result['paths']['problem'], problem_base_path),
                    'probe': result['probe'],
                    'timeout': result['timeout'],
                    'exit_code': result['exit_code'],
                    'time_elapsed': result['time_elapsed']
                })
                if result['exit_code'] == 0 and result['paths']['problem'] not in problems_successful:
                    problems_successful_file.write(result['paths']['problem'] + '\n')
                    problems_successful.add(result['paths']['problem'])
                if result['probe']:
                    mode = 'probe'
                else:
                    mode = 'solve'
                if result['timeout']:
                    termination = 'timeout'
                    assert result['exit_code'] is None
                else:
                    assert result['exit_code'] is not None
                    termination = result['exit_code']
                stats[(mode, termination)] += 1
                t.set_postfix_str(stats)
                if result['probe'] and (namespace.solve_runs == 0 or result['timeout'] or result['exit_code'] != 0):
                    t.update(1)
                if not result['probe']:
                    solve_runs[result['paths']['problem']] += 1
                    if solve_runs[result['paths']['problem']] == namespace.solve_runs:
                        t.update(1)


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('problem', type=str, nargs='*', help='glob pattern of problem path')
    parser.add_argument('--problem-list', action='append', default=[],
                        help='input file with a list of problem paths')
    parser.add_argument('--problem-base-path', type=str,
                        help='the problem paths are relative to the base path')
    parser.add_argument('--output', '-o', type=str, help='main output directory')
    parser.add_argument('--output-batch', type=str, help='directory to store the batch results')
    parser.add_argument('--output-runs', type=str, help='directory to store the run results')
    # Output files: problems.txt, problems_successful.txt, runs.csv, result.json
    parser.add_argument('--vampire', type=str, default='vampire', help='Vampire command')
    parser.add_argument('--probe', action='store_true', help='probe each problem with a clausify run')
    parser.add_argument('--solve-runs', type=int, default=1,
                        help='Number of solving Vampire executions per problem. Useful namely with `--vampire_options_solve \"--symbol_precedence scramble\"`.')
    parser.add_argument('--timeout-probe', type=float, help='kill Vampire after this many seconds in probe runs')
    parser.add_argument('--timeout-solve', type=float, help='kill Vampire after this many seconds in solve runs')
    parser.add_argument('--jobs', '-j', type=int, default=1, help='number of jobs to run in parallel')

    vampire_options = parser.add_argument_group('Vampire options',
                                                'Options passed to Vampire. Run `vampire --show_options on --show_experimental_options on` to print the options supported by Vampire. Options automatically overriden: --random_seed, --json_output')
    vampire_options.add_argument('--vampire-options', action='append', default=[],
                                 help='Options for all runs. Recommended options: --include.')
    vampire_options.add_argument('--vampire-options-probe', action='append', default=[],
                                 help='Options for probe runs. Recommended options: --mode clausify, --time_limit.')
    vampire_options.add_argument('--vampire-options-solve', action='append', default=[],
                                 help='Options for solve runs. Recommended options: --time_limit, --symbol_precedence scramble.')
