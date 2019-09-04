#!/usr/bin/env python3.7

import collections
import itertools
import logging

from tqdm import tqdm

import file_path_list
from batch import Batch

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


def call(namespace):
    vampire_options_probe = compose_vampire_options(namespace.vampire_options, namespace.vampire_options_probe, 'probe')
    vampire_options_solve = compose_vampire_options(namespace.vampire_options, namespace.vampire_options_solve, 'solve')
    problem_paths, problem_base_path = file_path_list.compose(namespace.problem_list, namespace.problem,
                                                              namespace.problem_base_path)
    batch = Batch(namespace.vampire, vampire_options_probe, vampire_options_solve, namespace.output,
                  namespace.solve_runs, namespace.jobs)
    with tqdm(desc='Running Vampire', total=len(problem_paths), unit='problems') as t:
        solve_runs = collections.Counter()
        stats = {
            'probe': {
                'pass': 0,
                'fail': 0,
                'processed': 0,
                'expected': len(problem_paths)
            },
            'solve': {
                'pass': 0,
                'fail': 0,
                'skip': 0,
                'processed': 0,
                'expected': len(problem_paths) * namespace.solve_runs,
            }
        }
        t.set_postfix(stats)
        for result in batch.generate_results(problem_paths, namespace.probe, problem_base_path):
            if result['probe']:
                stats['probe']['processed'] += 1
                if result['exit_code'] == 0:
                    stats['probe']['pass'] += 1
                    if namespace.solve_runs == 0:
                        t.update(1)
                else:
                    stats['probe']['fail'] += 1
                    stats['solve']['skip'] += namespace.solve_runs
                    stats['solve']['processed'] += namespace.solve_runs
                    t.update(1)
            else:
                solve_runs[result['paths']['problem']] += 1
                stats['solve']['processed'] += 1
                if solve_runs[result['paths']['problem']] == namespace.solve_runs:
                    t.update(1)
                if result['exit_code'] == 0:
                    stats['solve']['pass'] += 1
                else:
                    stats['solve']['fail'] += 1
            t.set_postfix(stats)


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('problem', type=str, nargs='*', help='glob pattern of problem path')
    parser.add_argument('--problem_list', action='append', default=[],
                        help='input file with a list of problem paths')
    parser.add_argument('--problem_base_path', type=str,
                        help='the problem paths are relative to the base path')
    parser.add_argument('--output', '-o', required=True, type=str, help='path to store the output files')
    # Output files: problems.txt, problems_successful.txt, runs.csv, result.json
    parser.add_argument('--vampire', type=str, default='vampire', help='Vampire command')
    parser.add_argument('--probe', action='store_true', help='probe each problem with a clausify run')
    parser.add_argument('--solve_runs', type=int, default=1,
                        help='Number of solving Vampire executions per problem. Useful namely with `--vampire_options_solve \"--symbol_precedence scramble\"`.')
    parser.add_argument('--jobs', '-j', type=int, default=1, help='number of jobs to run in parallel')

    vampire_options = parser.add_argument_group('Vampire options',
                                                'Options passed to Vampire. Run `vampire --show_options on --show_experimental_options on` to print the options supported by Vampire. Options automatically overriden: --random_seed, --json_output')
    vampire_options.add_argument('--vampire_options', action='append', default=[],
                                 help='Options for all runs. Recommended options: --include.')
    vampire_options.add_argument('--vampire_options_probe', action='append', default=[],
                                 help='Options for probe runs. Recommended options: --mode clausify, --time_limit.')
    vampire_options.add_argument('--vampire_options_solve', action='append', default=[],
                                 help='Options for solve runs. Recommended options: --time_limit, --symbol_precedence scramble.')
