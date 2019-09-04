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


def call(namespace):
    vampire_options_probe = compose_vampire_options(namespace.vampire_options, namespace.vampire_options_probe, 'probe')
    vampire_options_solve = compose_vampire_options(namespace.vampire_options, namespace.vampire_options_solve, 'solve')

    problem_paths, problem_base_path = file_path_list.compose(namespace.problem_list, namespace.problem,
                                                              namespace.problem_base_path)

    output = namespace.output
    csv_file_path = os.path.join(output, 'runs.csv')
    problems_path = os.path.join(output, 'problems.txt')
    problems_successful_path = os.path.join(output, 'problems_successful.txt')

    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, 'configuration.json'), 'w') as output_json_file:
        json.dump({
            'vampire': namespace.vampire,
            'vampire_options_probe': vampire_options_probe,
            'vampire_options_solve': vampire_options_solve,
            'runs_per_problem': namespace.solve_runs,
            'jobs': namespace.jobs,
            'runs_csv': os.path.relpath(csv_file_path, output),
            'problem_base_path': problem_base_path,
            'problems': os.path.relpath(problems_path, output),
            'problems_successful': os.path.relpath(problems_successful_path, output)
        }, output_json_file, indent=4)
    with open(problems_path, 'w') as problems_file:
        problems_file.write('\n'.join(problem_paths))
        problems_file.write('\n')
    batch = Batch(namespace.vampire, vampire_options_probe, vampire_options_solve, output, namespace.solve_runs,
                  namespace.jobs)
    problems_successful = set()
    with open(csv_file_path, 'w') as csv_file, open(problems_successful_path, 'w') as problems_successful_file:
        csv_writer = LazyCsvWriter(csv_file)
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
                csv_writer.writerow({
                    'output_path': os.path.relpath(result['paths']['output'], namespace.output),
                    'problem_path': os.path.relpath(result['paths']['problem'], problem_base_path),
                    'exit_code': result['exit_code'],
                    'time_elapsed': result['time_elapsed']
                })
                if result['exit_code'] == 0 and result['paths']['problem'] not in problems_successful:
                    problems_successful_file.write(result['paths']['problem'] + '\n')
                    problems_successful.add(result['paths']['problem'])
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
