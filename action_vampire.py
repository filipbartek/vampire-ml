#!/usr/bin/env python3.7

import glob
import itertools
import json
import logging
import os

from tqdm import tqdm

from batch import Batch
from lazy_csv_writer import LazyCsvWriter

default_vampire_options = {
    '--proof': 'off',
    '--time_limit': '10',
    '--statistics': 'full',
    '--time_statistics': 'on',
    '--encode': 'on'
}


def call(namespace):
    vampire_options = list(itertools.chain(*(s.split() for s in namespace.vampire_options)))
    for option, value in default_vampire_options.items():
        if option not in vampire_options:
            logging.info(f'Defaulting to `{option} {value}`.')
            vampire_options.extend([option, value])

    problem_paths = []
    for problem_list_path in namespace.problem_list:
        with open(problem_list_path, 'r') as problem_list_file:
            problem_paths.extend(l.rstrip('\n') for l in problem_list_file.readlines())
    problem_patterns = namespace.problem
    if namespace.problem_base_path is not None:
        problem_patterns = (os.path.join(namespace.problem_base_path, pattern) for pattern in problem_patterns)
    problem_paths.extend(itertools.chain(*(glob.iglob(pattern, recursive=True) for pattern in problem_patterns)))
    problem_base_path = namespace.problem_base_path
    if problem_base_path is None:
        problem_base_path = os.path.commonpath(problem_paths)
        logging.info(f'Defaulting problem base path to \"{problem_base_path}\".')

    output = namespace.output
    csv_file_path = os.path.join(output, 'runs.csv')
    problems_path = os.path.join(output, 'problems.txt')
    problems_successful_path = os.path.join(output, 'problems_successful.txt')

    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, 'configuration.json'), 'w') as output_json_file:
        json.dump({
            'vampire': namespace.vampire,
            'vampire_options': vampire_options,
            'runs_per_problem': namespace.runs,
            'jobs': namespace.jobs,
            'runs_csv': os.path.relpath(csv_file_path, output),
            'problem_base_path': problem_base_path,
            'problems': os.path.relpath(problems_path, output),
            'problems_successful': os.path.relpath(problems_successful_path, output)
        }, output_json_file, indent=4)
    with open(problems_path, 'w') as problems_file:
        problems_file.write('\n'.join(problem_paths))
        problems_file.write('\n')
    batch = Batch(namespace.vampire, vampire_options, output, namespace.jobs)
    problems_successful = set()
    with open(csv_file_path, 'w') as csv_file, open(problems_successful_path, 'w') as problems_successful_file:
        csv_writer = LazyCsvWriter(csv_file)
        for result in tqdm(batch.generate_results(problem_paths, namespace.runs, problem_base_path),
                           desc='Running Vampire', total=(len(problem_paths) * namespace.runs), unit='runs'):
            csv_writer.writerow({
                'output_path': os.path.relpath(result['paths']['output_directory'], namespace.output),
                'problem_path': os.path.relpath(result['problem_path'], problem_base_path),
                'exit_code': result['exit_code'],
                'time_elapsed': result['time_elapsed']
            })
            if result['exit_code'] == 0 and result['problem_path'] not in problems_successful:
                problems_successful_file.write(result['problem_path'] + '\n')
                problems_successful.add(result['problem_path'])


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
    parser.add_argument('--vampire_options', action='append', default=[],
                        help='Additional Vampire options. Recommended options: --include, --symbol_precedence. Automatically overriden options: --random_seed, --json_output.')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of Vampire executions per problem. Useful namely with `--vampire_options \"--symbol_precedence scramble\"`.')
    parser.add_argument('--jobs', '-j', type=int, default=1, help='number of jobs to run in parallel')
