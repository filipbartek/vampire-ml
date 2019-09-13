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
    '--encode': 'on',
    '--time_limit': '10',
    '--statistics': 'full',
    '--time_statistics': 'on',
    '--proof': 'off',
    '--symbol_precedence': 'scramble'
}


def compose_vampire_options(custom_options):
    vampire_options = list(itertools.chain(*(s.split() for s in custom_options)))
    for option, value in default_vampire_options.items():
        if option not in vampire_options:
            logging.info(f'Defaulting Vampire option: {option} {value}')
            vampire_options.extend([option, value])
    return vampire_options


def call(namespace):
    vampire_options = compose_vampire_options(namespace.vampire_options)
    problem_paths, problem_base_path = file_path_list.compose(namespace.problem_list, namespace.problem,
                                                              namespace.problem_base_path)
    assert namespace.output is not None
    output_job = os.path.join(namespace.output, 'jobs')
    output_problems = os.path.join(namespace.output, 'problems')
    if namespace.strategy_id is not None:
        output_job = os.path.join(output_job, namespace.strategy_id)
    if namespace.job_id is not None:
        output_job = os.path.join(output_job, namespace.job_id)
    csv_file_path = os.path.join(output_job, 'runs.csv')
    # TODO: Do not save any problems list. runs.csv should suffice.
    problems_path = os.path.join(output_job, 'problems.txt')
    problems_successful_path = os.path.join(output_job, 'problems_successful.txt')

    # We assume that `os.makedirs` is thread-safe.
    os.makedirs(output_job, exist_ok=True)
    with open(os.path.join(output_job, 'job.json'), 'w') as output_json_file:
        # TODO: Improve output property names.
        json.dump({
            'output': os.path.relpath(namespace.output, output_job),
            'run_output_base_path': os.path.relpath(output_problems, output_job),
            'runs_csv': os.path.relpath(csv_file_path, output_job),
            'problem_base_path': problem_base_path,
            'problems': os.path.relpath(problems_path, output_job),
            'problems_successful': os.path.relpath(problems_successful_path, output_job),
            'solve_runs_per_problem': namespace.solve_runs,
            'cpus': namespace.cpus,
            'strategy_id': namespace.strategy_id,
            'job_id': namespace.job_id,
            'scratch': namespace.scratch,
            'cwd': os.getcwd(),
            'no_clobber': namespace.no_clobber,
            'vampire_timeout': namespace.vampire_timeout,
            'vampire': namespace.vampire,
            'vampire_options': vampire_options
        }, output_json_file, indent=4)
    with open(problems_path, 'w') as problems_file:
        problems_file.write('\n'.join(problem_paths))
        problems_file.write('\n')
    assert namespace.solve_runs >= 1
    batch = Batch(namespace.vampire, vampire_options, output_problems, namespace.solve_runs, namespace.strategy_id,
                  namespace.vampire_timeout, namespace.cpus, namespace.no_clobber, namespace.scratch,
                  os.path.join(output_job, 'job.json'))
    problems_successful = set()
    with open(csv_file_path, 'w') as csv_file, open(problems_successful_path, 'w') as problems_successful_file:
        csv_writer = LazyCsvWriter(csv_file)
        with tqdm(desc='Running Vampire', total=len(problem_paths) * namespace.solve_runs, unit='problems') as t:
            stats = collections.Counter()
            t.set_postfix_str(stats)
            for run_info in batch.generate_results(problem_paths, problem_base_path, csv_writer):
                if run_info['result']['exit_code'] == 0 and run_info['paths']['problem'] not in problems_successful:
                    problems_successful_file.write(run_info['paths']['problem'] + '\n')
                    problems_successful.add(run_info['paths']['problem'])
                stats[(run_info['result']['status'], run_info['result']['exit_code'])] += 1
                t.set_postfix_str(stats)
                t.update()


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('problem', type=str, nargs='*', help='glob pattern of a problem path')
    parser.add_argument('--problem-list', action='append', default=[],
                        help='input file with a list of problem paths')
    parser.add_argument('--problem-base-path', type=str,
                        help='the problem paths are relative to the base path')
    # Naming convention: `sbatch --output`
    parser.add_argument('--output', '-o', required=True, type=str, help='main output directory')
    parser.add_argument('--strategy-id',
                        help='Identifier of strategy. Disambiguates job and problem run output directories.')
    parser.add_argument('--job-id', help='Identifier of job. Disambiguates job output directory.')
    # Naming convention: `wget --no-clobber`
    parser.add_argument('--no-clobber', '-nc', action='store_true',
                        help='skip runs that would overwrite existing files')
    parser.add_argument('--scratch', help='temporary output directory')
    # Naming convention: `sbatch --cpus-per-task`
    parser.add_argument('--cpus', '-c', type=int, default=1, help='number of jobs to run in parallel')
    # TODO: Expose finer control of the Vampire option `--random_seed`.
    parser.add_argument('--solve-runs', type=int, default=1,
                        help='Number of Vampire executions per problem. '
                             'Each of the executions uses a different value of the Vampire option `--random_seed`. '
                             'Useful namely with `--vampire_options \"--symbol_precedence scramble\"`.')
    parser.add_argument('--vampire', type=str, default='vampire', help='Vampire command')
    parser.add_argument('--vampire-options', action='append', default=[],
                        help='Options passed to Vampire. '
                             'Run `vampire --show_options on --show_experimental_options on` to print the options '
                             'supported by Vampire. '
                             'Options automatically overridden: --random_seed, --json_output. '
                             'Recommended options: --time_limit, --mode, --symbol_precedence, --include.')
    parser.add_argument('--vampire-timeout', type=float, help='kill Vampire after this many seconds')
