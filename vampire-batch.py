#!/usr/bin/env python3.7

import argparse
import glob
import itertools
import logging

import batchsolver
import vampire


def parse_args():
    parser = argparse.ArgumentParser()

    # Batch arguments
    parser.add_argument('problem', type=str, nargs='+', help='glob pattern of problem path')
    parser.add_argument('--output', '-o', type=str, help='path to store the output files')
    parser.add_argument('--vampire', type=str, default='vampire', help='Vampire command')
    parser.add_argument('--runs', type=int, default=1, help='number of Vampire executions per problem')
    parser.add_argument('--jobs', '-j', type=int, default=1, help='number of jobs to run in parallel')

    # Vampire arguments
    parser.add_argument('--vampire_include', type=str, help='path to TPTP directory')
    parser.add_argument('--vampire_time_limit_solve', type=float, default=10,
                        help='time limit for Vampire problem solving runs in wall clock seconds')
    parser.add_argument('--vampire_time_limit_probe', type=float, default=1,
                        help='time limit for Vampire parse probing runs in wall clock seconds')
    parser.add_argument('--vampire_proof', type=str, choices=['off', 'on'], default='off',
                        help='should Vampire print the proof?')
    parser.add_argument('--vampire_symbol_precedence', type=str,
                        choices=['arity', 'occurrence', 'reverse_arity', 'scramble', 'frequency', 'reverse_frequency',
                                 'weighted_frequency', 'reverse_weighted_frequency'],
                        default='scramble', help='symbol precedence')

    return parser.parse_args()


# TODO: Store results in a CSV table.
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    namespace = parse_args()

    # TODO: Allow saving and loading the list of problems.

    bs = batchsolver.BatchSolver(vampire.Vampire(namespace.vampire), namespace.vampire_time_limit_probe,
                                 namespace.vampire_time_limit_solve)
    problem_paths = itertools.chain(*map(lambda pattern: glob.iglob(pattern, recursive=True), namespace.problem))
    parameters = {
        'vampire': {
            'include': namespace.vampire_include,
            'symbol_precedence': namespace.vampire_symbol_precedence,
            'proof': namespace.vampire_proof,
            'encode': 'on',
            'statistics': 'full'
        },
        'run_count': namespace.runs
    }
    bs.solve_problems(problem_paths, parameters, namespace.output, namespace.jobs)
