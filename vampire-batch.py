#!/usr/bin/env python3.7

import argparse
import glob
import itertools
import os
import sys

import solver


def parse_args():
    parser = argparse.ArgumentParser()

    # Batch arguments
    parser.add_argument('problem', type=str, nargs='+', help='glob patten of problem path')
    parser.add_argument('--output_path', type=str, help='path to store the output files')
    parser.add_argument('--output_stdout', type=str, help='filename of the output stdout file')
    parser.add_argument('--output_stderr', type=str, help='filename of the output stderr file')
    parser.add_argument('--vampire', type=str, default='vampire', help='Vampire command')
    parser.add_argument('--runs', type=int, default=1, help='number of Vampire executions per problem')
    parser.add_argument('--jobs', '-j', type=int, default=1, help='number of jobs to run in parallel')

    # Vampire arguments
    parser.add_argument('--vampire_include', type=str, help='path to TPTP directory')
    parser.add_argument('--vampire_time_limit_solve', type=str, default='10',
                        help='time limit for Vampire problem solving runs')
    parser.add_argument('--vampire_time_limit_probe', type=str, default='1',
                        help='time limit for Vampire parse probing runs')
    parser.add_argument('--vampire_proof', type=str, choices=['off', 'on'], default='off',
                        help='should Vampire print the proof?')
    parser.add_argument('--vampire_symbol_precedence', type=str,
                        choices=['arity', 'occurrence', 'reverse_arity', 'scramble', 'frequency', 'reverse_frequency',
                                 'weighted_frequency', 'reverse_weighted_frequency'],
                        default='scramble', help='symbol precedence')

    return parser.parse_args()


if __name__ == '__main__':
    namespace = parse_args()

    # TODO: Allow saving and loading the list of problems.

    vampire_args_common = [namespace.vampire]
    if namespace.vampire_include is not None:
        vampire_args_common.extend(['--include', namespace.vampire_include])
    vampire_args_common.extend(['--encode', 'on'])
    vampire_args_common.extend(['--symbol_precedence', namespace.vampire_symbol_precedence])
    vampire_args_common.extend(['--proof', namespace.vampire_proof])
    vampire_args_common.extend(['--statistics', 'full'])

    results = {
        'args': sys.argv,
        'namespace': {
            'problem': namespace.problem,
            'vampire': namespace.vampire,
            'vampire_include': namespace.vampire_include,
            'vampire_symbol_precedence': namespace.vampire_symbol_precedence,
            'vampire_proof': namespace.vampire_proof,
            'runs': namespace.runs,
            'output_path': namespace.output_path
        },
        'problems': []
    }

    if namespace.output_path is not None:
        os.makedirs(namespace.output_path, exist_ok=True)

    problem_paths = itertools.chain(*map(lambda pattern: glob.iglob(pattern, recursive=True), namespace.problem))

    s = solver.Solver(namespace)
    s.solve_problems(vampire_args_common, problem_paths, results)
