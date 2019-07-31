#!/usr/bin/env python3.7

import collections
import json
import logging
import multiprocessing
import os


def get_updated(d, u):
    result = d.copy()
    for k, v in u.items():
        if k in result and isinstance(v, collections.Mapping):
            result[k] = get_updated(result[k], v)
        else:
            result[k] = v
    return result


class BatchSolver:
    def __init__(self, vampire, time_limit_probe, time_limit_solve):
        self.vampire = vampire
        self.time_limit_probe = time_limit_probe
        self.time_limit_solve = time_limit_solve

    @staticmethod
    def generate_tasks(problem_paths, parameters, outpath):
        for problem_index, problem_path in enumerate(problem_paths):
            problem_outpath = os.path.join(outpath, str(problem_index))
            problem_parameters = get_updated(parameters, {
                'problem_index': problem_index,
                'paths': {
                    'problem': problem_path
                }
            })
            yield (problem_parameters, problem_outpath, outpath)

    def solve_problems(self, problem_paths, parameters, outpath, jobs=1):
        results = []
        with multiprocessing.Pool(jobs) as p:
            try:
                os.makedirs(outpath, exist_ok=True)
                with open(os.path.join(outpath, 'prove_runs.txt'), 'w') as runs_file:
                    for result in p.imap_unordered(self.solve_problem_tuple,
                                                   self.generate_tasks(problem_paths, parameters, outpath)):
                        results.append(result)
                        if len(result['prove_runs']) > 0:
                            runs_file.write('%s\n' % '\n'.join(result['prove_runs']))
            finally:
                if outpath is not None:
                    os.makedirs(outpath, exist_ok=True)
                    with open(os.path.join(outpath, 'result.json'), 'w') as result_file:
                        json.dump(results, result_file, indent=4)

    def solve_problem_tuple(self, args):
        return self.solve_problem(*args)

    def solve_problem(self, parameters, outpath, base_path=None):
        result = {
            'outpath': os.path.relpath(outpath, start=base_path),
            'probe': None,
            'prove_runs': []
        }
        parameters_probe = get_updated(parameters, {
            'vampire': {
                'time_limit': self.time_limit_probe,
                'mode': 'clausify'
            }
        })
        probe_result = self.run_vampire_once(parameters_probe, os.path.join(outpath, 'probe'))
        result['probe'] = self.result_data_relpath(probe_result, base_path)
        if probe_result['output']['exit_code'] != 0:
            logging.warning('Probe run failed. Exit code: %s. Termination: %s'
                            % (probe_result['output']['exit_code'], probe_result['output']['data']['termination']))
            return result
        for solve_index in range(parameters['run_count']):
            parameters_run = get_updated(parameters, {
                'vampire': {
                    'time_limit': self.time_limit_solve,
                    'random_seed': solve_index + 1
                }
            })
            solve_result = self.run_vampire_once(parameters_run, os.path.join(outpath, str(solve_index)))
            result['prove_runs'].append(self.result_data_relpath(solve_result, base_path))
        return result

    def run_vampire_once(self, parameters, outpath):
        parameters_run = get_updated(parameters, {
            'vampire': {
                'json_output': os.path.join(outpath, 'out.json')
            },
            'paths': {
                'base': outpath,
                'stdout': 'stdout.txt',
                'stderr': 'stderr.txt',
                'json_output': 'out.json',
                'data': 'data.json'
            }
        })
        # The Vampire option --json_output requires the output directory to exist.
        os.makedirs(outpath, exist_ok=True)
        return self.vampire(parameters_run, outpath)

    @staticmethod
    def result_data_relpath(result, base_path):
        result_base_path = result['parameters']['paths']['base']
        result_data_path = result['parameters']['paths']['data']
        return os.path.relpath(os.path.join(result_base_path, result_data_path), start=base_path)
