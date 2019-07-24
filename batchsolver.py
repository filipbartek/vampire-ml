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
            yield (problem_parameters, problem_outpath)

    def solve_problems(self, problem_paths, parameters, outpath, jobs=1):
        results = []
        with multiprocessing.Pool(jobs) as p:
            try:
                for result in p.imap_unordered(self.solve_problem_tuple,
                                               self.generate_tasks(problem_paths, parameters, outpath)):
                    results.append(result)
                # TODO: Save a CSV continuously.
            finally:
                if outpath is not None:
                    with open(os.path.join(outpath, 'result.json'), 'w') as result_file:
                        json.dump(results, result_file, indent=4)

    def solve_problem(self, parameters, outpath):
        results = []
        probe_result = self.run_vampire_once(parameters, self.time_limit_probe, None, os.path.join(outpath, 'probe'))
        results.append(probe_result['parameters']['paths']['data'])
        probe_result_termination = probe_result['output']['data']['termination']
        if probe_result_termination['phase'] == 'Parsing':
            logging.warning(
                'Probe run finished in parsing phase. Termination reason: %s' % probe_result_termination['reason'])
            return results
        for solve_index in range(parameters['run_count']):
            solve_result = self.run_vampire_once(parameters, self.time_limit_solve, solve_index + 1,
                                                 os.path.join(outpath, str(solve_index)))
            results.append(solve_result['parameters']['paths']['data'])
        return results

    def run_vampire_once(self, parameters, time_limit, random_seed, outpath):
        parameters_run = get_updated(parameters, {
            'vampire': {
                'time_limit': time_limit
            },
            'paths': {
                'stdout': os.path.join(outpath, 'stdout.txt'),
                'stderr': os.path.join(outpath, 'stderr.txt'),
                'data': os.path.join(outpath, 'data.json')
            }
        })
        if random_seed is not None:
            parameters_run['vampire']['random_seed'] = random_seed
        return self.vampire(parameters_run)

    def solve_problem_tuple(self, args):
        return self.solve_problem(*args)
