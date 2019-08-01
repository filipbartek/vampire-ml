#!/usr/bin/env python3.7

import collections
import concurrent.futures
import json
import logging
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

    def solve_problems(self, problem_paths, parameters, outpath, jobs=1):
        results = dict()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = set()
                for problem_index, problem_path in enumerate(problem_paths):
                    self.reduce_futures(futures, results, outpath, jobs)
                    problem_outpath = os.path.join(outpath, str(problem_index))
                    problem_parameters = get_updated(parameters, {
                        'problem_index': problem_index,
                        'paths': {
                            'problem': problem_path
                        }
                    })
                    probe_parameters = get_updated(problem_parameters, {
                        'probe': True,
                        'vampire': {
                            'time_limit': self.time_limit_probe,
                            'mode': 'clausify'
                        }
                    })
                    assert probe_parameters['vampire']['time_limit'] == self.time_limit_probe
                    prove_parameters = get_updated(problem_parameters, {
                        'probe': False,
                        'vampire': {
                            'time_limit': self.time_limit_solve
                        }
                    })
                    assert probe_parameters['vampire']['time_limit'] == self.time_limit_probe
                    futures.add(executor.submit(self.run_probe, problem_outpath, probe_parameters, prove_parameters,
                                                parameters['run_count'], executor, futures))
                self.reduce_futures(futures, results, outpath)
        finally:
            if outpath is not None:
                os.makedirs(outpath, exist_ok=True)
                with open(os.path.join(outpath, 'result.json'), 'w') as result_file:
                    json.dump(results, result_file, indent=4)
        return results

    @staticmethod
    def reduce_futures(futures, results, base_path, n=0):
        while len(futures) > n:
            done, _ = concurrent.futures.wait(list(futures), return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                assert future.done()
                result = future.result()
                problem_index = result['problem_path']
                data_path = os.path.relpath(os.path.join(result['output_path']), start=base_path)
                if problem_index not in results:
                    results[problem_index] = {
                        'prove_runs': []
                    }
                if 'mode' in result['vampire_parameters'] and result['vampire_parameters']['mode'] == 'clausify':
                    results[problem_index]['probe'] = data_path
                else:
                    results[problem_index]['prove_runs'].append(data_path)
                futures.remove(future)

    def run_probe(self, outpath, parameters_probe, parameters_prove, n_prove_runs, executor, futures):
        result = self.run_vampire_once(parameters_probe, outpath)
        if result['call']['exit_code'] != 0:
            logging.warning('Probe run failed. Exit code: %s. Termination: %s'
                            % (result['call']['exit_code'], result['output']['data']['termination']))
        else:
            for prove_run_index in range(n_prove_runs):
                futures.add(executor.submit(self.run_prove, os.path.join(outpath, str(prove_run_index)),
                                            get_updated(parameters_prove, {
                                                'prove_run_index': prove_run_index,
                                                'vampire': {
                                                    'random_seed': prove_run_index + 1
                                                }
                                            })))
        return result

    def run_prove(self, outpath, parameters):
        return self.run_vampire_once(parameters, outpath)

    def run_vampire_once(self, parameters, outpath):
        parameters_run = get_updated(parameters['vampire'], {
            'json_output': os.path.join(outpath, 'out.json')
        })
        # The Vampire option --json_output requires the output directory to exist.
        os.makedirs(outpath, exist_ok=True)
        return self.vampire(parameters_run, parameters['paths']['problem'], outpath)
