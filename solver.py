#!/usr/bin/env python3.7

import json
import logging
import multiprocessing
import os
import subprocess

import extractor


class Solver:
    def __init__(self, namespace):
        self.namespace = namespace

    def save_output(self, cp, problem_index, run_index):
        result = {
            'stdout': None,
            'stderr': None
        }
        if self.namespace.output_path is not None:
            if self.namespace.output_stdout is not None:
                with open(
                        os.path.join(self.namespace.output_path,
                                     f'{problem_index:03} {run_index:03}_{self.namespace.output_stdout}'),
                        'w') as output_stdout:
                    output_stdout.write(cp.stdout)
                    result['stdout'] = output_stdout.name
            if self.namespace.output_stderr is not None:
                with open(
                        os.path.join(self.namespace.output_path,
                                     f'{problem_index:03} {run_index:03}_{self.namespace.output_stderr}'),
                        'w') as output_stderr:
                    output_stderr.write(cp.stderr)
                    result['stderr'] = output_stderr.name
        return result

    def solve_run(self, vampire_args, problem_index, run_index):
        vampire_command = ' '.join(vampire_args)
        print(f'  {problem_index:03} {run_index:03} {vampire_command}')
        cp = subprocess.run(vampire_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = {
            'problem_index': problem_index,
            'run_index': run_index,
            'vampire_command': vampire_command,
            'exit_code': cp.returncode
        }
        result.update(self.save_output(cp, problem_index, run_index))
        result.update(extractor.complete(cp.stdout))
        return result

    def solve_problem(self, vampire_args, problem_index):
        vampire_args_probe = vampire_args + ['--time_limit', self.namespace.vampire_time_limit_probe]
        cp = subprocess.run(vampire_args_probe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        probe_result = extractor.probe(cp.stdout)
        result = {
            'problem_index': problem_index,
            'probe': {
                'vampire_command': ' '.join(vampire_args_probe),
                'output': probe_result
            },
            'runs': []
        }
        if probe_result['termination']['phase'] == 'Parsing':
            logging.warning(f'Probe run finished in parsing phase. Termination reason: {probe_result["termination"]["reason"]}')
            return result
        for run_index in range(self.namespace.runs):
            vampire_args_run = vampire_args + ['--time_limit', self.namespace.vampire_time_limit_solve,
                                               '--random_seed', str(run_index + 1)]
            run_result = self.solve_run(vampire_args_run, problem_index, run_index)
            if run_result['function_names'] != probe_result['function_names']:
                logging.warning('Function names mismatch')
            else:
                del run_result['function_names']
            if run_result['predicate_names'] != probe_result['predicate_names']:
                logging.warning('Predicate names mismatch')
            else:
                del run_result['predicate_names']
            result['runs'].append(run_result)
        return result

    def solve_problem_tuple(self, t):
        return self.solve_problem(t[1], t[0])

    def solve_problems(self, vampire_args, problem_paths, results):
        if 'problems' not in results:
            results['problems'] = []
        vampire_args_problem_generator = (vampire_args + [problem_path] for problem_path in problem_paths)
        with multiprocessing.Pool(self.namespace.jobs) as p:
            try:
                for result in p.imap_unordered(self.solve_problem_tuple, enumerate(vampire_args_problem_generator)):
                    results['problems'].append(result)
            finally:
                if self.namespace.output_path is not None:
                    with open(os.path.join(self.namespace.output_path, 'result.json'), 'w') as result_file:
                        json.dump(results, result_file, indent=4)
        return results
