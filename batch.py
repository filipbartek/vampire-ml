#!/usr/bin/env python3.7

import concurrent.futures
import json
import logging
import os
import subprocess
import time


class Batch:
    def __init__(self, vampire, vampire_options, output_path, jobs=1):
        self._vampire = vampire
        self._vampire_options = vampire_options
        assert output_path is not None
        self._output_path = output_path
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
        self._futures = set()

    def generate_results(self, problem_paths, runs_per_problem, problem_base_path=None):
        try:
            self.__solve_all_async(problem_paths, runs_per_problem, problem_base_path)
            for future in concurrent.futures.as_completed(self._futures):
                assert future.done()
                if future.cancelled():
                    continue
                yield future.result()
        except KeyboardInterrupt:
            for future in self._futures:
                future.cancel()
            raise
        finally:
            self._futures.clear()

    def __solve_all_async(self, problem_paths, runs_per_problem, problem_base_path):
        assert len(self._futures) == 0
        if len(problem_paths) == 0:
            return
        if problem_base_path is None:
            problem_base_path = os.path.commonpath(problem_paths)
        for problem_path in problem_paths:
            problem_output_path = self._output_path
            if problem_path != problem_base_path:
                problem_output_path = os.path.join(self._output_path, os.path.relpath(problem_path, problem_base_path))
            problem_vampire_options = self._vampire_options + [problem_path]
            self.__solve_one_problem_async(problem_output_path, problem_vampire_options, runs_per_problem, problem_path)

    def __solve_one_problem_async(self, output_path, vampire_options, runs_per_problem, problem_path):
        assert runs_per_problem >= 0
        if runs_per_problem == 0:
            return
        run_directory_name_width = len(str(runs_per_problem - 1))
        for problem_run_index in range(runs_per_problem):
            run_output_path = output_path
            run_vampire_options = vampire_options
            if runs_per_problem > 1:
                run_output_path = os.path.join(output_path, str(problem_run_index).zfill(run_directory_name_width))
                if '--random_seed' in run_vampire_options:
                    logging.warning('Overriding --random_seed.')
                run_vampire_options = run_vampire_options + ['--random_seed', str(problem_run_index + 1)]
            self.__solve_one_run_async(run_output_path, run_vampire_options, problem_path)

    def __solve_one_run_async(self, output_path, vampire_options, problem_path):
        self._futures.add(self._executor.submit(self.__solve_one_run_sync, output_path, vampire_options, problem_path))

    def __solve_one_run_sync(self, output_path, vampire_options, problem_path):
        json_output_path = os.path.join(output_path, 'vampire.json')
        if '--json_output' in vampire_options:
            logging.warning('Overriding --json_output.')
        vampire_options = vampire_options + ['--json_output', json_output_path]
        vampire_args = [self._vampire] + vampire_options
        complete_command = ' '.join(vampire_args)
        # If JSON output directory does not exist, Vampire fails.
        os.makedirs(output_path, exist_ok=True)
        time_start = time.time()
        cp = subprocess.run(vampire_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time_elapsed = time.time() - time_start
        result_thin = {
            'cwd': os.getcwd(),
            'command': complete_command,
            'exit_code': cp.returncode,
            'time_elapsed': time_elapsed,
            'problem_path': problem_path,
            'vampire': self._vampire,
            'paths': {
                'output_directory': output_path,
                'stdout': 'stdout.txt',
                'stderr': 'stderr.txt',
                'vampire_json': 'vampire.json'
            },
            'options': vampire_options
        }
        # TODO: Consider delegating the writing to the caller.
        with open(os.path.join(output_path, 'stdout.txt'), 'w') as stdout:
            stdout.write(cp.stdout)
        with open(os.path.join(output_path, 'stderr.txt'), 'w') as stderr:
            stderr.write(cp.stderr)
        with open(os.path.join(output_path, 'run.json'), 'w') as run_json:
            json.dump(result_thin, run_json, indent=4)
        result_complete = result_thin.copy()
        result_complete.update({
            'stdout': cp.stdout,
            'stderr': cp.stderr
        })
        result_complete['paths']['run_json'] = 'run.json'
        return result_complete
