#!/usr/bin/env python3.7

import concurrent.futures
import json
import logging
import os
import subprocess
import time

import utils


class Batch:
    # TODO: Add support for `--no-clobber`.
    # TODO: Add support for `--scratch`.
    def __init__(self, vampire, vampire_options, output_path, solve_runs_per_problem, strategy_id, vampire_timeout=None, cpus=1):
        self._vampire = vampire
        self._vampire_options = vampire_options
        assert output_path is not None
        self._output_path = output_path
        assert solve_runs_per_problem >= 1
        self._solve_runs_per_problem = solve_runs_per_problem
        self._strategy_id = strategy_id
        self._vampire_timeout = vampire_timeout
        assert cpus >= 1
        self._cpus = cpus
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=cpus)
        self._futures = set()

    def generate_results(self, problem_paths, problem_base_path=None):
        assert len(self._futures) == 0
        if len(problem_paths) == 0:
            logging.warning('No problems given.')
        if problem_base_path is None:
            problem_base_path = os.path.commonpath(problem_paths)
        try:
            for problem_path in problem_paths:
                problem_output_path = self._output_path
                # If we omitted this condition, then a '.' might be added to the output path.
                if problem_path != problem_base_path:
                    problem_output_path = os.path.join(self._output_path,
                                                       os.path.relpath(problem_path, problem_base_path))
                if self._strategy_id is not None:
                    problem_output_path = os.path.join(problem_output_path, self._strategy_id)
                self.__solve_one_problem_async(problem_output_path, problem_path)
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

    def __solve_one_problem_async(self, output_path, problem_path):
        assert self._solve_runs_per_problem >= 1
        run_directory_name_width = len(str(self._solve_runs_per_problem - 1))
        for problem_run_index in range(self._solve_runs_per_problem):
            run_output_path = output_path
            random_seed_zero_based = None
            if self._solve_runs_per_problem > 1:
                run_output_path = os.path.join(run_output_path, str(problem_run_index).zfill(run_directory_name_width))
                random_seed_zero_based = problem_run_index
            self.__solve_one_run_async(run_output_path, output_path, problem_path, random_seed_zero_based)

    def __solve_one_run_async(self, output_path, problem_output_path, problem_path, random_seed_zero_based=None):
        vampire_options = self.compose_vampire_options(problem_path, random_seed_zero_based)
        future = self._executor.submit(self.__solve_one_run_sync, output_path, problem_output_path, vampire_options,
                                       problem_path)
        self._futures.add(future)

    def compose_vampire_options(self, problem_path, random_seed_zero_based=None):
        vampire_options = self._vampire_options.copy()
        vampire_options.extend([problem_path])
        if random_seed_zero_based is not None:
            if '--random_seed' in vampire_options:
                logging.warning('Overriding --random_seed.')
            vampire_options.extend(['--random_seed', str(random_seed_zero_based + 1)])
        return vampire_options

    def __solve_one_run_sync(self, output_path, problem_output_path, vampire_options, problem_path):
        # TODO: Move to `compose_vampire_options`.
        # TODO: Add support for scratch space.
        json_output_path = os.path.join(output_path, 'vampire.json')
        if '--json_output' in vampire_options:
            logging.warning('Overriding --json_output.')
        vampire_options = vampire_options + ['--json_output', json_output_path]
        vampire_args = [self._vampire] + vampire_options
        complete_command = ' '.join(vampire_args)
        timeout_seconds = self._vampire_timeout
        if timeout_seconds is None:
            vampire_time_limit = utils.option_value(vampire_options, '--time_limit')
            if vampire_time_limit is not None:
                # TODO: Support all the time formats supported by Vampire.
                vampire_time_limit_seconds = float(vampire_time_limit)
                timeout_seconds = vampire_time_limit_seconds + 1
        # TODO: Add path to `job.json`.
        paths = {
            'cwd': os.getcwd(),
            'result': 'result.json',
            'problem': problem_path,
            'problem_output': problem_output_path,
            'output': output_path,
            'stdout': 'stdout.txt',
            'stderr': 'stderr.txt',
            'vampire_json': 'vampire.json',
            'configuration': 'configuration.json'
        }
        configuration = {
            'command': complete_command,
            'timeout_seconds': timeout_seconds,
            'vampire': self._vampire,
            'options': vampire_options
        }
        # If JSON output directory does not exist, Vampire fails.
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'configuration.json'), 'w') as configuration_json:
            json.dump({'paths': paths, 'configuration': configuration}, configuration_json, indent=4)
        time_start = time.time()
        try:
            cp = subprocess.run(vampire_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_seconds,
                                text=True)
            time_elapsed = time.time() - time_start
            result_timeout = False
            exit_status = cp.returncode
            stdout_str = cp.stdout
            stderr_str = cp.stderr
        except subprocess.TimeoutExpired as e:
            time_elapsed = time.time() - time_start
            result_timeout = True
            exit_status = None
            stdout_str = e.stdout
            stderr_str = e.stderr
        result = {
            'timeout_expired': result_timeout,
            'exit_code': exit_status,
            'time_elapsed': time_elapsed
        }
        with open(os.path.join(output_path, paths['result']), 'w') as run_json:
            json.dump({'paths': paths, 'result': result}, run_json, indent=4)
        with open(os.path.join(output_path, paths['stdout']), 'w') as stdout_file:
            stdout_file.write(stdout_str)
        with open(os.path.join(output_path, paths['stderr']), 'w') as stderr_file:
            stderr_file.write(stderr_str)
        # The JSON files may take a lot of space.
        if result_timeout or exit_status != 0:
            try:
                os.remove(json_output_path)
            except FileNotFoundError:
                pass
        return {
            'paths': paths,
            'configuration': configuration,
            'result': result,
            'stdout': stdout_str,
            'stderr': stderr_str
        }
