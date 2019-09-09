#!/usr/bin/env python3.7

import concurrent.futures
import json
import logging
import os
import subprocess
import time

import utils


class Batch:
    def __init__(self, vampire, vampire_options_probe, vampire_options_solve, output_path, solve_runs_per_problem,
                 timeout_probe=None, timeout_solve=None, jobs=1):
        self._vampire = vampire
        self._vampire_options_probe = vampire_options_probe
        self._vampire_options_solve = vampire_options_solve
        assert output_path is not None
        self._output_path = output_path
        self._solve_runs_per_problem = solve_runs_per_problem
        self._timeout_probe = timeout_probe
        self._timeout_solve = timeout_solve
        assert jobs >= 1
        self._jobs = jobs
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
        self._futures = set()

    def generate_results(self, problem_paths, probe, problem_base_path=None):
        assert len(self._futures) == 0
        if len(problem_paths) == 0:
            logging.warning('No problems given.')
        if problem_base_path is None:
            problem_base_path = os.path.commonpath(problem_paths)
        try:
            for problem_path in problem_paths:
                assert self._jobs >= 1
                for result in self.reduce_futures(self._jobs - 1):
                    yield result
                problem_output_path = self._output_path
                # TODO: Can this condition be omitted?
                if problem_path != problem_base_path:
                    problem_output_path = os.path.join(self._output_path,
                                                       os.path.relpath(problem_path, problem_base_path))
                self.__solve_one_problem_async(problem_output_path, problem_path, probe)
            for result in self.reduce_futures():
                yield result
        except KeyboardInterrupt:
            for future in self._futures:
                future.cancel()
            raise
        finally:
            self._futures.clear()

    def reduce_futures(self, n=0):
        while len(self._futures) > n:
            done, _ = concurrent.futures.wait(list(self._futures), return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                self._futures.remove(future)
                assert future.done()
                if future.cancelled():
                    continue
                yield future.result()

    def __solve_one_problem_async(self, output_path, problem_path, probe):
        assert self._solve_runs_per_problem >= 0
        if probe:
            self.__solve_one_run_async(os.path.join(output_path, 'probe'), output_path, problem_path, probe)
            return
        if self._solve_runs_per_problem == 0:
            return
        run_directory_name_width = len(str(self._solve_runs_per_problem - 1))
        for problem_run_index in range(self._solve_runs_per_problem):
            run_output_path = os.path.join(output_path, str(problem_run_index).zfill(run_directory_name_width))
            self.__solve_one_run_async(run_output_path, output_path, problem_path, probe, problem_run_index)

    def __solve_one_run_async(self, output_path, problem_output_path, problem_path, probe, problem_run_index=None):
        vampire_options = self.compose_vampire_options(problem_path, probe, problem_run_index)
        future = self._executor.submit(self.__solve_one_run_sync, output_path, problem_output_path, vampire_options,
                                       problem_path, probe)
        future.probe = probe
        if probe:
            future.add_done_callback(self.probe_done_callback)
        self._futures.add(future)

    def compose_vampire_options(self, problem_path, probe, random_seed_zero_based=None):
        vampire_options = self._vampire_options_solve
        if probe:
            vampire_options = self._vampire_options_probe
        vampire_options = vampire_options + [problem_path]
        if random_seed_zero_based is not None:
            if '--random_seed' in vampire_options:
                logging.warning('Overriding --random_seed.')
            vampire_options = vampire_options + ['--random_seed', str(random_seed_zero_based + 1)]
        return vampire_options

    def __solve_one_run_sync(self, output_path, problem_output_path, vampire_options, problem_path, probe):
        json_output_path = os.path.join(output_path, 'vampire.json')
        if '--json_output' in vampire_options:
            logging.warning('Overriding --json_output.')
        vampire_options = vampire_options + ['--json_output', json_output_path]
        vampire_args = [self._vampire] + vampire_options
        complete_command = ' '.join(vampire_args)
        if probe:
            timeout_seconds = self._timeout_probe
        else:
            timeout_seconds = self._timeout_solve
        if timeout_seconds is None:
            vampire_time_limit = utils.option_value(vampire_options, '--time_limit')
            if vampire_time_limit is not None:
                # TODO: Support all the time formats supported by Vampire.
                vampire_time_limit_seconds = int(vampire_time_limit)
                timeout_seconds = vampire_time_limit_seconds + 1
        exit_status = None
        # If JSON output directory does not exist, Vampire fails.
        os.makedirs(output_path, exist_ok=True)
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
            stdout_str = e.stdout
            stderr_str = e.stderr
        # The JSON files may take a lot of space.
        if result_timeout or exit_status != 0:
            try:
                os.remove(json_output_path)
            except FileNotFoundError:
                pass
        result_thin = {
            'cwd': os.getcwd(),
            'command': complete_command,
            'timeout': result_timeout,
            'exit_code': exit_status,
            'time_elapsed': time_elapsed,
            'vampire': self._vampire,
            'probe': probe,
            'solve_runs': self._solve_runs_per_problem,
            'paths': {
                'problem': problem_path,
                'problem_output': problem_output_path,
                'output': output_path,
                'stdout': 'stdout.txt',
                'stderr': 'stderr.txt',
                'vampire_json': 'vampire.json'
            },
            'options': vampire_options
        }
        result_complete = result_thin.copy()
        result_complete.update({
            'stdout': stdout_str,
            'stderr': stderr_str
        })
        result_complete['paths']['run_json'] = 'run.json'
        # TODO: Consider delegating the writing to the caller.
        with open(os.path.join(output_path, result_thin['paths']['stdout']), 'w') as stdout_file:
            stdout_file.write(stdout_str)
        with open(os.path.join(output_path, result_thin['paths']['stderr']), 'w') as stderr_file:
            stderr_file.write(stderr_str)
        with open(os.path.join(output_path, result_complete['paths']['run_json']), 'w') as run_json:
            json.dump(result_thin, run_json, indent=4)
        return result_complete

    def probe_done_callback(self, future):
        assert future.probe
        assert future.done()
        if future.cancelled():
            return
        result = future.result()
        assert result['probe']
        if result['exit_code'] == 0:
            # Schedule the solve runs.
            self.__solve_one_problem_async(result['paths']['problem_output'], result['paths']['problem'], False)
