#!/usr/bin/env python3.7

import concurrent.futures
import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time

import utils


class Batch:
    def __init__(self, vampire, vampire_options, output_path, solve_runs_per_problem, strategy_id, vampire_timeout=None,
                 cpus=1, overwrite='none', scratch=None, job_file_path=None):
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
        self._overwrite = overwrite
        self._scratch = scratch
        self._job_file_path = job_file_path

    def generate_results(self, problem_paths, problem_base_path=None, csv_writer=None):
        # TODO: Yield instances of Run instead of dicts.
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
                self.__solve_one_problem_async(problem_output_path, problem_path, problem_base_path, csv_writer)
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

    def __solve_one_problem_async(self, output_path, problem_path, problem_base_path, csv_writer):
        assert self._solve_runs_per_problem >= 1
        run_directory_name_width = len(str(self._solve_runs_per_problem - 1))
        for problem_run_index in range(self._solve_runs_per_problem):
            run_output_path = output_path
            random_seed_zero_based = None
            if self._solve_runs_per_problem > 1:
                run_output_path = os.path.join(run_output_path, str(problem_run_index).zfill(run_directory_name_width))
                random_seed_zero_based = problem_run_index
            self.__solve_one_run_async(run_output_path, output_path, problem_path, problem_base_path, csv_writer,
                                       random_seed_zero_based)

    def __solve_one_run_async(self, output_path, problem_output_path, problem_path, problem_base_path, csv_writer,
                              random_seed_zero_based=None):
        future = self._executor.submit(self.__solve_one_run_sync, output_path, problem_output_path, problem_path,
                                       problem_base_path, csv_writer, random_seed_zero_based)
        self._futures.add(future)

    def compose_vampire_options(self, problem_path, json_output_dir, random_seed_zero_based=None):
        vampire_options = self._vampire_options.copy()
        vampire_options.extend([problem_path])
        if random_seed_zero_based is not None:
            if '--random_seed' in vampire_options:
                logging.warning('Overriding --random_seed.')
            vampire_options.extend(['--random_seed', str(random_seed_zero_based + 1)])
        assert json_output_dir is not None
        json_output_path = os.path.join(json_output_dir, 'vampire.json')
        if '--json_output' in vampire_options:
            logging.warning('Overriding --json_output.')
        vampire_options.extend(['--json_output', json_output_path])
        return vampire_options

    def __solve_one_run_sync(self, output_path, problem_output_path, problem_path, problem_base_path, csv_writer,
                             random_seed_zero_based):
        scratch_root = None
        if self._scratch is not None:
            scratch_root = os.path.abspath(self._scratch)
        paths = {
            'configuration': 'configuration.json',
            'result': 'result.json',
            'stdout': 'stdout.txt',
            'stderr': 'stderr.txt',
            'vampire_json': 'vampire.json',
            'job': os.path.relpath(self._job_file_path, output_path),
            'problem': problem_path,
            'scratch_root': scratch_root,
            'scratch_job': None,
            'cwd': os.getcwd(),
            'output_problem': os.path.abspath(problem_output_path),
            'output_run': os.path.abspath(output_path)
        }
        configuration = {
            'overwrite': self._overwrite,
            'vampire': self._vampire
        }
        result = {
            'status': None,
            'exit_code': None,
            'time_elapsed': None
        }
        stdout_str = None
        stderr_str = None
        # Supported values of `self._overwrite`:
        # * all: Always overwrite
        # * failed: Overwrite all runs with non-zero exit codes
        # * interrupted: Overwrite all interrupted run results
        # * none: Never overwrite
        assert self._overwrite in ['all', 'failed', 'interrupted', 'none']
        if self._overwrite != 'all' and os.path.isfile(os.path.join(output_path, paths['result'])):
            with open(os.path.join(output_path, paths['result'])) as prev_result_file:
                prev_result = json.load(prev_result_file)
                prev_exit_code = prev_result['result']['exit_code']
                # Vampire terminates with exit code 3 if it is interrupted by SIGINT.
                if self._overwrite == 'none' or (self._overwrite == 'failed' and prev_exit_code == 0) or (
                        self._overwrite == 'interrupted' and prev_exit_code != 3):
                    result['status'] = 'skipped'
                if result['status'] == 'skipped':
                    result['exit_code'] = prev_exit_code
        if result['status'] is None:
            with self.scratch_directory() as scratch_directory_name:
                json_output_dir = output_path
                if scratch_directory_name is not None:
                    json_output_dir = scratch_directory_name
                    paths['scratch_job'] = os.path.abspath(scratch_directory_name)
                vampire_options = self.compose_vampire_options(problem_path, json_output_dir, random_seed_zero_based)
                vampire_args = [self._vampire] + vampire_options
                timeout_seconds = self._vampire_timeout
                if timeout_seconds is None:
                    vampire_time_limit = utils.option_value(vampire_options, '--time_limit')
                    if vampire_time_limit is not None:
                        # TODO: Support all the time formats supported by Vampire.
                        vampire_time_limit_seconds = float(vampire_time_limit)
                        timeout_seconds = vampire_time_limit_seconds + 1
                configuration.update({
                    'command': ' '.join(vampire_args),
                    'timeout_seconds': timeout_seconds,
                    'options': vampire_options
                })
                # If JSON output directory does not exist, Vampire fails.
                os.makedirs(output_path, exist_ok=True)
                with open(os.path.join(output_path, paths['configuration']), 'w') as configuration_json:
                    json.dump({'paths': paths, 'configuration': configuration}, configuration_json, indent=4)
                time_start = time.time()
                try:
                    cp = subprocess.run(vampire_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        timeout=timeout_seconds,
                                        text=True)
                    result['time_elapsed'] = time.time() - time_start
                    result['status'] = 'completed'
                    result['exit_code'] = cp.returncode
                    stdout_str = cp.stdout
                    stderr_str = cp.stderr
                except subprocess.TimeoutExpired as e:
                    result['time_elapsed'] = time.time() - time_start
                    result['status'] = 'timeout_expired'
                    stdout_str = e.stdout
                    stderr_str = e.stderr
                if json_output_dir != output_path and result['exit_code'] == 0:
                    try:
                        shutil.move(os.path.join(json_output_dir, paths['vampire_json']),
                                    os.path.join(output_path, paths['vampire_json']))
                    except FileNotFoundError:
                        pass
            assert self._scratch is None or result['exit_code'] == 0 or not os.path.isfile(
                os.path.join(output_path, paths['vampire_json']))
            # If self._scratch is None, the Vampire output JSON file has been deleted along with the scratch directory.
            if self._scratch is None and result['exit_code'] != 0:
                # Vampire may or may not have created the file.
                try:
                    os.remove(os.path.join(output_path, paths['vampire_json']))
                except FileNotFoundError:
                    pass
            csv_writer.writerow({
                'output_path': os.path.relpath(paths['output_run'], self._output_path),
                'problem_path': os.path.relpath(paths['problem'], problem_base_path),
                'status': result['status'],
                'exit_code': result['exit_code'],
                'time_elapsed': result['time_elapsed']
            })
            with open(os.path.join(output_path, paths['result']), 'w') as run_json:
                json.dump({'paths': paths, 'result': result}, run_json, indent=4)
            with open(os.path.join(output_path, paths['stdout']), 'w') as stdout_file:
                stdout_file.write(stdout_str)
            with open(os.path.join(output_path, paths['stderr']), 'w') as stderr_file:
                stderr_file.write(stderr_str)
        return {
            'paths': paths,
            'configuration': configuration,
            'result': result,
            'stdout': stdout_str,
            'stderr': stderr_str
        }

    @contextlib.contextmanager
    def scratch_directory(self):
        """Yield temporary scratch directory name or None if self._scratch is None."""
        if self._scratch is not None:
            os.makedirs(self._scratch, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self._scratch) as tempdirname:
                yield tempdirname
        else:
            yield None
