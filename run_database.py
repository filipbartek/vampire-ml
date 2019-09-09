#!/usr/bin/env python3.7

import csv
import itertools
import json
import os

import methodtools
import numpy as np
import pandas as pd

import extractor
import utils


class Run:
    def __init__(self, csv_row, base_path, batch_id):
        self._csv_row = csv_row
        self._base_path = base_path
        self._batch_id = batch_id

    def __hash__(self):
        # https://stackoverflow.com/a/16162138/4054250
        return hash((frozenset(self._csv_row), frozenset(self._csv_row.values()), self._base_path))

    @property
    def batch_id(self):
        return self._batch_id

    @property
    def path_rel(self):
        return self._csv_row['output_path']

    @property
    def path_abs(self):
        return self.get_path_abs()

    @methodtools.lru_cache(maxsize=1)
    def get_path_abs(self):
        return os.path.join(self._base_path, self.path_rel)

    @property
    def problem_path(self):
        return self._csv_row['problem_path']

    @property
    def problem_dir(self):
        return os.path.dirname(self.problem_path)

    @property
    def probe(self):
        assert self._csv_row['probe'] in ['False', 'True']
        return self._csv_row['probe'] == 'True'

    @property
    def timeout(self):
        assert self._csv_row['timeout'] in ['False', 'True']
        return self._csv_row['timeout'] == 'True'

    @property
    def stdout(self):
        return self.get_stdout()

    @methodtools.lru_cache(maxsize=1)
    def get_stdout(self):
        with open(os.path.join(self.path_abs, 'stdout.txt')) as stdout_file:
            return stdout_file.read()

    @property
    def vampire_json_data(self):
        return self.get_vampire_json_data()

    @methodtools.lru_cache(maxsize=1)
    def get_vampire_json_data(self):
        if self.exit_code != 0:
            raise RuntimeError('This run failed. The output JSON data may be missing or invalid.')
        with open(os.path.join(self.path_abs, 'vampire.json')) as vampire_json_file:
            return json.load(vampire_json_file)

    @property
    def exit_code(self):
        try:
            return int(self._csv_row['exit_code'])
        except ValueError:
            return None

    @property
    def time_elapsed_process(self):
        return float(self._csv_row['time_elapsed'])

    @property
    def termination_reason(self):
        return self.__extract(extractor.termination_reason)

    @property
    def termination_phase(self):
        return self.__extract(extractor.termination_phase)

    @property
    def saturation_iterations(self):
        return self.__extract(extractor.saturation_iterations)

    @methodtools.lru_cache()
    def __extract(self, extract):
        try:
            return extract(self.stdout)
        except (RuntimeError, FileNotFoundError):
            return None

    @property
    def predicates(self):
        return self.vampire_json_data['predicates']

    @property
    def predicates_count(self):
        try:
            return len(self.predicates)
        except (RuntimeError, FileNotFoundError):
            return None

    @property
    def time_elapsed_vampire(self):
        return self.__extract(extractor.time_elapsed)

    @property
    def functions(self):
        return self.vampire_json_data['functions']

    @property
    def functions_count(self):
        try:
            return len(self.functions)
        except (RuntimeError, FileNotFoundError):
            return None

    @property
    def clauses(self):
        return self.vampire_json_data['clauses']

    @property
    def clauses_count(self):
        try:
            return len(self.clauses)
        except (RuntimeError, FileNotFoundError):
            return None

    @property
    def predicate_precedence(self):
        return self.__extract(extractor.predicate_precedence)

    @property
    def success(self):
        return self.exit_code == 0

    # Predicate features: equality?, arity, usageCnt, unitUsageCnt, inGoal?, inUnit?
    predicate_feature_count = 6

    # TODO: Construct embedding using convolution a graph network.
    def predicate_embedding(self, predicate_index):
        predicate = self.predicates[predicate_index]
        assert not predicate['skolem']
        assert not predicate['inductionSkolem']
        is_equality = predicate_index == 0
        assert is_equality == (predicate['name'] == '=')
        assert not is_equality or predicate['arity'] == 2
        return np.asarray([
            is_equality,
            predicate['arity'],
            predicate['usageCnt'],
            predicate['unitUsageCnt'],
            predicate['inGoal'],
            predicate['inUnit']
        ], dtype=float)

    @methodtools.lru_cache(maxsize=1)
    def predicate_embeddings(self):
        result = np.zeros((self.predicates_count, self.predicate_feature_count), dtype=float)
        for i in range(self.predicates_count):
            # TODO: Omit constructing an array for each predicate.
            result[i] = self.predicate_embedding(i)
        return result

    def csv_row_vampire(self):
        row = {
            'problem_path': self.problem_path,
            'exit_code': self.exit_code,
            'time_elapsed.process': self.time_elapsed_process,
            'time_elapsed.vampire': self.time_elapsed_vampire,
            'termination.reason': self.termination_reason,
            'termination.phase': self.termination_phase,
            'saturation.iterations': self.saturation_iterations
        }
        assert row['problem_path'] is not None
        assert row['exit_code'] is not None
        assert row['time_elapsed.process'] is not None
        return row

    def csv_row_clausify(self):
        assert self.saturation_iterations is None
        row = {
            'problem_path': self.problem_path,
            'exit_code': self.exit_code,
            'time_elapsed.process': self.time_elapsed_process,
            'time_elapsed.vampire': self.time_elapsed_vampire,
            'termination.reason': self.termination_reason,
            'termination.phase': self.termination_phase,
            'predicates.count': self.predicates_count,
            'functions.count': self.functions_count,
            'clauses.count': self.clauses_count
        }
        assert row['problem_path'] is not None
        assert row['exit_code'] is not None
        assert row['time_elapsed.process'] is not None
        return row

    @staticmethod
    def get_data_frame(runs):
        return pd.DataFrame({
            'batch': (run.batch_id for run in runs),
            'path_rel': (run.path_rel for run in runs),
            'problem_path': (run.problem_path for run in runs),
            'problem_dir': (run.problem_dir for run in runs),
            'probe': pd.Series((run.probe for run in runs), dtype=np.bool),
            'timeout': pd.Series((run.timeout for run in runs), dtype=np.bool),
            'exit_code': pd.Categorical(run.exit_code for run in runs),
            'termination_reason': pd.Categorical(run.termination_reason for run in runs),
            'termination_phase': pd.Categorical(run.termination_phase for run in runs),
            'success': pd.Series((run.success for run in runs), dtype=np.bool),
            'time_elapsed_process': (run.time_elapsed_process for run in runs),
            'time_elapsed_vampire': (run.time_elapsed_vampire for run in runs),
            'saturation_iterations': (run.saturation_iterations for run in runs),
            'predicates_count': (run.predicates_count for run in runs),
            'functions_count': (run.functions_count for run in runs),
            'clauses_count': (run.clauses_count for run in runs)
        })


class BatchResult:
    def __init__(self, result_path, base_path):
        self._result_path = result_path
        self._base_path = base_path

    def __hash__(self):
        return hash(self._result_path)

    @property
    def result_path(self):
        return self._result_path

    @property
    def id(self):
        return os.path.relpath(self._result_path, self._base_path)

    @property
    def batch_output_directory(self):
        return self.get_batch_output_directory()

    @methodtools.lru_cache(maxsize=1)
    def get_batch_output_directory(self):
        return os.path.dirname(self.result_path)

    @property
    def run_output_directory(self):
        return self.get_run_output_directory()

    @methodtools.lru_cache(maxsize=1)
    def get_run_output_directory(self):
        return os.path.join(self.batch_output_directory, self.result['run_output_base_path'])

    @property
    def result(self):
        return self.get_result()

    @methodtools.lru_cache(maxsize=1)
    def get_result(self):
        with open(self.result_path) as result_file:
            return json.load(result_file)

    @property
    def vampire_options(self):
        return self.result['vampire_options']

    @methodtools.lru_cache(maxsize=1)
    def __option_value(self, option_name, default=None):
        return utils.option_value(self.vampire_options, option_name, default)

    @property
    def mode(self):
        return self.__option_value('--mode', 'vampire')

    @property
    def problem_base_path(self):
        return self.result['problem_base_path']

    @property
    def runs(self):
        return self.get_run_list()

    @methodtools.lru_cache(maxsize=1)
    def get_run_list(self):
        runs_csv_path = self.result['runs_csv']
        with open(os.path.join(self.batch_output_directory, runs_csv_path)) as runs_csv:
            csv_reader = csv.DictReader(runs_csv)
            return [Run(row, self.run_output_directory, self.id) for row in csv_reader]

    @property
    def problem_dict(self):
        return self.get_problem_dict()

    @methodtools.lru_cache(maxsize=1)
    def get_problem_dict(self):
        problem_dict = {}
        for run in self.runs:
            problem_path = run.problem_path
            if problem_path not in problem_dict:
                problem_dict[problem_path] = []
            problem_dict[problem_path].append(run)
        return problem_dict

    @property
    def problems_with_some_success(self):
        return self.get_problems_with_some_success()

    @methodtools.lru_cache(maxsize=1)
    def get_problems_with_some_success(self):
        """
        :return: paths of problems that have at least one successful run
        """
        # We use Python 3.7 dict instead of set because it iterates in a deterministic order.
        # https://stackoverflow.com/a/53657523/4054250
        return {run.problem_path: None for run in self.runs if run.success}.keys()

    @property
    def runs_data_frame(self):
        return self.get_runs_data_frame()

    @methodtools.lru_cache(maxsize=1)
    def get_runs_data_frame(self):
        return Run.get_data_frame(self.runs)

    @property
    def representative_runs(self):
        return self.get_representative_runs()

    @methodtools.lru_cache(maxsize=1)
    def get_representative_runs(self):
        return [problem_runs[0] for problem_runs in self.problem_dict.values()]

    @property
    def representative_runs_data_frame(self):
        return self.get_representative_runs_data_frame()

    @methodtools.lru_cache(maxsize=1)
    def get_representative_runs_data_frame(self):
        assert Run.get_data_frame(self.representative_runs)['problem_path'].is_unique
        return Run.get_data_frame(self.representative_runs).set_index('problem_path')

    @property
    def problems(self):
        run_groups = self.runs_data_frame.groupby('problem_path')
        result = run_groups.size().to_frame('runs_count')
        result = result.join(run_groups.agg([np.mean, np.std, np.min, np.max]))
        return result


class MultiBatchResult:
    def __init__(self, result_paths):
        result_paths = list(result_paths)
        assert len(result_paths) >= 1
        base_path = os.path.commonpath(result_paths)
        self._batch_results = [BatchResult(result_path, base_path) for result_path in result_paths]

    @property
    def runs(self):
        return self.get_run_list()

    @methodtools.lru_cache(maxsize=1)
    def get_run_list(self):
        return list(itertools.chain(*(br.runs for br in self._batch_results)))

    @property
    def problem_dict(self):
        return self.get_problem_dict()

    @methodtools.lru_cache(maxsize=1)
    def get_problem_dict(self):
        # TODO: Ensure all the runs have the same problem base path.
        problem_dict = {}
        for run in self.runs:
            problem_path = run.problem_path
            if problem_path not in problem_dict:
                problem_dict[problem_path] = []
            problem_dict[problem_path].append(run)
        return problem_dict

    @property
    def problems_with_some_success(self):
        return self.get_problems_with_some_success()

    @methodtools.lru_cache(maxsize=1)
    def get_problems_with_some_success(self):
        """
        :return: paths of problems that have at least one successful run
        """
        # We use Python 3.7 dict instead of set because it iterates in a deterministic order.
        # https://stackoverflow.com/a/53657523/4054250
        return {run.problem_path: None for run in self.runs if run.success}.keys()

    @property
    def runs_data_frame(self):
        return self.get_runs_data_frame()

    @methodtools.lru_cache(maxsize=1)
    def get_runs_data_frame(self):
        # TODO: Add batch id column.
        return Run.get_data_frame(self.runs)

    @property
    def representative_runs(self):
        return self.get_representative_runs()

    @methodtools.lru_cache(maxsize=1)
    def get_representative_runs(self):
        return [problem_runs[0] for problem_runs in self.problem_dict.values()]

    @property
    def representative_runs_data_frame(self):
        return self.get_representative_runs_data_frame()

    @methodtools.lru_cache(maxsize=1)
    def get_representative_runs_data_frame(self):
        assert Run.get_data_frame(self.representative_runs)['problem_path'].is_unique
        return Run.get_data_frame(self.representative_runs).set_index('problem_path')

    @property
    def problems(self):
        run_groups = self.runs_data_frame.groupby('problem_path')
        result = run_groups.size().to_frame('runs_count')
        result = result.join(run_groups.agg([np.mean, np.std, np.min, np.max]))
        return result
