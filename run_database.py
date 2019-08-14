#!/usr/bin/env python3.7

import csv
import functools
import json
import os

import numpy as np

import extractor


class Run:
    def __init__(self, csv_row, base_path):
        self._csv_row = csv_row
        self._base_path = base_path

    def __hash__(self):
        # https://stackoverflow.com/a/16162138/4054250
        return hash((frozenset(self._csv_row), frozenset(self._csv_row.values()), self._base_path))

    @property
    def path_rel(self):
        return self._csv_row['output_path']

    @property
    @functools.lru_cache(maxsize=1)
    def path_abs(self):
        return os.path.join(self._base_path, self.path_rel)

    @property
    def problem_path(self):
        return self._csv_row['problem_path']

    @property
    @functools.lru_cache(maxsize=1)
    def stdout(self):
        with open(os.path.join(self.path_abs, 'stdout.txt')) as stdout_file:
            return stdout_file.read()

    @property
    @functools.lru_cache(maxsize=1)
    def vampire_json_data(self):
        if self.exit_code != 0:
            raise RuntimeError('This run failed. The output JSON data may be missing or invalid.')
        with open(os.path.join(self.path_abs, 'vampire.json')) as vampire_json_file:
            return json.load(vampire_json_file)

    @property
    def exit_code(self):
        return int(self._csv_row['exit_code'])

    @property
    def time_elapsed_process(self):
        return float(self._csv_row['time_elapsed'])

    @property
    @functools.lru_cache(maxsize=1)
    def termination_reason(self):
        return self.__extract(extractor.termination_reason)

    @property
    @functools.lru_cache(maxsize=1)
    def termination_phase(self):
        return self.__extract(extractor.termination_phase)

    @property
    @functools.lru_cache(maxsize=1)
    def saturation_iterations(self):
        return self.__extract(extractor.saturation_iterations)

    def __extract(self, extract):
        try:
            return extract(self.stdout)
        except RuntimeError:
            return None

    @property
    @functools.lru_cache(maxsize=1)
    def predicates(self):
        return self.vampire_json_data['predicates']

    @property
    def predicates_count(self):
        try:
            return len(self.predicates)
        except RuntimeError:
            return None

    @property
    @functools.lru_cache(maxsize=1)
    def time_elapsed_vampire(self):
        return self.__extract(extractor.time_elapsed)

    @property
    @functools.lru_cache(maxsize=1)
    def functions(self):
        return self.vampire_json_data['functions']

    @property
    def functions_count(self):
        try:
            return len(self.functions)
        except RuntimeError:
            return None

    @property
    @functools.lru_cache(maxsize=1)
    def clauses(self):
        return self.vampire_json_data['clauses']

    @property
    def clauses_count(self):
        try:
            return len(self.clauses)
        except RuntimeError:
            return None

    @property
    @functools.lru_cache(maxsize=1)
    def predicate_precedence(self):
        return self.__extract(extractor.predicate_precedence)

    @property
    def success(self):
        assert (self.termination_reason in ['Refutation', 'Satisfiable']) == (self.exit_code == 0)
        return self.termination_reason in ['Refutation', 'Satisfiable']

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

    @functools.lru_cache(maxsize=1)
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
        assert row['time_elapsed.vampire'] is not None or self.exit_code != 0
        assert row['termination.reason'] is not None or self.exit_code != 0
        assert row['saturation.iterations'] is not None or self.exit_code != 0
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
        assert (row['time_elapsed.vampire'] is None) == (self.exit_code == 0)
        assert (row['termination.reason'] is None) == (self.exit_code == 0)
        assert (row['termination.phase'] is None) == (self.exit_code == 0)
        assert (row['predicates.count'] is None) == (self.exit_code != 0)
        assert (row['functions.count'] is None) == (self.exit_code != 0)
        assert (row['clauses.count'] is None) == (self.exit_code != 0)
        return row


class BatchResult:
    def __init__(self, result_path):
        self._result_path = result_path

    def __hash__(self):
        return hash(self._result_path)

    @property
    @functools.lru_cache(maxsize=1)
    def base_directory(self):
        return os.path.dirname(self._result_path)

    @property
    @functools.lru_cache(maxsize=1)
    def result(self):
        with open(self._result_path) as result_file:
            return json.load(result_file)

    @property
    def vampire_options(self):
        return self.result['vampire_options']

    def __option_value(self, option_name, default=None):
        try:
            option_index = list(reversed(self.vampire_options)).index(option_name)
        except ValueError:
            return default
        if option_index == 0:
            # The option name was the last argument.
            return default
        assert option_index >= 1
        return self.vampire_options[-option_index]

    @property
    def mode(self):
        return self.__option_value('--mode', 'vampire')

    @property
    @functools.lru_cache(maxsize=1)
    def run_list(self):
        runs_csv_path = self.result['runs_csv']
        with open(os.path.join(self.base_directory, runs_csv_path)) as runs_csv:
            csv_reader = csv.DictReader(runs_csv)
            return [Run(row, self.base_directory) for row in csv_reader]

    @property
    @functools.lru_cache(maxsize=1)
    def problem_dict(self):
        problem_dict = {}
        for run in self.run_list:
            problem_path = run.problem_path
            if problem_path not in problem_dict:
                problem_dict[problem_path] = []
            problem_dict[problem_path].append(run)
        return problem_dict

    @property
    @functools.lru_cache(maxsize=1)
    def problems_with_some_success(self):
        """
        :return: paths of problems that have at least one successful run
        """
        # We use Python 3.7 dict instead of set because it iterates in a deterministic order.
        # https://stackoverflow.com/a/53657523/4054250
        return {run.problem_path: None for run in self.run_list if run.success}.keys()
