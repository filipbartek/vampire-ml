#!/usr/bin/env python3.7

import csv
import itertools
import json
import logging
import os

import methodtools
import numpy as np
import pandas as pd
import tqdm

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

    def __getitem__(self, key):
        g = self.field_getter(key)
        return g(self)

    @methodtools.lru_cache()
    def __extract(self, extract):
        try:
            return extract(self.stdout())
        except RuntimeError:
            pass
        except FileNotFoundError:
            logging.warning(f'{self.path_rel()}: Stdout file not found. Cannot extract a value.')
        return None

    @methodtools.lru_cache(maxsize=1)
    def stdout(self):
        with open(os.path.join(self.path_abs(), 'stdout.txt')) as stdout_file:
            return stdout_file.read()

    @methodtools.lru_cache(maxsize=1)
    def vampire_json_data(self):
        """Loads and returns JSON data output by Vampire.

        Warning: This call may be very expensive.
        """
        if self.exit_code() != 0:
            raise RuntimeError('This run failed. The output JSON data may be missing or invalid.')
        with open(os.path.join(self.path_abs(), 'vampire.json')) as vampire_json_file:
            return json.load(vampire_json_file)

    def batch_id(self):
        return self._batch_id

    def path_rel(self):
        return self._csv_row['output_path']

    def path_abs(self):
        return os.path.join(self._base_path, self.path_rel())

    def problem_path(self):
        return self._csv_row['problem_path']

    def problem_dir(self):
        return os.path.dirname(self.problem_path())

    def status(self):
        return self._csv_row['status']

    def exit_code(self):
        try:
            return int(self._csv_row['exit_code'])
        except ValueError:
            return None

    def termination_reason(self):
        return self.__extract(extractor.termination_reason)

    def termination_phase(self):
        return self.__extract(extractor.termination_phase)

    def time_elapsed_process(self):
        return float(self._csv_row['time_elapsed'])

    def time_elapsed_vampire(self):
        return self.__extract(extractor.time_elapsed)

    def memory_used(self):
        return self.__extract(extractor.memory_used)

    def saturation_iterations(self):
        return self.__extract(extractor.saturation_iterations)

    def predicates(self):
        return self.vampire_json_data()['predicates']

    def predicates_count(self):
        try:
            return len(self.predicates())
        except (RuntimeError, FileNotFoundError):
            return None

    def functions(self):
        return self.vampire_json_data()['functions']

    def functions_count(self):
        try:
            return len(self.functions())
        except (RuntimeError, FileNotFoundError):
            return None

    def clauses(self):
        return self.vampire_json_data()['clauses']

    def clauses_count(self):
        try:
            return len(self.clauses())
        except (RuntimeError, FileNotFoundError):
            return None

    fields = {
        'batch': (batch_id, pd.CategoricalDtype(ordered=True)),
        'path_rel': (path_rel, 'object'),
        'problem_path': (problem_path, 'object'),
        'problem_dir': (problem_dir, pd.CategoricalDtype()),
        'status': (status, pd.CategoricalDtype()),
        'exit_code': (exit_code, pd.CategoricalDtype()),
        'termination_reason': (termination_reason, pd.CategoricalDtype()),
        'termination_phase': (termination_phase, pd.CategoricalDtype()),
        'time_elapsed_process': (time_elapsed_process, np.float),
        'time_elapsed_vampire': (time_elapsed_vampire, np.float),
        'memory_used': (memory_used, pd.UInt64Dtype()),
        'saturation_iterations': (saturation_iterations, pd.UInt64Dtype()),
        'predicates_count': (predicates_count, pd.UInt64Dtype()),
        'functions_count': (functions_count, pd.UInt64Dtype()),
        'clauses_count': (clauses_count, pd.UInt64Dtype())
    }

    field_sources = {
        'stdout': ['termination_reason', 'termination_phase', 'time_elapsed_vampire', 'memory_used',
                   'saturation_iterations'],
        'vampire_json': ['predicates_count', 'functions_count', 'clauses_count']
    }

    @classmethod
    def fieldnames(cls):
        return cls.fields.keys()

    @classmethod
    def field_getter(cls, fieldname):
        return cls.fields[fieldname][0]

    @classmethod
    def field_dtype(cls, fieldname):
        return cls.fields[fieldname][1]

    @staticmethod
    def empty_pd_series(n, dtype):
        if dtype == pd.UInt64Dtype():
            return pd.Series(np.empty(n, dtype=np.uint), dtype=dtype)
        if isinstance(dtype, pd.CategoricalDtype):
            return pd.Series(index=range(n), dtype='object')
        return pd.Series(index=range(n), dtype=dtype)

    @classmethod
    def get_fieldnames_final(cls, fieldnames_initial=None, excluded_sources=None):
        fieldnames_final = fieldnames_initial
        if fieldnames_final is None:
            fieldnames_final = cls.fieldnames()
        else:
            fieldnames_final = fieldnames_final.copy()
        fieldnames_final = list(fieldnames_final)
        if excluded_sources is not None:
            for source in excluded_sources:
                for field in cls.field_sources[source]:
                    try:
                        while True:
                            fieldnames_final.remove(field)
                    except ValueError:
                        pass
        return fieldnames_final

    @classmethod
    def get_data_frame(cls, runs, run_count=None, fieldnames=None, excluded_sources=None):
        if run_count is None:
            runs = list(runs)
            run_count = len(runs)
        assert run_count >= 0
        fieldnames = cls.get_fieldnames_final(fieldnames, excluded_sources)
        assert set(fieldnames) <= set(cls.fieldnames())
        # Initialize empty series
        series = {fieldname: cls.empty_pd_series(run_count, cls.field_dtype(fieldname)) for fieldname in fieldnames}
        # Populate the series with run data
        for i, run in tqdm.tqdm(enumerate(runs), total=run_count, unit='run'):
            assert i < run_count
            for fieldname in fieldnames:
                series[fieldname][i] = run[fieldname]
        # Establish categories in respective series
        for fieldname in fieldnames:
            dtype = cls.field_dtype(fieldname)
            if isinstance(dtype, pd.CategoricalDtype):
                series[fieldname] = series[fieldname].astype(dtype, copy=False)
        # Create the dataframe
        df = pd.DataFrame(series)
        return df

    def predicate_precedence(self):
        return self.__extract(extractor.predicate_precedence)

    # Predicate features: equality?, arity, usageCnt, unitUsageCnt, inGoal?, inUnit?
    predicate_feature_count = 6

    # TODO: Construct embedding using convolution a graph network.
    def predicate_embedding(self, predicate_index):
        predicate = self.predicates()[predicate_index]
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
        result = np.zeros((self.predicates_count(), self.predicate_feature_count), dtype=float)
        for i in range(self.predicates_count()):
            # TODO: Omit constructing an array for each predicate.
            result[i] = self.predicate_embedding(i)
        return result


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
        return list(self.generate_runs())

    def generate_runs(self):
        with open(os.path.join(self.batch_output_directory, self.result['runs_csv'])) as runs_csv:
            csv_reader = csv.DictReader(runs_csv)
            for row in csv_reader:
                yield Run(row, self.run_output_directory, self.id)

    @property
    def run_count(self):
        return self.get_run_count()

    @methodtools.lru_cache(maxsize=1)
    def get_run_count(self):
        with open(os.path.join(self.batch_output_directory, self.result['runs_csv'])) as runs_csv:
            csv_reader = csv.DictReader(runs_csv)
            acc = 0
            for _ in csv_reader:
                acc += 1
            return acc

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
    def runs_data_frame(self):
        return self.get_runs_data_frame()

    @methodtools.lru_cache(maxsize=1)
    def get_runs_data_frame(self):
        return Run.get_data_frame(self.generate_runs(), self.run_count)

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
        return list(self.generate_runs())

    def generate_runs(self):
        return itertools.chain(*(br.generate_runs() for br in self._batch_results))

    @property
    def run_count(self):
        return self.get_run_count()

    @methodtools.lru_cache(maxsize=1)
    def get_run_count(self):
        acc = 0
        for br in self._batch_results:
            acc += br.run_count
        return acc

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
    def runs_data_frame(self):
        return self.get_runs_data_frame()

    @methodtools.lru_cache()
    def get_runs_data_frame(self, fields=None, excluded_sources=None):
        return Run.get_data_frame(self.generate_runs(), self.run_count, fields, excluded_sources)

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
