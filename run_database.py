#!/usr/bin/env python3.7

import json
import logging
import os

import methodtools
import numpy as np
import pandas as pd
import tqdm

import extractor


class Run:
    def __init__(self, result_path, problem_base_path=None):
        self._result_path = result_path
        self._problem_base_path = problem_base_path
        self._stdout = None
        self._symbols = None
        self._clauses = None

    def __getitem__(self, key):
        g = self.field_getter(key)
        return g(self)

    # TODO: Rename to `directory_path`.
    @methodtools.lru_cache(maxsize=1)
    def path_abs(self):
        return os.path.dirname(self._result_path)

    @methodtools.lru_cache(maxsize=1)
    def result_json_data(self):
        with open(self._result_path) as result_json_file:
            return json.load(result_json_file)

    def paths(self):
        return self.result_json_data()['paths']

    def result(self):
        return self.result_json_data()['result']

    @methodtools.lru_cache()
    def __extract(self, extract):
        if self._stdout is None:
            return None
        try:
            return extract(self._stdout)
        except RuntimeError:
            pass
        return None

    def load_stdout(self):
        if self._stdout is not None:
            return
        file_path = os.path.join(self.path_abs(), self.paths()['stdout'])
        try:
            with open(file_path) as stdout_file:
                self._stdout = stdout_file.read()
        except FileNotFoundError:
            logging.warning(f'Stdout file not found: {file_path}')

    def load_symbols(self):
        if self._symbols is not None:
            return
        if self.exit_code() != 0:
            logging.debug(f'Skipping loading symbols because the run failed: {self._result_path}')
            return
        file_path = os.path.join(self.path_abs(), self.paths()['symbols_csv'])
        try:
            self._symbols = pd.read_csv(file_path, index_col=['isFunction', 'id'], dtype={
                'isFunction': np.bool,
                'id': pd.UInt64Dtype(),
                'name': 'object',
                'arity': pd.UInt64Dtype(),
                'usageCnt': pd.UInt64Dtype(),
                'unitUsageCnt': pd.UInt64Dtype(),
                'inGoal': np.bool,
                'inUnit': np.bool,
                'skolem': np.bool,
                'inductionSkolem': np.bool
            })
        except FileNotFoundError:
            logging.warning(f'Symbols CSV file not found: {file_path}')
        except pd.errors.ParseError as e:
            logging.warning(f'Parse error in {file_path}: {e}')

    def load_clauses(self):
        if self._clauses is not None:
            return
        if self.exit_code() != 0:
            logging.debug(f'Skipping loading clauses because the run failed: {self._result_path}')
            return
        file_path = os.path.join(self.path_abs(), self.paths()['clauses_json'])
        try:
            with open(file_path) as clauses_json_file:
                self._clauses = json.load(clauses_json_file)
        except FileNotFoundError:
            logging.warning(f'Clauses JSON file not found: {file_path}')

    def unload(self):
        self._stdout = None
        self._symbols = None
        self._clauses = None

    # TODO: Rename to `job` or `job_id`.
    def batch_id(self):
        # TODO: Simplify this path.
        return self.paths()['job']

    def path_rel(self):
        # TODO: Try to simplify the path. What is it supposed to be relative to?
        return self.path_abs()

    def problem_path(self):
        if self._problem_base_path is not None:
            return os.path.relpath(self.paths()['problem'], self._problem_base_path)
        return self.paths()['problem']

    def problem_dir(self):
        return os.path.dirname(self.problem_path())

    def status(self):
        return self.result()['status']

    def exit_code(self):
        assert isinstance(self.result()['exit_code'], int)
        return self.result()['exit_code']

    def termination_reason(self):
        return self.__extract(extractor.termination_reason)

    def termination_phase(self):
        return self.__extract(extractor.termination_phase)

    def time_elapsed_process(self):
        assert isinstance(self.result()['time_elapsed'], float)
        return self.result()['time_elapsed']

    def time_elapsed_vampire(self):
        return self.__extract(extractor.time_elapsed)

    def memory_used(self):
        return self.__extract(extractor.memory_used)

    def saturation_iterations(self):
        return self.__extract(extractor.saturation_iterations)

    @methodtools.lru_cache(maxsize=2)
    def symbols(self, is_function=None):
        if self._symbols is None:
            return None
        if is_function is None:
            return self._symbols
        result = self._symbols[self._symbols.index.get_level_values('isFunction') == is_function]
        result.index = result.index.droplevel(0)
        return result

    def predicates(self):
        return self.symbols(False)

    def predicates_count(self):
        try:
            return len(self.predicates())
        except TypeError:
            return None

    def functions(self):
        return self.symbols(True)

    def functions_count(self):
        try:
            return len(self.functions())
        except TypeError:
            return None

    def clauses(self):
        return self._clauses

    def clauses_count(self):
        try:
            return len(self.clauses())
        except TypeError:
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
        'symbols': ['predicates_count', 'functions_count'],
        'clauses': ['clauses_count']
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
    def get_fieldnames_final(cls, fieldnames_initial=None, sources=None):
        fieldnames_final = fieldnames_initial
        if fieldnames_final is None:
            fieldnames_final = cls.fieldnames()
        else:
            fieldnames_final = fieldnames_final.copy()
        fieldnames_final = list(fieldnames_final)
        if sources is not None:
            for fields in (fields for source, fields in cls.field_sources.items() if source not in sources):
                for field in fields:
                    try:
                        while True:
                            fieldnames_final.remove(field)
                    except ValueError:
                        pass
        return fieldnames_final

    @classmethod
    def get_data_frame(cls, runs, run_count=None, fieldnames=None, sources=None):
        if run_count is None:
            runs = list(runs)
            run_count = len(runs)
        assert run_count >= 0
        fieldnames = cls.get_fieldnames_final(fieldnames, sources)
        assert set(fieldnames) <= set(cls.fieldnames())
        # Initialize empty series
        series = {fieldname: cls.empty_pd_series(run_count, cls.field_dtype(fieldname)) for fieldname in fieldnames}
        # Populate the series with run data
        for i, run in tqdm.tqdm(enumerate(runs), total=run_count, unit='run'):
            assert i < run_count
            # Run expensive functions first to facilitate profiling.
            run.result_json_data()
            if 'stdout' in sources:
                run.load_stdout()
            if 'symbols' in sources:
                run.load_symbols()
            if 'clauses' in sources:
                run.load_clauses()
            for fieldname in fieldnames:
                series[fieldname][i] = run[fieldname]
            run.unload()
        # Establish categories in respective series
        for fieldname in fieldnames:
            dtype = cls.field_dtype(fieldname)
            if isinstance(dtype, pd.CategoricalDtype):
                series[fieldname] = series[fieldname].astype(dtype, copy=False)
        # Create the dataframe
        df = pd.DataFrame(series).set_index('path_rel').sort_index()
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
