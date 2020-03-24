#!/usr/bin/env python3.7

import collections
import contextlib
import copy
import hashlib
import itertools
import json
import logging
import os
import tempfile
import warnings

import methodtools
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import len_robust
from . import extractor
from . import process


class Execution:
    def __init__(self, configuration, result, path):
        self.configuration = configuration
        self.result = result
        self.path = path

    def __str__(self):
        return str({'path': self.path, 'exit_code': self.result.exit_code})

    def __getitem__(self, key):
        return self.fields[key].extract(self)

    Field = collections.namedtuple('Field', ['extract', 'dtype'])

    # TODO: Add vampire job id.
    fields = {
        'output_dir': Field(lambda job: job.path, 'object'),
        'problem_path': Field(lambda job: job.configuration.problem, 'object'),
        'status': Field(lambda job: job.result.status, pd.CategoricalDtype()),
        'exit_code': Field(lambda job: job.result.exit_code, pd.CategoricalDtype()),
        'termination_reason': Field(lambda job: extractor.termination_reason(job.result.stdout), pd.CategoricalDtype()),
        'termination_phase': Field(lambda job: extractor.termination_phase(job.result.stdout), pd.CategoricalDtype()),
        'time_elapsed_process': Field(lambda job: job.result.time_elapsed, np.float),
        'time_elapsed_vampire': Field(lambda job: extractor.time_elapsed(job.result.stdout), np.float),
        'memory_used': Field(lambda job: extractor.memory_used(job.result.stdout), pd.UInt64Dtype()),
        'saturation_iterations': Field(lambda job: extractor.saturation_iterations(job.result.stdout),
                                       pd.UInt64Dtype()),
        'predicates_count': Field(lambda job: len_robust(job.result.get_symbols('predicate')), pd.UInt64Dtype()),
        'functions_count': Field(lambda job: len_robust(job.result.get_symbols('function')), pd.UInt64Dtype()),
        'clauses_count': Field(lambda job: len_robust(job.result.clauses), pd.UInt64Dtype())
    }

    field_names_clausify = ['output_dir', 'problem_path', 'status', 'exit_code', 'termination_reason',
                            'termination_phase', 'time_elapsed_process', 'time_elapsed_vampire', 'memory_used',
                            'predicates_count', 'functions_count', 'clauses_count']
    field_names_solve = ['output_dir', 'problem_path', 'status', 'exit_code', 'termination_reason', 'termination_phase',
                         'time_elapsed_process', 'time_elapsed_vampire', 'memory_used', 'saturation_iterations']

    def get_dataframe(self, field_names_obligatory=None):
        if field_names_obligatory is None:
            field_names_obligatory = set()
        field_names_obligatory = set(field_names_obligatory)
        typed_series = {field_name: pd.Series([self[field_name]], dtype=self.fields[field_name].dtype) for field_name in
                        self.fields.keys() if field_name in field_names_obligatory or self[field_name] is not None}
        df = pd.DataFrame(typed_series)
        return df

    @classmethod
    def concat_dfs(cls, dfs):
        dfs = [df.astype({col: np.object for col in df.select_dtypes(['category'])}) for df in dfs]
        if len(dfs) == 0:
            return None
        df = pd.concat(dfs, sort=False)
        return df.astype({col: cls.fields[col].dtype for col in df.columns if col in cls.fields})

    def base_score(self):
        if self['exit_code'] == 0:
            return self['saturation_iterations']
        return np.nan


class Result(process.Result):
    def __init__(self, process_result, symbols=None, clauses=None):
        super().__init__(**process_result.__dict__)
        self.symbols = symbols
        self.clauses = clauses

    def save(self, path):
        super().save(path)
        if self.symbols is not None:
            save_symbols(self.symbols, os.path.join(path, 'symbols.csv'))
        if self.clauses is not None:
            save_clauses(self.clauses, os.path.join(path, 'clauses.json'))

    @classmethod
    def load(cls, path):
        process_result = super().load(path)
        try:
            symbols = load_symbols(os.path.join(path, 'symbols.csv'))
        except FileNotFoundError:
            logging.debug('Ignoring missing symbols.csv')
            symbols = None
        try:
            clauses = load_clauses(os.path.join(path, 'clauses.json'))
        except FileNotFoundError:
            logging.debug('Ignoring missing clauses.json')
            clauses = None
        return Result(process_result, symbols, clauses)

    def get_symbols(self, symbol_type=None):
        symbols = self.symbols
        if symbol_type is not None and symbols is not None:
            assert symbol_type in ['predicate', 'function']
            if symbol_type == 'predicate':
                symbols = symbols[symbols.index.get_level_values('isFunction') == False]
            if symbol_type == 'function':
                symbols = symbols[symbols.index.get_level_values('isFunction') == True]
            symbols.index = symbols.index.droplevel(0)
        return symbols


class Problem:
    """Configuration of a Vampire run on a problem."""

    def __init__(self, path, base_options=None, timeout=None):
        self.path = path
        if base_options is None:
            self.base_options = dict()
        else:
            self.base_options = base_options
        self.timeout = timeout

    def __str__(self):
        return self.path

    def __repr__(self):
        return f'{type(self).__name__}({self.path})'

    def __getstate__(self):
        return {key: self.__dict__[key] for key in ['path', 'base_options', 'timeout']}

    def name(self):
        return str(self).replace('/', '_')

    @methodtools.lru_cache(maxsize=1)
    def get_embedding(self):
        return np.asarray([len(self.get_clauses())], dtype=np.uint)

    @methodtools.lru_cache(maxsize=2)
    def get_all_symbol_embeddings(self, symbol_type):
        assert symbol_type == 'function' or np.all(~self.get_symbols(symbol_type)[['skolem']])
        assert np.all(~self.get_symbols(symbol_type)[['inductionSkolem']])
        # TODO: Only include 'skolem' in function symbol representations.
        return self.get_symbols(symbol_type)[
            ['arity', 'usageCnt', 'unitUsageCnt', 'inGoal', 'inUnit', 'skolem']].to_numpy(dtype=np.uint)

    def get_symbols_embedding(self, symbol_type, symbol_indexes):
        return self.get_all_symbol_embeddings(symbol_type)[symbol_indexes]

    def get_predicates(self):
        return self.get_symbols(symbol_type='predicate')

    def get_functions(self):
        return self.get_symbols(symbol_type='function')

    @methodtools.lru_cache(maxsize=2)
    def get_symbols(self, symbol_type=None):
        symbols = self.get_successful_clausify_result().get_symbols(symbol_type)
        if symbols is None:
            raise RuntimeError('Fetching symbols failed.')
        return symbols

    def get_clauses(self):
        return self.get_successful_clausify_result().clauses

    def get_successful_clausify_result(self):
        result = self.get_clausify_execution().result
        if result.exit_code != 0:
            raise RuntimeError('Clausify run failed.',
                               {'status': result.status, 'exit_code': result.exit_code, 'stdout': result.stdout})
        return result

    @methodtools.lru_cache(maxsize=1)
    def get_clausify_execution(self):
        return self.get_execution(mode='clausify')

    def get_execution(self, mode='vampire', precedences=None):
        return workspace.get_execution(self.get_configuration(mode, precedences))

    def get_configuration(self, mode='vampire', precedences=None):
        assert self.base_options is not None
        base_options = self.base_options.copy()
        base_options.update({'mode': mode})
        return Configuration(self.path, base_options=base_options, precedences=precedences, timeout=self.timeout)

    def get_configuration_path(self, mode='vampire', precedences=None):
        return workspace.get_configuration_path(self.get_configuration(mode=mode, precedences=precedences))

    def solve_with_random_precedences(self, solve_count=1, random_predicates=False, random_functions=False,
                                      reverse=False, progress=True):
        # TODO: Parallelize.
        # TODO: Consider exhausting all permutations if they fit in `namespace.solve_runs`. Watch out for imbalance in distribution when learning from all problems.
        # TODO: Allow solving for reverse precedences automatically.
        if solve_count > 1 and not random_predicates and not random_functions:
            warnings.warn('Multiple solve runs without randomized precedences')
        seed_count = solve_count
        if reverse:
            seed_count = solve_count // 2
            solve_count = seed_count * 2
        with tqdm(total=solve_count, desc=self.path, unit='run', disable=not progress) as t:
            for seed in range(seed_count):
                precedences = dict()
                if random_predicates:
                    precedences['predicate_precedence'] = self.random_predicate_precedence(seed)
                if random_functions:
                    precedences['function_precedence'] = self.random_function_precedence(seed)
                execution = self.get_execution(precedences=precedences)
                assert execution.path == self.get_configuration_path(precedences=precedences)
                yield execution
                t.update()
                if reverse:
                    reversed_precedences = dict()
                    if random_predicates:
                        head = np.asarray([0], dtype=np.uint)
                        tail = precedences['predicate_precedence'][:0:-1]
                        reversed_precedences['predicate_precedence'] = np.concatenate((head, tail))
                    if random_functions:
                        reversed_precedences['function_precedence'] = precedences['function_precedence'][::-1]
                    execution = self.get_execution(precedences=reversed_precedences)
                    assert execution.path == self.get_configuration_path(precedences=reversed_precedences)
                    yield execution
                    t.update()

    def random_predicate_precedence(self, seed=None):
        # The equality symbol should be placed first in all the predicate precedences.
        # We assume that the equality symbol has the index 0, which is a convention in Vampire.
        head = np.asarray([0], dtype=np.uint)
        length = len(self.get_predicates())
        tail = np.random.RandomState((0, seed)).permutation(np.arange(1, length, dtype=np.uint))
        return np.concatenate((head, tail))

    def random_function_precedence(self, seed=None):
        length = len(self.get_functions())
        return np.random.RandomState((1, seed)).permutation(np.arange(length, dtype=np.uint))


class Workspace:
    def __init__(self, path=None, program='vampire', problem_dir=None, include_dir=None, scratch_dir=None,
                 never_load=False, never_run=False, result_is_ok_to_load=None, check_hash_collisions=True):
        self.path = path
        self.program = program
        self.problem_dir = problem_dir
        self.include_dir = include_dir
        self.scratch_dir = scratch_dir
        self.never_load = never_load
        self.never_run = never_run
        if result_is_ok_to_load is None:
            self.result_is_ok_to_load = lambda result: True
        else:
            self.result_is_ok_to_load = result_is_ok_to_load
        self.check_hash_collisions = check_hash_collisions
        self.cache_info = collections.Counter()

    def __str__(self):
        return self.path

    def get_execution(self, configuration):
        return Execution(configuration, *self.load_or_run(configuration))

    def load_or_run(self, configuration):
        if self.path is None:
            logging.debug('Running because result caching is disabled.')
            self.cache_info['misses'] += 1
            return self.run(configuration), None
        configuration_path = self.get_configuration_path(configuration)
        logging.debug({'problem': configuration.problem_path, 'configuration_path': configuration_path})
        try:
            result = self.load(configuration, configuration_path)
            self.cache_info['hits'] += 1
            return result, configuration_path
        except (FileNotFoundError, RuntimeError, KeyError):
            # KeyError may occur when 'configuration.json' is missing a required configuration property.
            self.cache_info['misses'] += 1
            if self.never_run:
                raise RuntimeError('Loading failed.')
            logging.debug('Loading failed. Running.', exc_info=True)
            configuration.save(os.path.join(configuration_path, 'configuration.json'))
            result = self.run(configuration, configuration_path=configuration_path)
            result.save(configuration_path)
            return result, configuration_path

    def run(self, configuration, configuration_path=None):
        return configuration.run(program=self.program, problem_dir=self.problem_dir, include_dir=self.include_dir,
                                 scratch_dir=self.scratch_dir, configuration_dir=configuration_path)

    def get_configuration_path(self, configuration):
        m = hashlib.md5()
        m.update(configuration.tobytes())
        assert self.path is not None
        return os.path.join(self.path, 'md5', m.hexdigest())

    def load(self, configuration, configuration_path):
        if self.path is None:
            raise RuntimeError('Result caching is disabled.')
        if self.never_load:
            raise RuntimeError('Result loading is disabled.')
        if self.check_hash_collisions:
            configuration_other = Configuration.load(os.path.join(configuration_path, 'configuration.json'))
            if configuration_other != configuration:
                self.cache_info['collisions'] += 1
                raise RuntimeError('Configuration mismatch. Hash collision.')
        result = Result.load(configuration_path)
        if not self.result_is_ok_to_load(result):
            raise RuntimeError('Loaded result is not ok.')
        return result


workspace = Workspace()


@contextlib.contextmanager
def workspace_context(**kwargs):
    global workspace
    old = copy.copy(workspace)
    workspace.__dict__.update(kwargs)
    try:
        yield workspace
    finally:
        workspace = old


class Configuration:
    default_options = {
        'encode': 'on',
        'statistics': 'full',
        'time_statistics': 'on',
        'proof': 'off',
        'literal_comparison_mode': 'predicate',
        'symbol_precedence': 'frequency',
        'saturation_algorithm': 'discount',
        'age_weight_ratio': '10',
        'avatar': 'off'
    }

    def __init__(self, problem_path, base_options=None, precedences=None, timeout=None):
        self.problem_path = problem_path
        self.base_options = base_options
        self.precedences = dict()
        if precedences is not None:
            for key, value in precedences.items():
                if value is None:
                    continue
                self.precedences[key] = np.asarray(value, dtype=np.uint)
        self.timeout = timeout

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, Configuration):
            return False
        if self.problem_path != other.problem_path:
            return False
        if self.base_options != other.base_options:
            return False
        if self.timeout != other.timeout:
            return False
        if self.precedences.keys() != other.precedences.keys():
            return False
        for key in self.precedences.keys():
            if not np.array_equal(self.precedences[key], other.precedences[key]):
                return False
        return True

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        json.dump({
            'problem': self.problem_path,
            'base_options': self.base_options,
            'precedences': self.precedences,
            'timeout': self.timeout
        }, open(filename, 'w'), default=self.default, indent=4)

    @staticmethod
    def load(filename):
        data = json.load(open(filename))
        return Configuration(data['problem'], data['base_options'], data['precedences'], data['timeout'])

    @property
    def is_clausify(self):
        return 'mode' in self.base_options and self.base_options['mode'] == 'clausify'

    @contextlib.contextmanager
    def options(self, include_dir=None, scratch_dir=None):
        res = self.default_options.copy()
        res.update(self.base_options)
        if include_dir is not None:
            res.update({'include': include_dir})
        if scratch_dir is not None:
            os.makedirs(scratch_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_dir) as temp_dir:
            if len(self.precedences) >= 1:
                precedence_dir = os.path.join(temp_dir, 'precedence')
                os.makedirs(precedence_dir, exist_ok=True)
                for name, value in self.precedences.items():
                    precedence_file_path = os.path.join(precedence_dir, f'{name}.csv')
                    value.tofile(precedence_file_path, sep=',')
                    res.update({name: precedence_file_path})
            if self.is_clausify:
                # Vampire expects the output directory to exist.
                assert os.path.isdir(temp_dir)
                res.update({
                    'symbols_csv_output': os.path.join(temp_dir, 'symbols.csv'),
                    'clauses_json_output': os.path.join(temp_dir, 'clauses.json')
                })
            yield res

    def run(self, program='vampire', problem_dir='', include_dir=None, scratch_dir=None, configuration_dir=None):
        with self.options(include_dir=include_dir, scratch_dir=scratch_dir) as options:
            args = list(itertools.chain([program],
                                        *((f'--{name}', str(value)) for (name, value) in options.items()),
                                        [os.path.join(problem_dir, self.problem_path)]))
            if configuration_dir is not None:
                open(os.path.join(configuration_dir, 'run.sh'), 'w').write(' '.join(args))
            result = Result(process.run(args, timeout=self.timeout))
            if self.is_clausify and result.exit_code == 0:
                result.symbols = load_symbols(options['symbols_csv_output'])
                result.clauses = load_clauses(options['clauses_json_output'])
        return result

    def tobytes(self):
        # Function name inspired by `numpy.ndarray.tobytes`
        return json.dumps(self.__dict__, default=self.default).encode('utf-8')

    @staticmethod
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()


def load_symbols(file):
    # The column 'name' may contain single quoted strings.
    # See http://www.tptp.org/TPTP/SyntaxBNF.html
    # <fof_plain_term> ::= <functor> ::= <atomic_word> ::= <single_quoted> ::= <single_quote> ::: [']
    # We assume that there are no NAs in the symbols CSV table.
    # Note that for example in SWV478+2.p there is a symbol called 'null' that may alias with the NA filtering
    # (its name being misinterpreted as a missing value).
    return pd.read_csv(file, index_col=['isFunction', 'id'], quotechar='\'', escapechar='\\', na_filter=False,
                       dtype={
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


def save_symbols(symbols, file):
    symbols.to_csv(file, quotechar='\'', escapechar='\\')


def load_clauses(file):
    return json.load(open(file))


def save_clauses(clauses, file):
    json.dump(clauses, open(file, 'w'))
