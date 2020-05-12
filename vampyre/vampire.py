#!/usr/bin/env python3

import collections
import contextlib
import copy
import functools
import hashlib
import itertools
import json
import logging
import os
import tempfile
import warnings

import appdirs
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import ProgressBar
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
        'problem_path': Field(lambda job: job.configuration.problem_path, 'object'),
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
        'clauses_count': Field(lambda job: len_robust(job.result.clauses), pd.UInt64Dtype()),
        'predicate_precedence': Field(lambda job: extractor.predicate_precedence(job.result.stdout), 'object'),
        'function_precedence': Field(lambda job: extractor.function_precedence(job.result.stdout), 'object')
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
        dfs = [df.astype({col: np.object for col in df.select_dtypes(['category'])}) for df in dfs if df is not None]
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
            symbols = None
        try:
            clauses = load_clauses(os.path.join(path, 'clauses.json'))
        except FileNotFoundError:
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

    def __init__(self, path, vampire_options=None, timeout=None):
        self.path = path
        if vampire_options is None:
            vampire_options = dict()
        self.vampire_options = vampire_options
        self.timeout = timeout

    def __str__(self):
        return self.path

    def __repr__(self):
        return f'{type(self).__name__}({self.path})'

    def __getstate__(self):
        return {key: self.__dict__[key] for key in ['path', 'vampire_options', 'timeout']}

    def name(self):
        return str(self).replace('/', '_')

    dtype_embedding = np.uint

    @staticmethod
    def get_embedding_column_names():
        return ['clauses_count']

    @functools.lru_cache(maxsize=1)
    def get_embedding(self):
        try:
            clauses_count = len(self.get_clauses())
        except TypeError:
            clauses_count = 1024 * 1024
            logging.warning(f'Failed to determine number of clauses. Defaulting to {clauses_count}.')
        return np.asarray([clauses_count], dtype=self.dtype_embedding)

    @staticmethod
    def get_symbol_embedding_column_names(symbol_type):
        res = ['arity', 'usageCnt', 'unitUsageCnt', 'inGoal', 'inUnit', 'introduced']
        return res

    @functools.lru_cache(maxsize=2)
    def get_all_symbol_embeddings(self, symbol_type):
        assert symbol_type == 'function' or np.all(~self.get_symbols(symbol_type)[['skolem']])
        assert np.all(~self.get_symbols(symbol_type)[['inductionSkolem']])
        return self.get_symbols(symbol_type)[self.get_symbol_embedding_column_names(symbol_type)].to_numpy(
            dtype=self.dtype_embedding)

    def get_symbols_embedding(self, symbol_type, symbol_indexes):
        return self.get_all_symbol_embeddings(symbol_type)[symbol_indexes]

    def get_symbol_pair_embeddings(self, symbol_type, symbol_indexes):
        n_samples = len(symbol_indexes)
        assert symbol_indexes.shape == (n_samples, 2)
        problem_embeddings = np.asarray(self.get_embedding()).reshape(1, -1).repeat(n_samples, axis=0)
        symbol_embeddings = self.get_symbols_embedding(symbol_type, symbol_indexes.flatten()).reshape(n_samples, 2, -1)
        return np.concatenate((problem_embeddings, symbol_embeddings[:, 0], symbol_embeddings[:, 1]), axis=1)

    @classmethod
    def get_symbol_pair_embedding_column_names(cls, symbol_type):
        return np.concatenate((cls.get_embedding_column_names(),
                               [f'l.{s}' for s in cls.get_symbol_embedding_column_names(symbol_type)],
                               [f'r.{s}' for s in cls.get_symbol_embedding_column_names(symbol_type)]))

    def get_predicates(self):
        return self.get_symbols(symbol_type='predicate')

    def get_functions(self):
        return self.get_symbols(symbol_type='function')

    @functools.lru_cache(maxsize=2)
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
            raise RuntimeError('Clausify run failed.', {'status': result.status, 'exit_code': result.exit_code})
        return result

    @functools.lru_cache(maxsize=1)
    def get_clausify_execution(self):
        return self.get_execution(mode='clausify')

    def get_execution(self, mode='vampire', precedences=None):
        return workspace.get_execution(self.get_configuration(mode, precedences))

    def get_configuration(self, mode='vampire', precedences=None):
        assert self.vampire_options is not None
        current_options = self.vampire_options.copy()
        current_options.update({'mode': mode})
        return Configuration(self.path, options=current_options, precedences=precedences, timeout=self.timeout)

    def get_configuration_path(self, mode='vampire', precedences=None):
        return workspace.get_configuration_path(self.get_configuration(mode=mode, precedences=precedences))

    def solve_with_random_precedences(self, solve_count=1, random_predicates=False, random_functions=False,
                                      reverse=False, progress_bar=True):
        # TODO: Consider exhausting all permutations if they fit in `namespace.solve_runs`. Watch out for imbalance in distribution when learning from all problems.
        if solve_count > 1 and not random_predicates and not random_functions:
            warnings.warn('Multiple solve runs without randomized precedences')
        seed_count = solve_count
        if reverse:
            seed_count = solve_count // 2
        seeds = range(seed_count)
        r = Parallel()(delayed(self.solve_one_seed)(seed, random_predicates, random_functions, reverse) for seed in
                       ProgressBar(seeds, desc=f'Solving {self.path} with random precedences', unit='seed',
                                   disable=not progress_bar))
        return list(itertools.chain(*r))

    def solve_one_seed(self, seed, random_predicates=False, random_functions=False, reverse=False):
        precedences = self.random_precedences(random_predicates, random_functions, seed)
        execution = self.get_execution(precedences=precedences)
        assert execution.path == self.get_configuration_path(precedences=precedences)
        result = [execution]
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
            result.append(execution)
        return result

    def random_precedences(self, predicate, function, seed=None):
        precedences = dict()
        if predicate:
            precedences['predicate_precedence'] = self.random_predicate_precedence(seed)
        if function:
            precedences['function_precedence'] = self.random_function_precedence(seed)
        return precedences

    dtype_precedence = np.uint32

    def random_predicate_precedence(self, seed=None):
        # The equality symbol should be placed first in all the predicate precedences.
        # We assume that the equality symbol has the index 0, which is a convention in Vampire.
        head = np.asarray([0], dtype=np.uint)
        length = len(self.get_predicates())
        if seed is not None:
            # Salt the seed specifically for predicates.
            seed = (0, seed)
        tail = np.random.RandomState(seed).permutation(np.arange(1, length, dtype=self.dtype_precedence))
        return np.concatenate((head, tail))

    def random_function_precedence(self, seed=None):
        length = len(self.get_functions())
        if seed is not None:
            # Salt the seed specifically for functions.
            seed = (1, seed)
        return np.random.RandomState(seed).permutation(np.arange(length, dtype=self.dtype_precedence))


class Workspace:
    def __init__(self, path=None, program='vampire', problem_dir=None, include_dir=None, scratch_dir=None,
                 never_load=False, never_run=False, result_is_ok_to_load=None, check_hash_collisions=True):
        if path is None:
            path = os.path.join(appdirs.user_cache_dir('vampire_ml'), 'workspace')
        self.path = path
        self.program = program
        self.problem_dir = problem_dir
        self.include_dir = include_dir
        self.scratch_dir = scratch_dir
        if self.scratch_dir is None:
            try:
                self.scratch_dir = os.environ['SCRATCH']
                #logging.debug('Scratch dir set to $SCRATCH: %s', self.scratch_dir)
            except KeyError:
                #logging.debug('Set $SCRATCH to the path to scratch directory.')
                pass
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
        try:
            result = self.load(configuration, configuration_path)
            self.cache_info['hits'] += 1
        except (FileNotFoundError, RuntimeError, KeyError, json.JSONDecodeError) as e:
            # KeyError may occur when 'configuration.json' is missing a required configuration property.
            # JSONDecodeError occurs if the configuration file is malformed.
            self.cache_info['misses'] += 1
            if self.never_run:
                raise RuntimeError('Loading failed.') from e
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
        if configuration.mode == 'clausify':
            # Hack:
            # The following Vampire version has updated clausify output.
            # We salt the hash to recompute the previously cached results.
            m.update(json.dumps({'vampire_version': '926154f2193d876feed0b34b9aa421b93aa5e69b'}).encode('utf-8'))
        assert self.path is not None
        return os.path.join(self.path, 'vampire_runs', configuration.problem_path, configuration.mode, m.hexdigest())

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
    def __init__(self, problem_path, options, precedences=None, timeout=None):
        self.problem_path = problem_path
        self.options = options
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
        if self.options != other.options:
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
            'options': self.options,
            'precedences': self.precedences,
            'timeout': self.timeout
        }, open(filename, 'w'), default=self.default, indent=4)

    @staticmethod
    def load(filename):
        data = json.load(open(filename))
        return Configuration(data['problem'], data['options'], data['precedences'], data['timeout'])

    @property
    def is_clausify(self):
        return self.mode == 'clausify'

    @property
    def mode(self):
        if 'mode' in self.options:
            return self.options['mode']
        return 'vampire'

    @contextlib.contextmanager
    def instantiate_options(self, include_dir=None, scratch_dir=None):
        res = self.options.copy()
        if include_dir is not None:
            res.update({'include': include_dir})
        if scratch_dir is not None:
            os.makedirs(scratch_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vampire_ml_', dir=scratch_dir) as temp_dir:
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
        with self.instantiate_options(include_dir=include_dir, scratch_dir=scratch_dir) as options:
            assert problem_dir is not None
            args = list(itertools.chain([program],
                                        *((f'--{name}', str(value)) for (name, value) in options.items()),
                                        [os.path.join(problem_dir, self.problem_path)]))
            if configuration_dir is not None:
                open(os.path.join(configuration_dir, 'run.sh'), 'w').write(' '.join(args))
            capture_stdout = 'mode' not in options or options['mode'] != 'clausify'
            result = Result(process.run(args, timeout=self.timeout, capture_stdout=capture_stdout))
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
    logging.debug(f'Loading {file} of size {os.path.getsize(file)}.')
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
    file_size = os.path.getsize(file)
    logging.debug(f'Loading {file} of size {file_size}.')
    return json.load(open(file))


def save_clauses(clauses, file):
    json.dump(clauses, open(file, 'w'))
