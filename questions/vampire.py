import copy
import itertools
import json
import logging
import os
import re
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

from questions import config
from questions import process
from questions import symbols

log = logging.getLogger(__name__)
supported_precedence_names = ['predicate', 'function']


class Result(process.Result):
    def __init__(self, process_result, result_symbols, clauses):
        log.debug(process_result)
        super().__init__(**process_result.__dict__)
        self.symbols = result_symbols
        self.clauses = clauses

    pd_dtypes = {
        **process.Result.pd_dtypes,
        'time_elapsed_vampire': float,
        'saturation_iterations': pd.UInt32Dtype(),
        'memory_used': pd.UInt32Dtype()
    }

    def symbols_of_type(self, symbol_type):
        return symbols.symbols_of_type(self.symbols, symbol_type)

    @property
    def saturation_iterations(self):
        try:
            return int(re.search(r'^% Main loop iterations started: (\d+)$', self.stdout, re.MULTILINE)[1])
        except TypeError:
            return None

    @property
    def memory_used(self):
        try:
            return int(re.search(r'^% Memory used \[KB\]: (\d+)$', self.stdout, re.MULTILINE)[1])
        except TypeError:
            return None

    @property
    def time_elapsed_vampire(self):
        try:
            return float(re.search(r'^% Time elapsed: (\d+\.\d+) s$', self.stdout, re.MULTILINE)[1])
        except TypeError:
            return None


def call(problem, options=None, timeout=None, precedences=None, get_symbols=False, get_clauses=False, get_stdout=True,
         get_stderr=True):
    try:
        mode = options['mode']
    except KeyError:
        mode = None
    log.debug(f'Running Vampire. Problem: {problem}. Mode: {mode}.')
    result_symbols = None
    clauses = None
    with OptionManager(problem, base_options=options, precedences=precedences, get_symbols=get_symbols,
                       get_clauses=get_clauses) as option_manager:
        args_instantiated = option_manager.args()
        result = process.run(args_instantiated, timeout=timeout, capture_stdout=get_stdout, capture_stderr=get_stderr)
        if get_symbols:
            try:
                result_symbols = option_manager.symbols()
            except FileNotFoundError:
                pass
            except Exception as e:
                raise RuntimeError(f'Failed to load symbols of problem {problem}.') from e
        if get_clauses:
            try:
                clauses = option_manager.clauses()
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            except Exception as e:
                raise RuntimeError(f'Failed to load clauses of problem {problem}.') from e
    return Result(result, result_symbols, clauses)


def random_precedence(symbol_type, length, seed=None, dtype=np.uint32):
    if seed is not None:
        if not isinstance(seed, tuple):
            seed = (seed,)
        salt = supported_precedence_names.index(symbol_type)
        # Salt the seed for the given symbol type.
        seed = (salt, *seed)
    rng = np.random.RandomState(seed)
    if symbol_type == 'predicate':
        # The equality symbol should be placed first in all the predicate precedences.
        # We assume that the equality symbol has the index 0, which is a convention in Vampire.
        head = np.asarray([0], dtype=dtype)
        tail = rng.permutation(np.arange(1, length, dtype=dtype))
        res = np.concatenate((head, tail))
    else:
        res = rng.permutation(np.arange(length, dtype=dtype))
    assert res.dtype == dtype
    assert res.shape == (length,)
    return res


def program():
    try:
        return os.environ['VAMPIRE']
    except KeyError:
        pass
    return 'vampire'


def include_path():
    return config.tptp_path()


class OptionManager:
    def __init__(self, problem, base_options=None, precedences=None, get_symbols=False, get_clauses=False):
        self.problem = problem
        if base_options is None:
            base_options = {}
        self.base_options = base_options
        if precedences is None:
            precedences = {}
        assert set(precedences.keys()) <= set(supported_precedence_names)
        self.precedences = precedences
        self.symbols_enabled = get_symbols
        self.clauses_enabled = get_clauses
        self.temp_dir = None

    def enabled(self):
        return len(self.precedences) > 0 or self.symbols_enabled or self.clauses_enabled

    def __enter__(self):
        if self.enabled():
            self.temp_dir = tempfile.TemporaryDirectory(prefix=f'{config.program_name()}_', dir=config.scratch_dir())
            for name, precedence in self.precedences.items():
                if isinstance(precedence, tf.Tensor):
                    precedence = precedence.numpy()
                precedence.tofile(self.precedence_path(name), sep=',')
        return self

    def __exit__(self, type, value, traceback):
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None

    def options(self):
        res = copy.deepcopy(self.base_options)
        for name, precedence in self.precedences.items():
            assert name in supported_precedence_names
            res[f'{name}_precedence'] = self.precedence_path(name)
        if self.symbols_enabled:
            res['symbols_csv_output'] = self.symbols_path()
        if self.clauses_enabled:
            res['clauses_json_output'] = self.clauses_path()
        include = include_path()
        if include is not None:
            res['include'] = include
        return res

    def args(self):
        return list(itertools.chain([program(), config.full_problem_path(self.problem)],
                                    *((f'--{name}', str(value)) for (name, value) in self.options().items())))

    def symbols(self):
        return symbols.load(self.symbols_path())

    def clauses(self):
        return load_clauses(self.clauses_path())

    def precedence_path(self, name):
        return os.path.join(self.temp_dir.name, f'{name}_precedence.csv')

    def symbols_path(self):
        return os.path.join(self.temp_dir.name, 'symbols.csv')

    def clauses_path(self):
        return os.path.join(self.temp_dir.name, 'clauses.json')


def load_clauses(file):
    # Throws FileNotFoundError if `file` does not exist.
    log.debug(f'Loading {file} of size {os.path.getsize(file)}.')
    # Throws json.JSONDecodeError if the content is malformed.
    with open(file) as f:
        return json.load(f)


def save_clauses(clauses, file):
    with open(file, 'w') as f:
        json.dump(clauses, f)
