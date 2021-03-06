#!/usr/bin/env python3

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections import namedtuple
from copy import deepcopy
from itertools import chain

import numpy as np
import pandas as pd

from utils import makedirs_open, get_consistent, len_robust
from . import extractor
from .SymbolPrecedence import SymbolPrecedence
from .vampire import load_symbols


class Run:
    base_name_run = 'job.json'
    base_name_symbols = 'symbols.csv'
    base_name_clauses = 'clauses.json'
    base_name_stdout = 'stdout.txt'
    base_name_stderr = 'stderr.txt'

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

    def __init__(self, program, base_options=None, timeout=None, output_dir=None, problem_base_path=None,
                 scratch_dir=None, run_policy='interrupted'):
        # Run configuration:
        self.program = program
        self.base_options = self.default_options.copy()
        self.base_options.update(base_options)
        self.timeout = timeout
        self.output_dir = output_dir
        self.output_dir_base = output_dir
        self.problem_base_path = problem_base_path
        self.scratch_dir = scratch_dir
        self.run_policy = run_policy
        self.problem_path = None
        self.precedences = dict()

        # Result properties:
        self.time_elapsed = None
        self.status = None
        self.exit_code = None

        # Expensive result content - loaded on demand
        self.stdout = None
        self.stderr = None
        self.symbols = None
        self.clauses = None

        self.score = None

    def __str__(self):
        return self.output_dir

    @property
    def is_clausify(self):
        return 'mode' in self.base_options and self.base_options['mode'] == 'clausify'

    def spawn(self, problem_path=None, output_dir_rel=None, base_options=None, precedences=None):
        instance = deepcopy(self)
        if problem_path is not None:
            instance.problem_path = problem_path
        instance.output_dir = os.path.join(instance.output_dir, output_dir_rel)
        if base_options is not None:
            instance.base_options.update(base_options)
        if precedences is not None:
            instance.precedences.update(precedences)
        return instance

    def load_or_run(self):
        assert self.run_policy in ['none', 'interrupted', 'failed', 'all']
        if self.run_policy != 'all':
            # If run_policy is 'always', we always run, irrespective of the presence or content of the previous result.
            try:
                self.load_shallow()
                assert self.run_policy != 'all'
                # See definitions of Vampire exit codes in `vampire/Lib/System.hpp`.
                # Filip Bartek: I have observed the exit codes 0, 1, 3, -9 and -11.
                # Out of these, only 0 and 1 seem to guarantee a completed execution.
                # See also: https://www.nsnam.org/wiki/HOWTO_understand_and_find_cause_of_exited_with_code_-11_errors
                if (self.run_policy == 'none' or
                        (self.run_policy == 'interrupted' and self.exit_code in [0, 1]) or
                        (self.run_policy == 'failed' and self.exit_code == 0)):
                    # We are satisfied with the loaded result.
                    return True
            except (FileNotFoundError, RuntimeError, json.decoder.JSONDecodeError) as e:
                if self.run_policy == 'none':
                    raise RuntimeError(f'Failed to load "{self}"') from e
                logging.debug(f'Failed to load "{self}"', exc_info=True)
        try:
            assert self.run_policy != 'none'
            self.run()
            self.save()
        except Exception as e:
            raise RuntimeError(f'Failed to run "{self}"') from e
        return False

    def run(self):
        with self.current_output_directory() as vampire_output_dir:
            time_start = time.time()
            try:
                # Vampire fails if the output directory does not exist.
                cp = subprocess.run(list(self.args(vampire_output_dir)), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=self.timeout, text=True)
                self.time_elapsed = time.time() - time_start
                self.status = 'completed'
                self.exit_code = cp.returncode
                self.stdout = cp.stdout
                self.stderr = cp.stderr
            except subprocess.TimeoutExpired as e:
                self.time_elapsed = time.time() - time_start
                self.status = 'timeout_expired'
                self.exit_code = None
                self.stdout = e.stdout
                self.stderr = e.stderr
            finally:
                if self.exit_code != 0:
                    for base_name in [self.base_name_symbols, self.base_name_clauses]:
                        # Vampire may or may not have created the file.
                        try:
                            os.remove(os.path.join(self.output_dir, base_name))
                        except FileNotFoundError:
                            pass

    def args(self, current_output_dir):
        return chain([self.program],
                     *((f'--{name}', str(value)) for (name, value) in self.options(current_output_dir).items()),
                     [self.problem_path])

    def options(self, current_output_dir=None):
        if current_output_dir is None:
            current_output_dir = self.output_dir
        res = self.base_options.copy()
        for precedence in self.precedences.values():
            # `SymbolPrecedence.options()` saves the precedence into a file.
            res.update(precedence.options(self.output_dir))
        if self.is_clausify:
            res.update({
                'symbols_csv_output': os.path.join(current_output_dir, self.base_name_symbols),
                'clauses_json_output': os.path.join(current_output_dir, self.base_name_clauses)
            })
        return res

    @contextlib.contextmanager
    def current_output_directory(self):
        if self.scratch_dir is not None and os.path.abspath(self.scratch_dir) != os.path.abspath(self.output_dir):
            os.makedirs(self.scratch_dir, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.scratch_dir) as temp_dir:
                try:
                    yield temp_dir
                finally:
                    if self.exit_code == 0:
                        os.makedirs(self.output_dir, exist_ok=True)
                        for base_name in [self.base_name_symbols, self.base_name_clauses]:
                            try:
                                shutil.move(os.path.join(temp_dir, base_name),
                                            os.path.join(self.output_dir, base_name))
                            except FileNotFoundError:
                                pass
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            yield self.output_dir

    def save(self):
        map(lambda precedence: precedence.save(), self.precedences.values())
        if self.stdout is not None:
            with makedirs_open(self.output_dir, self.base_name_stdout, 'w') as fp:
                fp.write(self.stdout)
        if self.stderr is not None:
            with makedirs_open(self.output_dir, self.base_name_stderr, 'w') as fp:
                fp.write(self.stderr)
        with makedirs_open(self.output_dir, self.base_name_run, 'w') as fp:
            json.dump(self.as_dict(), fp, indent=4)

    def as_dict(self):
        res = {
            'program': self.program,
            'base_options': self.base_options,
            'command': ' '.join(self.args(self.output_dir)),
            'timeout': self.timeout,
            'problem': self.problem_path,
            'precedence': {name: precedence.file_base_name for name, precedence in self.precedences.items()},
            'scratch': self.scratch_dir,
            'time_elapsed': self.time_elapsed,
            'status': self.status,
            'exit_code': self.exit_code,
            'stdout': self.base_name_stdout,
            'stderr': self.base_name_stderr
        }
        if self.is_clausify:
            res.update({
                'symbols': self.base_name_symbols,
                'clauses': self.base_name_clauses
            })
        return res

    def load_shallow(self):
        """
        Raises an exception if a loaded value is inconsistent with a pre-set value.

        Loads the symbol precedences namely to check their consistency.
        """
        with open(os.path.join(self.output_dir, self.base_name_run)) as fp:
            data = json.load(fp)

        # Configuration:
        self.program = self.get_consistent(data, 'program', self.program)
        self.base_options = self.get_consistent(data, 'base_options', self.base_options)
        # We ignore `data['command']`.
        self.timeout = self.get_consistent(data, 'timeout', self.timeout)
        self.problem_path = self.get_consistent(data, 'problem', self.problem_path)
        assert self.precedences is not None
        if 'precedence' in data:
            if set(data['precedence'].keys()) != set(self.precedences.keys()):
                raise RuntimeError('The stored run contains different precedences than the configuration.')
            for symbol_type, file in data['precedence'].items():
                other = SymbolPrecedence(symbol_type=symbol_type, file_base_name=file)
                other.load(self.output_dir)
                assert symbol_type in self.precedences
                if other != self.precedences[symbol_type]:
                    raise RuntimeError('Precedence value mismatch.')
        elif len(self.precedences) > 0:
            raise RuntimeError('The stored run contains no precedence.')

        # Result:
        if 'time_elapsed' in data:
            # We do not check time_elapsed for consistency.
            self.time_elapsed = data['time_elapsed']
        self.status = self.get_consistent(data, 'status', self.status)
        self.exit_code = self.get_consistent(data, 'exit_code', self.exit_code)

        if 'stdout' in data and data['stdout'] != self.base_name_stdout:
            raise RuntimeError('Stdout filename mismatch.')
        if 'stderr' in data and data['stderr'] != self.base_name_stderr:
            raise RuntimeError('Stderr filename mismatch.')
        if 'symbols' in data and data['symbols'] != self.base_name_symbols:
            raise RuntimeError('Symbols filename mismatch.')
        if 'clauses' in data and data['clauses'] != self.base_name_clauses:
            raise RuntimeError('Clauses filename mismatch.')

    def get_consistent(self, d, key, value=None):
        return get_consistent(d, key, value, override=self.run_policy == 'none')

    def unload(self):
        self.stdout = None
        self.stderr = None
        self.symbols = None
        self.clauses = None

    def load_stdout(self):
        if self.stdout is not None:
            return
        with open(os.path.join(self.output_dir, self.base_name_stdout)) as stdout_file:
            self.stdout = stdout_file.read()

    def get_stdout(self):
        self.load_stdout()
        return self.stdout

    def load_symbols(self):
        if self.symbols is not None:
            return
        self.symbols = load_symbols(os.path.join(self.output_dir, self.base_name_symbols))

    def get_symbols(self, symbol_type=None):
        try:
            self.load_symbols()
        except FileNotFoundError:
            return None
        if symbol_type is None:
            return self.symbols
        result = self.symbols[
            self.symbols.index.get_level_values('isFunction') == SymbolPrecedence.symbol_types[symbol_type].is_function]
        result.index = result.index.droplevel(0)
        return result

    def get_symbol_count(self, symbol_type=None):
        symbols = self.get_symbols(symbol_type)
        if symbols is None:
            raise RuntimeError(
                f'This Vampire run does not have {symbol_type} symbols. Cannot determine the symbol count.')
        return len(symbols)

    def load_clauses(self):
        if self.clauses is not None:
            return
        with open(os.path.join(self.output_dir, self.base_name_clauses)) as fp:
            self.clauses = json.load(fp)

    def get_clauses(self):
        try:
            self.load_clauses()
        except FileNotFoundError:
            return None
        return self.clauses

    def saturation_iterations(self):
        return extractor.saturation_iterations(self.get_stdout())

    def random_precedence(self, symbol_type, seed=None):
        return SymbolPrecedence.random(symbol_type, self.get_symbol_count(symbol_type), seed)

    def __getitem__(self, key):
        return self.fields[key].extract(self)

    Field = namedtuple('Field', ['extract', 'dtype'])

    # TODO: Add vampire job id.
    fields = {
        'output_dir': Field(lambda job: os.path.relpath(job.output_dir, job.output_dir_base), 'object'),
        'problem_path': Field(lambda job: os.path.relpath(job.problem_path, job.problem_base_path), 'object'),
        'status': Field(lambda job: job.status, pd.CategoricalDtype()),
        'exit_code': Field(lambda job: job.exit_code, pd.CategoricalDtype()),
        'termination_reason': Field(lambda job: extractor.termination_reason(job.get_stdout()), pd.CategoricalDtype()),
        'termination_phase': Field(lambda job: extractor.termination_phase(job.get_stdout()), pd.CategoricalDtype()),
        'time_elapsed_process': Field(lambda job: job.time_elapsed, np.float),
        'time_elapsed_vampire': Field(lambda job: extractor.time_elapsed(job.get_stdout()), np.float),
        'memory_used': Field(lambda job: extractor.memory_used(job.get_stdout()), pd.UInt64Dtype()),
        'saturation_iterations': Field(lambda job: job.saturation_iterations(), pd.UInt64Dtype()),
        'predicates_count': Field(lambda job: len_robust(job.get_symbols('predicate')), pd.UInt64Dtype()),
        'functions_count': Field(lambda job: len_robust(job.get_symbols('function')), pd.UInt64Dtype()),
        'clauses_count': Field(lambda job: len_robust(job.get_clauses()), pd.UInt64Dtype()),
        'score': Field(lambda job: job.score, np.float)
    }

    field_names_clausify = ['output_dir', 'problem_path', 'status', 'exit_code', 'termination_reason',
                            'termination_phase', 'time_elapsed_process', 'time_elapsed_vampire', 'memory_used',
                            'predicates_count', 'functions_count', 'clauses_count']
    field_names_solve = ['output_dir', 'problem_path', 'status', 'exit_code', 'termination_reason', 'termination_phase',
                         'time_elapsed_process', 'time_elapsed_vampire', 'memory_used', 'saturation_iterations',
                         'score']
