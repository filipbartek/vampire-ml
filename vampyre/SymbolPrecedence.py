#!/usr/bin/env python3.7

import os
from collections import namedtuple

import numpy as np


class SymbolPrecedence:
    dtype = np.uint
    file_sep = ','
    SymbolType = namedtuple('SymbolType', ['file_base_name', 'option_name', 'is_function'])
    symbol_types = {
        'predicate': SymbolType('predicate_precedence.csv', 'predicate_precedence', False),
        'function': SymbolType('function_precedence.csv', 'function_precedence', True)
    }

    def __init__(self, symbol_type, value=None, file_base_name=None):
        if symbol_type not in self.symbol_types:
            raise RuntimeError(f'Unsupported symbol precedence symbol type: {symbol_type}')
        self.option_name = self.symbol_types[symbol_type].option_name
        assert value is None or (type(value) == np.ndarray and value.dtype == self.dtype)
        self.value = value
        self.file_base_name = file_base_name
        if self.file_base_name is None:
            self.file_base_name = self.symbol_types[symbol_type].file_base_name

    def __str__(self):
        return ' '.join((self.option_name, str(self.value)))

    def __eq__(self, other):
        if not isinstance(other, SymbolPrecedence):
            return False
        if other.option_name != self.option_name:
            return False
        if other.file_base_name != self.file_base_name:
            return False
        if not np.array_equal(other.value, self.value):
            return False
        return True

    def __getitem__(self, key):
        return self.value[key]

    def options(self, output_dir):
        self.save(output_dir)
        return {self.option_name: self.path_abs(output_dir)}

    def save(self, output_dir):
        assert type(self.value) == np.ndarray and self.value.dtype == self.dtype
        os.makedirs(output_dir, exist_ok=True)
        self.value.tofile(self.path_abs(output_dir), sep=self.file_sep)

    def load(self, output_dir):
        self.value = np.fromfile(self.path_abs(output_dir), dtype=self.dtype, sep=self.file_sep)

    def path_abs(self, output_dir):
        return os.path.join(output_dir, self.file_base_name)

    @classmethod
    def random(cls, symbol_type, length, seed=None, equality_first=True):
        if symbol_type == 'predicate' and equality_first:
            # The equality symbol should be placed first in all the predicate precedences.
            # We assume that the equality symbol has the index 0, which is a convention in Vampire.
            head = np.asarray([0], dtype=cls.dtype)
            tail = np.random.RandomState(seed).permutation(np.arange(1, length, dtype=cls.dtype))
            value = np.concatenate((head, tail))
            assert value.dtype == cls.dtype
        else:
            value = np.random.RandomState(seed).permutation(length).astype(cls.dtype)
        return cls(symbol_type, value=value)
