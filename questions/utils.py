import collections
import csv
import itertools
import os
import sys
import time
import warnings
from contextlib import contextmanager
from contextlib import suppress

import contexttimer
import numpy as np
import pandas as pd
import tensorflow as tf


def type_len(s):
    try:
        return f'{type(s).__name__}[{len(s)}]'
    except TypeError:
        return s


def instance_str(instance):
    params = ', '.join(f'{key}={type_len(value)}' for key, value in instance.__dict__.items())
    return f'{instance.__class__.__name__}({params})'


def dataframe_from_records(records, index_keys=None, dtypes=None, index=None, **kwargs):
    if isinstance(records, dict):
        assert index is None
        index, records = zip(*records.items())
    if dtypes is None:
        dtypes = {}
    df = pd.DataFrame.from_records(records, index=index, **kwargs)
    if len(df) == 0:
        return None
    dtypes = {k: v for k, v in dtypes.items() if k in df}
    df = df.astype(dtypes)
    if index_keys is not None:
        df = df.set_index(index_keys)
    return df


def array_stats(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        # nanmean warns if a is all-nan.
        return {'mean': np.nanmean(a),
                'std': np.nanstd(a),
                'nonnan': np.count_nonzero(~np.isnan(a))}


def named_array_stats(a, array_name):
    return {(array_name, stat_name): value for stat_name, value in array_stats(a).items()}


def join_dicts(dicts):
    # https://stackoverflow.com/a/3495395/4054250
    return {k: v for d in dicts for k, v in d.items()}


def invert_permutation(p, dtype=None):
    if dtype is None:
        dtype = p.dtype
    # https://stackoverflow.com/a/25535723/4054250
    s = np.empty(p.size, dtype)
    s[p] = np.arange(p.size)
    return s


def path_join(dirname, base, makedir=False):
    if dirname is None:
        return base
    if makedir:
        os.makedirs(dirname, exist_ok=True)
    return os.path.join(dirname, base)


def py_str(t):
    if isinstance(t, tf.Tensor):
        return bytes.decode(t.numpy())
    return str(t)


# https://stackoverflow.com/a/44500834/4054250
def count_iter_items(iterable, max_items=None):
    if max_items is None:
        counter = itertools.count()
    else:
        counter = iter(range(max_items))
    collections.deque(zip(iterable, counter), maxlen=0)
    try:
        return next(counter)
    except StopIteration:
        # We return a lower bound on the number of items.
        return max_items


def cardinality_finite(dataset, max_cardinality=None):
    n = dataset.cardinality()
    if n == tf.data.UNKNOWN_CARDINALITY:
        return count_iter_items(dataset, max_cardinality)
    elif n == tf.data.INFINITE_CARDINALITY:
        raise RuntimeError('The dataset has infinite cardinality.')
    return int(n)


def dataset_is_empty(dataset):
    return cardinality_finite(dataset, 1) == 0


def flatten_dict(d, **kwargs):
    return pd.json_normalize(d, **kwargs).to_dict(orient='records')[0]


def timer(*args, **kwargs):
    return contexttimer.Timer(*args, **kwargs, timer=time.perf_counter)


@contextmanager
def recursion_limit(limit=None):
    if limit is None:
        with suppress(KeyError):
            limit = int(os.environ['RECURSIONLIMIT'])
    default_limit = sys.getrecursionlimit()
    if limit is not None:
        sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(default_limit)


@contextmanager
def set_env(**environ):
    # Source: https://stackoverflow.com/a/34333710/4054250
    old_environ = os.environ.copy()
    os.environ.update({k: str(v) for k, v in environ.items()})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


class CsvDictWriter(csv.DictWriter):
    def __init__(self, f, fieldnames, *args, flush=True, sep='.', **kwargs):
        self.sep = sep
        super().__init__(f, self.flatten_dict(fieldnames).keys(), *args, **kwargs)
        self.f = f
        self.flush = flush

    def writerow(self, row):
        super().writerow(self.flatten_dict(row))
        if self.flush:
            self.f.flush()

    def flatten_dict(self, d):
        return flatten_dict(d, sep=self.sep)
