import itertools
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from contextlib import suppress
from decimal import Decimal
from enum import Enum

import hydra
import joblib
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf

from questions.utils import set_env

log = logging.getLogger(__name__)


def json_dump_default(obj, default=str):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, Decimal):
        return str(obj)
    return default(obj)


def astype(df, dtype, **kwargs):
    dtype_final = {}
    for column_pattern, column_dtype in dtype.items():
        matching_columns = df.columns[df.columns.str.fullmatch(column_pattern)]
        for matching_column in matching_columns:
            # If `matching_column` already is in `dtype_final`, we override it.
            # The latter entries override the former ones.
            dtype_final[matching_column] = column_dtype
    assert all(is_compatible(df[col], t) for col, t in dtype_final.items())
    # Cast the dataframe columns to appropriate dtypes
    return df.astype(dtype_final, **kwargs)


def save_df(df, basename, **kwargs):
    log.debug('%s.{csv,pkl}: Saving a dataframe of shape %s.' % (basename, df.shape))
    if os.path.dirname(basename) != '':
        os.makedirs(os.path.dirname(basename), exist_ok=True)
    with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
        df.to_csv(f'{basename}.csv', **kwargs)
    df.to_pickle(f'{basename}.pkl')


def invert_dict_of_lists(d):
    result = {}
    for k, vv in d.items():
        for v in vv:
            if k in result:
                if result[k] != v:
                    raise ValueError(f'Inconsistent value of {k}. Original: {result[k]}. New: {v}.')
            else:
                result[v] = k
    return result


def flatten(a):
    # Flatten a nested iterable.
    # The nesting depth is determined by the recursively first element.
    # Mimics `numpy.ndarray.flatten`.
    res = iter(a)
    while True:
        first = next(res)
        res = itertools.chain([first], res)
        try:
            iter(first)
        except TypeError:
            break
        res = itertools.chain.from_iterable(res)
    return res


assert list(flatten([[[0], [1]], [[2], [3]]])) == [0, 1, 2, 3]
assert list(flatten([[0, [1]], [2, 3]])) == [0, [1], 2, 3]


def is_compatible(data, dtype):
    if dtype in [None, 'string', 'category', float]:
        return True
    if isinstance(data, list):
        return all(is_compatible(d, dtype) for d in data)
    if any(s == 0 for s in data.shape):
        return True
    # Pandas
    if pd.api.types.is_bool_dtype(dtype):
        return data.isin([0, 1]).all()

    try:
        data_span = data.min(), data.max()
    except AttributeError:
        # We assume `data` to be scalar.
        data_span = data, data

    assert pd.isna(data_span[0]) == pd.isna(data_span[1])
    if all(pd.isna(ds) for ds in data_span):
        # This happens when data is all-NA, namely empty.
        # We assume that NA fits into all dtypes.
        return True

    # Watch out: bool is considered numeric.
    if pd.api.types.is_numeric_dtype(dtype):
        with suppress(AttributeError):
            # If `dtype` is a Pandas dtype, it exposes an underlying numpy dtype in `dtype.numpy_dtype`.
            # Otherwise we assume `dtype` to be a numpy dtype.
            dtype = dtype.numpy_dtype
        iinfo = np.iinfo(dtype)
        return iinfo.min <= data_span[0] and data_span[1] <= iinfo.max
    # TensorFlow
    if dtype.is_floating or dtype.is_integer:
        return dtype.min <= data_span[0] and data_span[1] <= dtype.max
    if dtype.is_bool:
        return data.dtype == np.bool
    return True


def to_tensor(rows, dtype, flatten_ragged=False, **kwargs):
    # `tf.ragged.constant` expects a list.
    rows = list(rows)
    assert is_compatible(rows, dtype)

    def densify(matrix):
        if isinstance(matrix, scipy.sparse.spmatrix):
            return matrix.toarray()
        return matrix

    rows = [densify(row) for row in rows]
    res = tf.ragged.constant(rows, dtype=dtype, **kwargs)
    if flatten_ragged and isinstance(res, tf.RaggedTensor):
        res = {k: getattr(res, k) for k in ['flat_values', 'nested_row_splits']}
    return res


def output_types(dtype, row_splits_dtype, rank):
    assert rank >= 1
    return {
        'flat_values': dtype,
        'nested_row_splits': (row_splits_dtype,) * rank
    }


def output_shapes(rank):
    assert rank >= 1
    return {
        'flat_values': tf.TensorShape([None]),
        'nested_row_splits': (tf.TensorShape([None]),) * rank
    }


def train_test_split(array, test_size=None, train_size=None, random_state=None):
    """
    Mimics `sklearn.model_selection.train_test_split`.
    Properties:
    - Supports `test_size=0` and `train_size=0`
    - Stable: Increasing the size of a dataset adds more elements into it but does not remove any. The size of test dataset has not impact on the elements in the train dataset.
    """
    if test_size is None and train_size is None:
        test_size = 0.25
    rng = np.random.default_rng(random_state)
    n = len(array)
    if isinstance(test_size, float):
        test_size = int(n * test_size)
    if isinstance(train_size, float):
        train_size = int(n * train_size)
    if test_size is None:
        test_size = n - train_size
    if train_size is None:
        train_size = n - test_size
    if test_size + train_size > n:
        raise RuntimeError(f'The requested datasets are too large: {test_size} + {train_size} > {n}')
    perm = rng.permutation(n)
    train_set = array[perm[:train_size]]
    test_set = array[perm[-1:-test_size - 1:-1]]
    assert len(test_set) == test_size
    return train_set, test_set


def to_str(value, precision=4):
    if isinstance(value, float) or isinstance(value, np.floating) or (
            isinstance(value, tf.Tensor) and value.dtype.is_floating):
        return f'%.{precision}f' % value
    return str(value)


def subsample(a, size=None, rng=None):
    if isinstance(a, int):
        if size >= a:
            return range(a)
    else:
        if size >= len(a):
            return a
    if rng is None:
        warnings.warn('Using the default NumPy random generator.')
        rng = np.random.default_rng()
    return rng.choice(a, size, replace=False)


def to_absolute_path(path):
    if path is None:
        return None
    return hydra.utils.to_absolute_path(path)


@contextmanager
def tf_trace(name, disable=False, **kwargs):
    if not disable:
        log.info(f'Starting trace {name}')
        tf.summary.trace_on(**kwargs)
    try:
        yield
    finally:
        if not disable:
            log.info(f'Exporting trace {name}')
            tf.summary.trace_export(name)


def error_record(e):
    return {'type': type(e).__name__, 'message': str(e)}


def path_join(*paths):
    try:
        return os.path.join(*paths)
    except TypeError:
        return None


def sparse_equal(l, r):
    return l.shape == r.shape and l.format == r.format and np.array_equal(l.indices, r.indices) and np.array_equal(
        l.indptr, r.indptr) and np.array_equal(l.data, r.data)


def get_verbose():
    return int(os.environ.get('VERBOSE', '1'))


@contextmanager
def set_verbose(verbose):
    with set_env(VERBOSE=int(verbose)):
        yield


@contextmanager
def get_parallel(n_tasks, n_jobs=None, verbose=None, **kwargs):
    if n_tasks < (joblib.parallel.get_active_backend()[1] or 1):
        if n_jobs is not None:
            n_jobs = min(n_jobs, max(n_tasks, 1))
        else:
            n_jobs = max(n_tasks, 1)
    if verbose is None:
        verbose = get_verbose()
    parallel = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, **kwargs)
    new_verbose = verbose and parallel.n_jobs == 1
    # We disable CUDA for the subprocesses so that they do not crash due to the exclusivity of access to a GPU.
    # https://github.com/tensorflow/tensorflow/issues/30594
    with set_verbose(new_verbose), set_env(CUDA_VISIBLE_DEVICES=-1):
        yield parallel


def range_count(stop, *args, **kwargs):
    if stop is None:
        return itertools.count(*args, **kwargs)
    return range(stop, *args, **kwargs)


class with_cardinality:
    # Inspiration: `more_itertools.countable`, `tf.data.Dataset`

    # See `tf.data.INFINITE_CARDINALITY`
    INFINITE = -1
    # See `tf.data.UNKNOWN_CARDINALITY`
    UNKNOWN = -2

    def __init__(self, iterable, n):
        self._it = iter(iterable)
        self.cardinality = n

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def __len__(self):
        if self.cardinality >= 0:
            return self.cardinality
        else:
            raise TypeError(f'Cardinality: {self.cardinality}')

    def is_finite(self):
        return self.cardinality >= 0
