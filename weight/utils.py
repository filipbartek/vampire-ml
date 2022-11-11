import itertools
import logging
import os
import sys
from contextlib import suppress
from decimal import Decimal
from enum import Enum

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf

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
            assert matching_column not in dtype_final
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
    # Pandas categorical
    if dtype in ['string', 'category']:
        return True
    if isinstance(data, list):
        return all(is_compatible(d, dtype) for d in data)
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
    if isinstance(rows[0], scipy.sparse.spmatrix):
        rows = [row.toarray() for row in rows]
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
    if isinstance(value, float) or isinstance(value, np.floating):
        return f'%.{precision}f' % value
    return str(value)


def subsample(a, size=None, rng=None):
    if size is None or size >= len(a):
        return a
    return rng.choice(a, size, replace=False)
