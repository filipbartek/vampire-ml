import itertools
import logging
import os
import sys

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)


def astype(df, dtype, **kwargs):
    # Cast the dataframe columns to appropriate dtypes
    return df.astype({k: v for k, v in dtype.items() if k in df}, **kwargs)


def save_df(df, basename, **kwargs):
    log.debug('%s.{csv,pkl}: Saving a dataframe of shape %s.' % (basename, df.shape))
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
    if dtype.is_floating or dtype.is_integer:
        return all(dtype.min <= v <= dtype.max for v in flatten(data))
    if dtype.is_bool:
        return all(v in [False, True] for v in flatten(data))
    return True


def to_tensor(rows, dtype, flatten_ragged=True, **kwargs):
    # `tf.ragged.constant` expects a list.
    rows = list(rows)
    assert is_compatible(rows, dtype)
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
