#!/usr/bin/env python3.7

import os
from contextlib import contextmanager

import numpy as np


def makedirs_open(dir_path, file, *args):
    file_abs = os.path.join(dir_path, file)
    os.makedirs(os.path.dirname(file_abs), exist_ok=True)
    return open(file_abs, *args)


def get_consistent(d, key, value=None, override=False):
    """
    Get the value `d[key]`, ensuring it is consistent with `value`.
    """
    if key not in d:
        return value
    if value is None:
        return d[key]
    if d[key] == value:
        return value
    if override:
        return d[key]
    raise RuntimeError(f'Inconsistent value of key {key}. Expected: {value}. Actual: {d[key]}.')


def len_robust(item):
    if item is None:
        return None
    return len(item)


def fill_category_na(df, value='NA'):
    # Replace NaN with 'NA' in category columns. This allows getting more useful statistics.
    for field in df.select_dtypes(['category']):
        series = df[field]
        assert value not in series.cat.categories
        series.cat.add_categories(value, inplace=True)
        series.fillna(value, inplace=True)


@contextmanager
def numpy_err_settings(**kwargs):
    old_settings = np.seterr(**kwargs)
    try:
        yield np.geterr()
    finally:
        np.seterr(**old_settings)


def truncate(s, n, ellipsis_str='..'):
    if n is None:
        return s
    return s[:n - len(ellipsis_str)] + ellipsis_str if len(s) > n else s
