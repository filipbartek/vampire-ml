import warnings

import numpy as np
import pandas as pd


def type_len(s):
    try:
        return f'{type(s).__name__}[{len(s)}]'
    except TypeError:
        return s


def instance_str(instance):
    params = ', '.join(f'{key}={type_len(value)}' for key, value in instance.__dict__.items())
    return f'{instance.__class__.__name__}({params})'


def dataframe_from_records(records, index_keys=None, dtypes=None):
    if dtypes is None:
        dtypes = {}
    df = pd.DataFrame.from_records(records)
    if len(df) == 0:
        return None
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
