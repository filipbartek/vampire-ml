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


def dataframe_from_records(records, index_keys):
    if len(records) == 0:
        return None
    return pd.DataFrame.from_records(records).set_index(index_keys)


def array_stats(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        # nanmean warns if a is all-nan.
        return {'mean': np.nanmean(a),
                'std': np.nanstd(a),
                'nonnan': np.count_nonzero(~np.isnan(a))}


def named_array_stats(a, array_name):
    return {(array_name, stat_name): value for stat_name, value in array_stats(a).items()}
