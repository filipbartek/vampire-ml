#!/usr/bin/env python3.7

import collections


def get_updated(d, u):
    """
    See also:
    * [dict.update()d](https://docs.python.org/3/library/stdtypes.html#dict.update)
    * [dirt-recursive-update](https://pypi.org/project/dict-recursive-update/)

    Unlike `dict.update`, this function updates the dictionary recursively, conserving as much data as possible.
    Besides that, this function does not modify `d` in place.
    Unlike `dict_recursive_update.recursive_update`, this function does not modify the source dictionary nor any of its
    descendants.
    """
    result = d.copy()
    for k, v in u.items():
        if k in result and isinstance(v, collections.Mapping):
            result[k] = get_updated(result[k], v)
        else:
            result[k] = v
    return result
