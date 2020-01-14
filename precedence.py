#!/usr/bin/env python3.7

import numpy as np

from utils import numpy_err_settings


def learn(runs, symbol_counts):
    return {symbol_type: learn_precedence(symbol_type, symbol_count, runs) for symbol_type, symbol_count in
            symbol_counts.items()}


def learn_precedence(symbol_type, symbol_count, runs):
    # `n[i, j]` is the number of hits of the symbol pair (i, j).
    n = np.zeros((symbol_count, symbol_count), dtype=np.uint)
    # `c[i, j]` is the cummulative score of the symbol pair (i, j).
    c = np.zeros((symbol_count, symbol_count), dtype=np.float)
    # Sum of scores across the results.
    score_sum = 0
    for run in runs:
        assert run.precedences is not None and symbol_type in run.precedences
        precedence = run.precedences[symbol_type]
        score = run.score
        score_sum += score
        for i in range(symbol_count):
            for j in range(i + 1, symbol_count):
                pi = precedence[i]
                pj = precedence[j]
                n[pi, pj] += 1
                c[pi, pj] += score
    with numpy_err_settings(invalid='ignore'):
        # Preference matrix. `v[i, j]` is the expected score of a run with the symbol pair (i, j).
        v = c / n
    # Default the unknown values to the mean value.
    mean_score = score_sum / len(runs)
    np.nan_to_num(v, copy=False, nan=mean_score)
    if symbol_type == 'predicate':
        head = np.asarray([0], dtype=np.uint)
        tail = construct_good_permutation(v[1:, 1:]) + 1
        assert np.all(tail > 0)
        perm = np.concatenate((head, tail))
    else:
        perm = construct_good_permutation(v)
    assert perm.shape == (symbol_count,)
    return perm, v


def construct_good_permutation(v):
    """Find good permutation greedily."""
    assert len(v.shape) == 2
    assert v.shape[0] == v.shape[1]
    n = v.shape[0]
    # s[i] = total score for row i - total score for column i
    # Symbol i should be picked as the first greedily if it maximizes s[i].
    s = v.sum(axis=1).flatten() - v.sum(axis=0).flatten()
    perm = np.zeros(n, dtype=np.uint)
    # https://papers.nips.cc/paper/1431-learning-to-order-things.pdf
    for i in range(n):
        cur = np.argmax(s)
        perm[i] = cur
        s += v[cur, :] - v[:, cur]
        s[cur] = np.NINF
    return perm
