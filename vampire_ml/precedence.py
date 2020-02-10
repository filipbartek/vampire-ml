#!/usr/bin/env python3.7

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import numpy_err_settings, truncate


def learn_precedence(precedence_scores, symbol_type, problem=None, output_dir=None):
    precedence_scores = list(precedence_scores)
    symbol_count = len(precedence_scores[0][0])
    # `n[i, j]` is the number of hits of the symbol pair (i, j).
    n = np.zeros((symbol_count, symbol_count), dtype=np.uint)
    # `c[i, j]` is the cumulative score of the symbol pair (i, j).
    c = np.zeros((symbol_count, symbol_count), dtype=np.float)
    # Sum of scores across the results.
    score_sum = 0
    for precedence, score in precedence_scores:
        assert len(precedence) == symbol_count
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
    mean_score = score_sum / len(precedence_scores)
    np.nan_to_num(v, copy=False, nan=mean_score)
    if symbol_type == 'predicate':
        head = np.asarray([0], dtype=np.uint)
        tail = construct_good_permutation(v[1:, 1:]) + 1
        assert np.all(tail > 0)
        perm = np.concatenate((head, tail))
    else:
        perm = construct_good_permutation(v)
    assert perm.shape == (symbol_count,)
    try:
        plot_preference_heatmap(v, perm, symbol_type, problem, output_dir=os.path.join(output_dir, 'preferences'))
    except ValueError:
        logging.debug('Preference heatmap plotting failed.', exc_info=True)
    return perm


def plot_preference_heatmap(preference, permutation, symbol_type, problem, output_dir=None):
    symbols = problem.get_symbols(symbol_type)
    n = len(symbols)
    assert preference.shape == (n, n)
    assert len(permutation) == n
    assert np.array_equal(preference[permutation, :][:, permutation], preference[:, permutation][permutation, :])
    # Vampire forces '=' to be the first predicate.
    assert symbol_type != 'predicate' or (symbols.name[0] == '=' and permutation[0] == 0)
    v_permuted = preference[permutation, :][:, permutation]
    tick_labels = False
    if n <= 32:
        tick_labels = [f'{truncate(symbols.name[i], 16)}' for i in permutation]
    file_type = 'svg'
    if n > 64:
        # For large n, the svg file size is too large.
        file_type = 'png'
    # We mask the diagonal because the values on the diagonal don't have a sensible interpretation.
    plt.figure()
    sns.heatmap(v_permuted, xticklabels=tick_labels, yticklabels=tick_labels,
                mask=np.eye(v_permuted.shape[0], dtype=np.bool), square=True)
    plt.title(
        f'Expected pairwise preferences of {symbol_type} symbols in problem {problem}')
    plt.ylabel('Early symbol')
    plt.xlabel('Late symbol')
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{problem.name()}_{symbol_type}.{file_type}'),
                    bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def construct_good_permutation(v):
    """Find good permutation greedily."""
    n = v.shape[0]
    assert v.shape == (n, n)
    # s[i] = total score for row i - total score for column i
    # Symbol i should be picked as the first greedily if it maximizes s[i].
    s = v.sum(axis=1).flatten() - v.sum(axis=0).flatten()
    perm = np.empty(n, dtype=np.uint)
    # https://papers.nips.cc/paper/1431-learning-to-order-things.pdf
    for i in range(n):
        cur = np.argmax(s)
        assert s[cur] != np.NINF
        perm[i] = cur
        s += v[cur, :] - v[:, cur]
        s[cur] = np.NINF
    return perm
