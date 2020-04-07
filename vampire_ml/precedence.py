#!/usr/bin/env python3.7

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import truncate


def learn_ltot(pair_scores, symbol_type=None):
    if symbol_type == 'predicate':
        head = np.asarray([0], dtype=np.uint)
        tail = ltot_construct_permutation(pair_scores[1:, 1:]) + 1
        assert np.all(tail > 0)
        perm = np.concatenate((head, tail))
    else:
        perm = ltot_construct_permutation(pair_scores)
    assert perm is not None
    return perm


def ltot_construct_permutation(pair_scores, nan=None):
    """Find good permutation greedily."""
    n = pair_scores.shape[0]
    assert (n, n) == pair_scores.shape
    perm = np.empty(n, dtype=np.uint)
    if n == 0:
        return perm
    if nan is None:
        assert np.any(~np.isnan(pair_scores))
        nan = np.nanmean(pair_scores)
    pair_scores = np.nan_to_num(pair_scores, nan=nan)
    # s[i] = total score for row i - total score for column i
    # s[i] is the score we get by picking symbol i as the first symbol.
    # Symbol i should be picked as the first greedily if it minimizes s[i].
    point_scores = np.sum(pair_scores, axis=1).flatten() - np.sum(pair_scores, axis=0).flatten()
    # https://papers.nips.cc/paper/1431-learning-to-order-things.pdf
    for i in range(n):
        cur = np.argmin(point_scores)
        assert point_scores[cur] != np.inf
        perm[i] = cur
        point_scores += pair_scores[cur, :] - pair_scores[:, cur]
        point_scores[cur] = np.inf
    return perm


def plot_preference_heatmap(preference, permutation, symbol_type, problem, output_file=None):
    symbols = problem.get_symbols(symbol_type)
    n = len(symbols)
    assert preference.shape == (n, n)
    assert len(permutation) == n
    # https://stackoverflow.com/a/58709110/4054250
    assert np.allclose(preference[permutation, :][:, permutation], preference[:, permutation][permutation, :],
                       rtol=0, atol=0, equal_nan=True)
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
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(f'{output_file}.{file_type}', bbox_inches='tight')
    else:
        plt.show()
    plt.close()
