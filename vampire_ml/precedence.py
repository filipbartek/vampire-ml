#!/usr/bin/env python3.7

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import numpy_err_settings, truncate


def learn_precedence_lex(precedence_scores, symbol_type, problem=None, output_dir=None, failure_score='dominate'):
    pair_hits, pair_failures, pair_score_sum = compute_pairwise_scores(precedence_scores)
    n = pair_hits.shape[0]
    assert (n, n) == pair_hits.shape == pair_failures.shape == pair_score_sum.shape
    assert np.all(pair_hits >= pair_failures)
    assert np.all(pair_failures >= 0)
    pair_successes = pair_hits - pair_failures
    if failure_score == 'dominate':
        with numpy_err_settings(invalid='ignore'):
            # Preference matrix. `v[i, j]` is the expected score of a run with the symbol pair (i, j).
            pair_failure_rate = pair_failures / pair_hits
    else:
        if failure_score != 'ignore':
            pair_score_sum += pair_failures * failure_score
            pair_successes = pair_hits
        pair_failure_rate = np.zeros(pair_hits.shape, dtype=np.float)
    with numpy_err_settings(invalid='ignore'):
        # Preference matrix. `v[i, j]` is the expected score of a run with the symbol pair (i, j).
        pair_mean_score = pair_score_sum / pair_successes
    if symbol_type == 'predicate':
        head = np.asarray([0], dtype=np.uint)
        tail = construct_good_permutation_lex(pair_failure_rate[1:, 1:], pair_mean_score[1:, 1:]) + 1
        assert np.all(tail > 0)
        perm = np.concatenate((head, tail))
    else:
        perm = construct_good_permutation_lex(pair_failure_rate, pair_mean_score)
    assert perm.shape == (n,)
    try:
        plot_preference_heatmap(pair_failure_rate, perm, symbol_type, problem,
                                output_file=os.path.join(output_dir, 'preferences',
                                                         f'{problem.name()}_{symbol_type}_failures'))
        plot_preference_heatmap(pair_mean_score, perm, symbol_type, problem,
                                output_file=os.path.join(output_dir, 'preferences',
                                                         f'{problem.name()}_{symbol_type}_score'))
    except ValueError:
        logging.debug('Preference heatmap plotting failed.', exc_info=True)
    return perm


def compute_pairwise_scores(precedence_scores):
    precedence_scores = list(precedence_scores)
    n = len(precedence_scores[0][0])
    # `pair_hits[i, j]` is the number of hits of the symbol pair (i, j).
    pair_hits = np.zeros((n, n), dtype=np.uint)
    # `pair_failures[i, j]` is the number of failures with the symbol pair (i, j).
    pair_failures = np.zeros((n, n), dtype=np.uint)
    # `pair_score_sum[i, j]` is the cumulative score of the symbol pair (i, j).
    pair_score_sum = np.zeros((n, n), dtype=np.float)
    for precedence, score in precedence_scores:
        precedence_inverse = invert_permutation(precedence)
        pairs_hit_now = np.tri(n, k=-1, dtype=np.uint).transpose()[precedence_inverse, :][:, precedence_inverse]
        pair_hits += pairs_hit_now
        if not np.isnan(score):
            pair_score_sum += pairs_hit_now * score
        else:
            pair_failures += pairs_hit_now
    return pair_hits, pair_failures, pair_score_sum


def invert_permutation(p):
    # https://stackoverflow.com/a/25535723/4054250
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def construct_good_permutation_lex(failure_rate, average_iterations):
    """Find good permutation greedily."""
    n = failure_rate.shape[0]
    assert (n, n) == failure_rate.shape == average_iterations.shape
    perm = np.empty(n, dtype=np.uint)
    if n == 0:
        return perm
    # s[i] = total score for row i - total score for column i
    # s[i] is the score we get by picking symbol i as the first symbol.
    # Symbol i should be picked as the first greedily if it minimizes s[i].
    # TODO: Test with all-nan slice (1 run).
    failure_rate = np.nan_to_num(failure_rate, nan=np.nanmean(failure_rate))
    average_iterations = np.nan_to_num(average_iterations, nan=np.nanmean(average_iterations))
    s_failure_rate = np.sum(failure_rate, axis=1).flatten() - np.sum(failure_rate, axis=0).flatten()
    s_average_iterations = np.sum(average_iterations, axis=1).flatten() - np.sum(average_iterations, axis=0).flatten()
    # https://papers.nips.cc/paper/1431-learning-to-order-things.pdf
    for i in range(n):
        minimum_failure_rate_indices = np.flatnonzero(np.isclose(s_failure_rate, np.min(s_failure_rate)))
        cur = minimum_failure_rate_indices[np.argmin(s_average_iterations[minimum_failure_rate_indices])]
        assert s_failure_rate[cur] != np.inf
        assert s_average_iterations[cur] != np.inf
        perm[i] = cur
        s_failure_rate += failure_rate[cur, :] - failure_rate[:, cur]
        s_failure_rate[cur] = np.inf
        s_average_iterations += average_iterations[cur, :] - average_iterations[:, cur]
        s_average_iterations[cur] = np.inf
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
