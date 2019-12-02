#!/usr/bin/env python3.7

import numpy as np

from utils import numpy_err_settings


def learn(problem_results, symbol_counts):
    saturation_iterations = [result['saturation_iterations'] for result in problem_results if
                             result['saturation_iterations'] is not None and result['exit_code'] == 0]
    if len(saturation_iterations) == 0:
        raise RuntimeError('Cannot learn from zero successful runs.')
    saturation_iterations_min = min(saturation_iterations)
    saturation_iterations_max = max(saturation_iterations)
    return {
        symbol_type: learn_precedence(symbol_type, symbol_count, problem_results, saturation_iterations_min,
                                      saturation_iterations_max) for symbol_type, symbol_count in symbol_counts.items()}


def learn_precedence(symbol_type, symbol_count, problem_results, saturation_iterations_min, saturation_iterations_max):
    # `n[i, j]` is the number of hits of the symbol pair (i, j).
    n = np.zeros((symbol_count, symbol_count), dtype=np.uint)
    # `c[i, j]` is the cummulative score of the symbol pair (i, j).
    c = np.zeros((symbol_count, symbol_count), dtype=np.float)
    # Sum of scores across the results.
    score_sum = 0
    for result in problem_results:
        assert 'precedences' in result and result['precedences'] is not None and symbol_type in result['precedences']
        precedence = result['precedences'][symbol_type]
        score = get_score(result, saturation_iterations_min, saturation_iterations_max)
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
    mean_score = score_sum / len(problem_results)
    np.nan_to_num(v, copy=False, nan=mean_score)
    return construct_good_permutation(v), v


def get_score(result, saturation_iterations_min, saturation_iterations_max):
    # TODO: Think through.
    # For example: assume that failed run would finish in "max(iterations) * 2" iterations.
    score = -1
    if result['exit_code'] == 0:
        assert result['saturation_iterations'] is not None
        score = np.interp(result['saturation_iterations'],
                          [saturation_iterations_min, saturation_iterations_max], [1, 0])
    return score


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
