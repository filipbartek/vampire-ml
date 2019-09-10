#!/usr/bin/env python3.7

import logging

import numpy as np
import sklearn.linear_model

import run_database


def get_score_stats(problem_prove_runs):
    elapsed_times = []
    n_successful = 0
    for prove_run in problem_prove_runs:
        exit_code = prove_run.exit_code
        if exit_code not in [0, 1]:
            logging.warning(f'Skipping prove run {prove_run.path_rel}. Exit code: {exit_code}.')
            continue
        if prove_run.success:
            n_successful += 1
        elapsed_times.append(prove_run.time_elapsed_vampire)
    if n_successful == 0:
        raise RuntimeError(f'Problem has no successful prove run.')
    return np.median(elapsed_times), np.amin(elapsed_times), np.amax(elapsed_times)


def get_score(prove_run, median, min_time, max_time):
    if not prove_run.success:
        run_is_good = False
        run_weight = 1
    else:
        time_elapsed = prove_run.time_elapsed_vampire
        run_is_good = time_elapsed <= median
        if run_is_good:
            assert time_elapsed <= median
            assert min_time <= median
            if min_time == median:
                run_weight = 1
            else:
                run_weight = (median - time_elapsed) / (median - min_time)
        else:
            assert median <= time_elapsed
            assert median <= max_time
            if median == max_time:
                run_weight = 1
            else:
                run_weight = (time_elapsed - median) / (max_time - median)
        assert run_weight >= 0
        assert run_weight <= 1
    return run_is_good, run_weight


def generate_example(br_prove, br_probe):
    problem_path = np.random.choice(list(br_prove.problems_with_some_success))
    assert problem_path in br_prove.problem_dict
    prove_runs = br_prove.problem_dict[problem_path]
    prove_run = np.random.choice(prove_runs)
    assert prove_run.problem_path == problem_path

    assert problem_path in br_probe.problem_dict
    assert len(br_probe.problem_dict[problem_path]) == 1
    probe_run = br_probe.problem_dict[problem_path][0]

    predicates = probe_run.predicate_embeddings()
    predicate_precedence = prove_run.predicate_precedence
    run_is_good, run_weight = get_score(prove_run, *get_score_stats(br_prove.problem_dict[problem_path]))

    assert len(predicate_precedence) == len(predicates)
    symbol_orders = np.random.choice(len(predicate_precedence), size=2, replace=False)
    l_order = np.minimum(symbol_orders[0], symbol_orders[1])
    r_order = np.maximum(symbol_orders[0], symbol_orders[1])
    assert l_order < r_order
    assert l_order < len(predicate_precedence)
    l_index = predicate_precedence[l_order]
    assert r_order < len(predicate_precedence)
    r_index = predicate_precedence[r_order]
    l_embedding = predicates[l_index, :]
    r_embedding = predicates[r_index, :]

    # TODO: Add problem embedding to input vector, namely so that ocurrence counts from large problems do not overweigh those from small problems.
    return np.concatenate((l_embedding, r_embedding)), run_is_good, run_weight


# TODO: Ensure that samples of both classes are generated.
def generate_batch(br_prove, br_probe, n):
    n_features = 12
    result_X = np.zeros((n, n_features), dtype=float)
    result_y = np.zeros(n, dtype=bool)
    result_weights = np.ones(n, dtype=float)
    for i in range(n):
        row_X, row_y, row_weight = generate_example(br_prove, br_probe)
        result_X[i] = row_X
        result_y[i] = row_y
        result_weights[i] = row_weight
    return result_X, result_y, result_weights


def call(namespace):
    if namespace.seed >= 0:
        np.random.seed(namespace.seed)

    br_prove = run_database.BatchResult(namespace.result_prove)
    assert br_prove.mode == 'vampire'
    br_probe = run_database.BatchResult(namespace.result_probe)
    assert br_probe.mode == 'clausify'

    for batch_size in [16, 256, 1024]:
        for i in range(4):
            X, y, weights = generate_batch(br_prove, br_probe, batch_size)
            clf = sklearn.linear_model.LogisticRegression(solver='liblinear')
            clf.fit(X, y, weights)
            print('Batch size: %s. Score: %s.' % (batch_size, clf.score(*generate_batch(br_prove, br_probe, 1024))))


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('result-prove', type=str, help='result of a prove run of `vampire-batch vampire`')
    parser.add_argument('result-probe', type=str, help='result of a probe run of `vampire-batch vampire`')
    parser.add_argument('--seed', type=int, default=0, help='randomness generator seed')
