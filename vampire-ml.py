#!/usr/bin/env python3.7

import argparse
import json
import logging
import os

import numpy as np
import sklearn.linear_model


def get_predicates_from_result_data(result_data):
    assert 'predicates' in result_data
    n_predicates = len(result_data['predicates'])
    # Predicate features: equality?, arity, usageCnt, unitUsageCnt, inGoal?, inUnit?
    n_predicate_features = 6
    predicates = np.zeros((n_predicates, n_predicate_features), dtype=int)
    for i, predicate in enumerate(result_data['predicates']):
        assert not predicate['skolem']
        assert not predicate['inductionSkolem']
        isEquality = i == 0
        assert (isEquality == (predicate['name'] == '='))
        assert (not isEquality or predicate['arity'] == 2)
        predicates[i, 0] = isEquality
        predicates[i, 1] = predicate['arity']
        predicates[i, 2] = predicate['usageCnt']
        predicates[i, 3] = predicate['unitUsageCnt']
        predicates[i, 4] = predicate['inGoal']
        predicates[i, 5] = predicate['inUnit']
    return predicates


def get_predicates_from_problem(probe_file_path):
    with open(os.path.join(probe_file_path)) as probe_file:
        probe_data = json.load(probe_file)
    probe_exit_code = probe_data['call']['exit_code']
    if probe_exit_code != 0:
        raise RuntimeError(f'Probe {probe_file_path} failed with exit code {probe_exit_code}.')
    with open(os.path.join(os.path.dirname(probe_file_path),
                           probe_data['output']['json_output_path'])) as json_output_file:
        result_data = json.load(json_output_file)
    return get_predicates_from_result_data(result_data)


def get_prove_run_data(main_dirname, prove_run_paths):
    n_successful = 0
    prove_runs = []
    elapsed_times = []
    # TODO: Cache the per-problem data so that we can sample runs from a problem more efficiently.
    for prove_run_path in prove_run_paths:
        with open(os.path.join(main_dirname, prove_run_path)) as prove_run_file:
            run_data = json.load(prove_run_file)
        exit_code = run_data['call']['exit_code']
        if exit_code not in [0, 1]:
            logging.warning(f'Skipping prove run {prove_run_path}. Exit code: {exit_code}.')
            continue
        success = run_data['output']['data']['termination']['reason'] in ['Refutation', 'Satisfiable']
        # TODO: Assert all the runs have the same time_limit.
        prove_runs.append({
            'success': success,
            'time_elapsed': run_data['output']['data']['time_elapsed'],
            'time_limit': run_data['vampire_parameters']['time_limit'],
            'predicate_precedence': run_data['output']['data']['predicate_precedence']
        })
        elapsed_times.append(run_data['output']['data']['time_elapsed'])
        if success:
            n_successful += 1
    if n_successful == 0:
        raise RuntimeError(f'Problem has no successful prove run.')
    prove_run = np.random.choice(prove_runs)
    if not prove_run['success']:
        run_is_good = False
        run_weight = 1
    else:
        time_elapsed = prove_run['time_elapsed']
        median = np.median(elapsed_times)
        run_is_good = time_elapsed <= median
        if run_is_good:
            min_time = np.amin(elapsed_times)
            assert time_elapsed <= median
            assert min_time <= median
            if min_time == median:
                run_weight = 1
            else:
                run_weight = (median - time_elapsed) / (median - min_time)
        else:
            max_time = np.amax(elapsed_times)
            assert median <= time_elapsed
            assert median <= max_time
            if median == max_time:
                run_weight = 1
            else:
                run_weight = (time_elapsed - median) / (max_time - median)
        assert run_weight >= 0
        assert run_weight <= 1
    return prove_run['predicate_precedence'], run_is_good, run_weight


def generate_example(main_dirname, problems):
    problem = np.random.choice(list(problems))
    predicates = get_predicates_from_problem(os.path.join(main_dirname, problem['probe']))
    predicate_precedence, run_is_good, run_weight = get_prove_run_data(main_dirname, problem['prove_runs'])

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


def generate_batch(main_dirname, problems, n):
    n_features = 12
    result_X = np.zeros((n, n_features), dtype=float)
    result_y = np.zeros(n, dtype=bool)
    result_weights = np.ones(n, dtype=float)
    for i in range(n):
        row_X, row_y, row_weight = generate_example(main_dirname, problems)
        result_X[i] = row_X
        result_y[i] = row_y
        result_weights[i] = row_weight
    return result_X, result_y, result_weights


# TODO: Generate training data on demand. Unify this script with vampire-batch.py.
if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=argparse.FileType('r'),
                        help='JSON document with data generated by vampire-batch.py')
    parser.add_argument('--histogram', action='store_true', help='plot histogram of prove run times for each problem')
    parser.add_argument('--seed', type=int, default=0, help='seed for the pseudo-random number generator')
    namespace = parser.parse_args()

    np.random.seed(namespace.seed)

    problems = json.load(namespace.data)
    namespace.data.close()

    main_dirname = os.path.dirname(namespace.data.name)

    X, y, weights = generate_batch(main_dirname, problems.values(), 64)
    clf = sklearn.linear_model.LogisticRegression(solver='liblinear')
    clf.fit(X, y, weights)
    print(clf)
    print(clf.score(*generate_batch(main_dirname, problems.values(), 16)))
