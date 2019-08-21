#!/usr/bin/env python3.7

import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd

import file_path_list
import run_database


def fillna(s, value='NA'):
    assert value not in s.cat.categories
    s.cat.add_categories([value], inplace=True)
    s.fillna(value, inplace=True)


def color_from_exit_codes(exit_codes):
    if (exit_codes == 0).all():
        return 'green'
    if (exit_codes == 0).any():
        return 'yellow'
    return 'red'


def call(namespace):
    problem_paths, problem_base_path = file_path_list.compose(namespace.problem_list, namespace.problem,
                                                              namespace.problem_base_path)

    br_vampire_indexes = []
    brs = []
    for i, result_path in enumerate(namespace.result):
        br = run_database.BatchResult(result_path)
        if br.mode == 'vampire':
            br_vampire_indexes.append(i)
        if problem_base_path is None:
            problem_base_path = br.problem_base_path
        assert os.path.samefile(problem_base_path, br.problem_base_path)
        brs.append(br)

    if len(problem_paths) == 0:
        problem_paths_set = set()
        for br in brs:
            for run in br.runs:
                problem_paths_set.add(run.problem_path)
        problem_paths = list(problem_paths_set)

    assert len(problem_paths) > 0
    problems = pd.DataFrame(index=problem_paths)
    problems.index.name = 'problem_path'
    problems.columns = pd.MultiIndex([[], []], [[], []], names=['batch', 'property'])

    for i, br in enumerate(brs):
        assert br.representative_runs_data_frame.index.name == 'problem_path'
        df = br.representative_runs_data_frame.copy()
        assert df['exit_code'].isna().sum() == 0
        for column in ['termination_reason', 'termination_phase']:
            fillna(df[column], '[Missing]')
        df.columns = pd.MultiIndex.from_product([[i], df.columns], names=['batch', 'property'])
        problems = problems.join(df)

    batch_level_values = problems.columns.levels[0]

    for column in itertools.product(batch_level_values, ['exit_code', 'termination_reason', 'termination_phase']):
        fillna(problems[column], '[Not executed]')

    print('Number of problems:', len(problems))
    print(problems.columns.levels[1])
    print(problems.groupby(list(itertools.product(br_vampire_indexes, ['success']))).size())
    print(problems.groupby(list(itertools.product(br_vampire_indexes, ['exit_code']))).size())
    print(problems.groupby(
        list(itertools.product(br_vampire_indexes, ['termination_reason', 'termination_phase']))).size())

    corr = problems[itertools.product(batch_level_values, [
        'exit_code',
        'time_elapsed_process',
        'saturation_iterations',
        'predicates_count',
        'functions_count',
        'clauses_count'
    ])].corr()
    print(corr)

    exit_codes = problems[[(batch, 'exit_code') for batch in br_vampire_indexes]]
    colors = exit_codes.apply(color_from_exit_codes, axis=1)

    problems.plot.scatter(x=(br_vampire_indexes[0], 'time_elapsed_process'),
                          y=(br_vampire_indexes[1], 'time_elapsed_process'),
                          logx=True, logy=True, c=colors, alpha=0.5)
    plt.title('time_elapsed_process')
    plt.xlabel(brs[br_vampire_indexes[0]].result_path)
    plt.ylabel(brs[br_vampire_indexes[1]].result_path)
    plt.plot([0, 10], [0, 10])
    plt.show()

    problems.plot.scatter(x=(br_vampire_indexes[0], 'saturation_iterations'),
                          y=(br_vampire_indexes[1], 'saturation_iterations'),
                          logx=True, logy=True, c=colors, alpha=0.5)
    plt.title('saturation_iterations')
    plt.xlabel(brs[br_vampire_indexes[0]].result_path)
    plt.ylabel(brs[br_vampire_indexes[1]].result_path)
    plt.plot([0, 1000], [0, 1000])
    plt.show()


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('result', nargs='+', help='result of a run')
    parser.add_argument('--problem_list', action='append', default=[],
                        help='input file with a list of problem paths')
    parser.add_argument('--problem', action='append', default=[], help='glob pattern of problem path')
    parser.add_argument('--problem_base_path', help='base path of problem paths')
    parser.add_argument('--output', '-o', type=str, required=True, help='output directory')
