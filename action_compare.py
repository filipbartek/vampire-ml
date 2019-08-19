#!/usr/bin/env python3.7

import itertools
import os

import pandas as pd

import file_path_list
import run_database


def fillna(s, value='NA'):
    assert value not in s.cat.categories
    s.cat.add_categories([value], inplace=True)
    s.fillna(value, inplace=True)


def call(namespace):
    problem_paths, problem_base_path = file_path_list.compose(namespace.problem_list, namespace.problem,
                                                              namespace.problem_base_path)

    assert len(namespace.result) == 2
    brs = []
    for result_path in namespace.result:
        br = run_database.BatchResult(result_path)
        assert br.mode == 'vampire'
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
        assert df['termination_reason'].isna().sum() == 0
        fillna(df['termination_phase'], '[Missing]')
        df.columns = pd.MultiIndex.from_product([[i], df.columns], names=['batch', 'property'])
        # print(len(df))
        problems = problems.join(df)

    batch_level_values = problems.columns.levels[0]

    for column in itertools.product(batch_level_values, ['exit_code', 'termination_reason', 'termination_phase']):
        fillna(problems[column], '[Not executed]')

    print('Number of problems:', len(problems))
    print(problems.columns.levels[1])
    print(problems.groupby(list(itertools.product(batch_level_values, ['success']))).size())
    print(problems.groupby(list(itertools.product(batch_level_values, ['exit_code']))).size())
    print(problems.groupby(list(itertools.product(batch_level_values, ['termination_reason', 'termination_phase']))).size())


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('result', type=str, nargs=2, help='result of a prove run')
    parser.add_argument('--problem_list', action='append', default=[],
                        help='input file with a list of problem paths')
    parser.add_argument('--problem', action='append', default=[], help='glob pattern of problem path')
    parser.add_argument('--problem_base_path', help='base path of problem paths')
    parser.add_argument('--output', '-o', type=str, required=True, help='output directory')
