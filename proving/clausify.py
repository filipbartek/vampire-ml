import logging

import joblib
from joblib import Parallel, delayed

from proving import config
from proving import file_path_list
from proving import utils
from proving.solver import Solver
from vampire_ml.results import save_df


def main():
    solver = Solver(timeout=20)
    problems = sorted(file_path_list.paths_from_patterns(('**/*-*.p', '**/*+*.p'), base_path=config.problems_path()))
    with joblib.parallel_backend('threading', n_jobs=-1):
        records = Parallel(verbose=1000)(delayed(process_one_problem)(problem, solver) for problem in problems)
        df = utils.dataframe_from_records(records, 'path')
        save_df(df, 'problems', 'out')


def process_one_problem(problem, solver):
    clausify_result = solver.clausify(problem)
    return {
        'path': problem,
        'predicates': len(clausify_result.symbols_of_type('predicate')),
        'functions': len(clausify_result.symbols_of_type('function')),
        'clauses': len(clausify_result.clauses)
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(threadName)s %(levelname)s - %(message)s')
    main()
