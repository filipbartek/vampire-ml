#!/usr/bin/env python3

import functools
import logging

import joblib
import numpy as np
import pandas as pd
import sklearn.pipeline
from sklearn import model_selection
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import FunctionTransformer

from proving import batch
from proving import cost_predictor
from proving.solver import Solver
from vampire_ml import sklearn_extensions
from vampire_ml.results import save_df
from vampire_ml.sklearn_extensions import InterceptRegression
from vampire_ml.sklearn_extensions import QuantileImputer
from vampire_ml.sklearn_extensions import StableStandardScaler
from vampire_ml.sklearn_extensions import fit

log = logging.getLogger(__name__)


def main():
    vampire_options = {
        'encode': 'on',
        'statistics': 'full',
        'time_statistics': 'on',
        'proof': 'off',
        'literal_comparison_mode': 'predicate',
        'symbol_precedence': 'frequency',
        'saturation_algorithm': 'discount',
        'age_weight_ratio': '10',
        'avatar': 'off',
        'time_limit': '10'
    }
    solver = Solver(options=vampire_options, timeout=20)

    cost_normalizer_steps = {
        'imputer': QuantileImputer(copy=False, quantile=1, factor=2, default_fill_value=1),
        'log': FunctionTransformer(func=np.log),
        'standardizer': StableStandardScaler(copy=False)
    }
    cost_normalizer = sklearn.pipeline.Pipeline(list(cost_normalizer_steps.items()))

    random_precedence_types = ['predicate']

    # problems = ['PUZ/PUZ001-1.p', 'LDA/LDA005-1.p', 'HWV/HWV129+1.p']
    # problems = ['HWV/HWV129+1.p']
    # problems = ['LDA/LDA005-1.p']  # All-nan costs
    # problems = ['PUZ/PUZ001-1.p', 'PUZ/PUZ002-1.p', 'LDA/LDA005-1.p', 'HWV/HWV129+1.p', 'LAT/LAT337+1.p']
    problems = ['PUZ/PUZ001-1.p', 'PUZ/PUZ002-1.p', 'LDA/LDA005-1.p', 'LAT/LAT337+1.p']
    # problems = ['PUZ/PUZ001-1.p', 'PUZ/PUZ002-1.p']
    predictors_all = {'InterceptRegression': InterceptRegression(),
                      'LinearRegression': LinearRegression(fit_intercept=False),
                      'LassoCV': LassoCV(random_state=0),
                      'RidgeCV': RidgeCV(),
                      'ElasticNetCV': ElasticNetCV(random_state=0)}
    predictors = {name: predictors_all[name] for name in ['LinearRegression']}
    seeds = range(100)
    cross_validate = functools.partial(model_selection.cross_validate, scoring=('r2', 'neg_mean_squared_error'), cv=5,
                                       return_train_score=True, error_score='raise')

    dataframes_problem = {}
    dataframes_preference = {}
    precedence_cost_generator = cost_predictor.PrecedenceCostGenerator(solver, random_precedence_types, seeds,
                                                                       cost_normalizer)

    with joblib.parallel_backend('threading', n_jobs=-1):
        records = []
        for predictor_name, predictor in predictors.items():
            log.info('Fitting cost predictor %s to estimate preference matrices.', predictor_name)
            assert predictor_name not in dataframes_problem
            assert predictor_name not in dataframes_preference
            preference_matrices, dataframes_problem[predictor_name], dataframes_preference[
                predictor_name] = cost_predictor.fit_on_problems(problems, precedence_cost_generator, predictor,
                                                                 cross_validate)
            for symbol_type, data in preference_matrices.items():
                log.info('Fitting preference predictor for %s symbols.', symbol_type)
                record = {'predictor': predictor_name, 'symbol_type': symbol_type}
                x = None
                try:
                    x, y = batch.generate_batch(data, 1000, rng=np.random.RandomState(0))
                except ValueError:
                    log.debug('Failed to generate batch.', exc_info=True)
                    record['error'] = 'ValueError: Failed to generate batch.'
                if x is not None:
                    reg = ElasticNetCV(random_state=0)
                    rec = fit(reg, x, y, cross_validate)
                    record.update(rec)
                    # TODO: Evaluate on each problem.
                    record.update(evaluate(reg, precedence_cost_generator, problems, symbol_type))
                print(record)
                records.append(record)
        log.info('Exporting dataframes')
        main_df_problems = pd.concat(dataframes_problem, names=('predictor', 'problem'))
        save_df(main_df_problems, 'predictor_problem', 'out')
        main_df_preferences = pd.concat(dataframes_preference, names=('predictor', 'symbol_type', 'problem'))
        save_df(main_df_preferences, 'predictor_symbol_type_problem', 'out')
        # TODO: Aggregate for each predictor and symbol_type across problems.
        # TODO: Plot distributions across problems.
        # TODO: Distplot: For each symbol type, across problems: cost_raw_nonnan
        # TODO: Distplot: For each predictor and symbol type: test_*_mean, fit_time_mean, n_iter_, alpha_, intercept_, coef_nonzero, coef_count, score_*


def evaluate(preference_predictor, precedence_cost_generator, problems, symbol_type):
    records = {}
    for problem in problems:
        try:
            # Throws RuntimeError when clausification fails.
            records[problem] = evaluate_on_problem(preference_predictor, problem, precedence_cost_generator,
                                                   symbol_type)
        except RuntimeError:
            logging.warning('%s: Failed to evaluate.', problem, exc_info=True)
    return records


def evaluate_on_problem(preference_predictor, problem, precedence_cost_generator, symbol_type):
    # Throws RuntimeError when clausification fails.
    precedence_dicts, y_true, symbols_by_type, rec = precedence_cost_generator.costs_of_random_precedences(
        problem)
    precedences = np.asarray([precedence_dict[symbol_type] for precedence_dict in precedence_dicts])
    symbols = symbols_by_type[symbol_type]
    y_pred = np.fromiter(predict_precedence_costs(preference_predictor, precedences, symbols),
                         dtype=precedence_cost_generator.dtype_cost, count=len(precedences))
    return sklearn_extensions.scores(y_true, y_pred)


def predict_precedence_costs(preference_predictor, precedences, symbols):
    for precedence in precedences:
        yield predict_cost_of_order_matrix(preference_predictor, precedence, symbols)


def predict_cost_of_order_matrix(preference_predictor, precedence, symbols):
    embeddings = symbol_pair_embeddings(precedence, symbols)
    y_pred = preference_predictor.predict(embeddings)
    return np.mean(y_pred)


def symbol_pair_embeddings(precedence, symbols):
    order_matrix = cost_predictor.order_matrices([precedence])[0]
    l, r = np.nonzero(order_matrix)
    index_pairs = np.concatenate((l.reshape(-1, 1), r.reshape(-1, 1)), axis=1)
    return batch.get_symbol_pair_embeddings(symbols, index_pairs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(threadName)s %(levelname)s - %(message)s')
    main()
