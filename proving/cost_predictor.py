import logging
import warnings

import joblib
import numpy as np
import sklearn
from vampire_ml import sklearn_extensions

from proving import utils
from proving.memory import memory
from vampire_ml.sklearn_extensions import Flattener
from vampire_ml.sklearn_extensions import fit

log = logging.getLogger(__name__)


class PrecedenceCostGenerator:
    def __init__(self, solver, random_precedence_types, seeds, cost_normalizer_template, dtype_cost=np.float):
        self.solver = solver
        self.random_precedence_types = random_precedence_types
        self.seeds = seeds
        self.cost_normalizer_template = cost_normalizer_template
        self.dtype_cost = dtype_cost

    def costs_of_random_precedences(self, problem):
        problem_record = {}
        # Throws RuntimeError when clausification fails.
        results, symbols_by_type = self.solver.costs_of_random_precedences(problem,
                                                                           random_precedence_types=self.random_precedence_types,
                                                                           seeds=self.seeds)
        problem_record['sample_count'] = len(results)
        precedence_dicts, costs = zip(*results)
        costs = np.asarray(costs, dtype=self.dtype_cost)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            problem_record.update(utils.named_array_stats(costs, 'costs_raw'))
        cost_normalizer = sklearn.base.clone(self.cost_normalizer_template)
        costs_normalized = cost_normalizer.fit_transform(costs.reshape(-1, 1)).reshape(-1)
        problem_record.update(utils.named_array_stats(costs_normalized, 'costs_normalized'))
        return precedence_dicts, costs_normalized, symbols_by_type, problem_record


def fit_on_problems(problems, precedence_cost_generator, predictor, cross_validate):
    preference_matrices = {symbol_type: {} for symbol_type in precedence_cost_generator.random_precedence_types}
    problem_records = []
    predictor_records = []
    for problem in problems:
        log.info('Processing problem %s', problem)
        parameters = (problem, precedence_cost_generator, predictor, cross_validate)
        if __debug__:
            hash_before = joblib.hash(parameters)
        problem_preference_matrices, problem_record, symbol_type_records = fit_on_problem(*parameters)
        assert hash_before == joblib.hash(parameters)
        for symbol_type, d in problem_preference_matrices.items():
            assert symbol_type in precedence_cost_generator.random_precedence_types
            assert problem not in preference_matrices[symbol_type]
            preference_matrices[symbol_type][problem] = d
        problem_records.append(problem_record)
        predictor_records.extend(symbol_type_records)
    df_problems = utils.dataframe_from_records(problem_records, 'problem')
    df_predictors = utils.dataframe_from_records(predictor_records, ['symbol_type', 'problem'])
    return preference_matrices, df_problems, df_predictors


#@memory.cache
def fit_on_problem(problem, precedence_cost_generator, predictor_template, cross_validate):
    log.info('Fitting on %s', problem)
    preference_matrices = {}
    symbol_type_records = []
    problem_record = {'problem': problem}
    try:
        # Throws RuntimeError when clausification fails.
        precedence_dicts, costs_normalized, symbols_by_type, rec = precedence_cost_generator.costs_of_random_precedences(
            problem)
        problem_record.update(rec)
        if np.count_nonzero(~np.isnan(costs_normalized)) == 0:
            raise RuntimeError('All costs are nan.')
        for symbol_type in precedence_cost_generator.random_precedence_types:
            symbol_type_record = {'problem': problem, 'symbol_type': symbol_type}
            precedences = np.asarray([precedence_dict[symbol_type] for precedence_dict in precedence_dicts])
            preference_matrix, row = fit_on_precedences(precedences, costs_normalized,
                                                        sklearn.base.clone(predictor_template), cross_validate)
            assert symbol_type not in preference_matrices
            if preference_matrix is not None:
                preference_matrices[symbol_type] = {'preference_matrix': preference_matrix,
                                                    'symbols': symbols_by_type[symbol_type]}
            # TODO: Evaluate how good the preference matrix is if we alias symbols with identical embeddings.
            symbol_type_record.update(row)
            symbol_type_records.append(symbol_type_record)
            # TODO: Cripple the predictor by aliasing by symbol embeddings.
            # TODO: Train a predictor on embeddings on one problem and symbol_type.
    except RuntimeError:
        log.debug('%s: Failed to estimate preference matrix.', problem, exc_info=True)
        problem_record['error'] = 'RuntimeError: Failed to estimate preference matrix.'
    return preference_matrices, problem_record, symbol_type_records


def evaluate_aliasing(preference_matrix, symbols, precedence):
    pass


def fit_on_precedences(precedences, y_true, estimator, cross_validate):
    assert len(precedences) == len(y_true)
    n_symbols = precedences.shape[1]
    record = {'estimator_type': type(estimator).__name__,
              'sample_count': len(y_true),
              'symbol_count': n_symbols}
    record.update(utils.named_array_stats(y_true, 'targets'))
    preference_matrix = None
    try:
        # Throws MemoryError if the output would be too large.
        order_mats = order_matrices(precedences)
        flattener = Flattener()
        x = flattener.fit_transform(order_mats)
        record['feature_count'] = x.shape[1]
        # Throws ValueError if targets contain a nan.
        rec = fit(estimator, x, y_true, cross_validate)
        record.update(rec)
        symbol_pairs_per_order_matrix = n_symbols * (n_symbols - 1) / 2
        preference_vector = estimator.coef_ * symbol_pairs_per_order_matrix
        # We scale the coefficients by n choose 2 to ensure that averaging over symbol pairs hit yields the cost estimate.
        # TODO: Cripple the predictions by aliasing symbols. Create a crippled preference vector.
        assert np.allclose(np.mean(preference_vector * x, axis=1) * (n_symbols * n_symbols / symbol_pairs_per_order_matrix), estimator.predict(x))
        preference_matrix = flattener.inverse_transform(preference_vector)[0]
        # TODO: Consider using a sparse representation of the output.
    except (MemoryError, ValueError) as e:
        log.warning('Failed to fit estimator on precedences.', exc_info=True)
        record['error'] = type(e).__name__
    return preference_matrix, record


def order_matrices(permutations):
    m = permutations.shape[0]
    n = permutations.shape[1]
    assert permutations.shape == (m, n)
    # Throws MemoryError if the array is too large.
    res = np.empty((m, n, n), np.bool)
    precedence_inverse = np.empty(n, permutations.dtype)
    # TODO: Consider optimizing this loop.
    for i, precedence in enumerate(permutations):
        # https://stackoverflow.com/a/25535723/4054250
        precedence_inverse[precedence] = np.arange(n)
        res[i] = np.tri(n, k=-1, dtype=np.bool).transpose()[precedence_inverse, :][:, precedence_inverse]
    return res
