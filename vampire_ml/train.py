import collections
import logging
import sys
import warnings

import numpy as np
import sklearn.base
from joblib import Parallel, delayed
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

import vampyre
from utils import ProgressBar
from utils import memory
from vampire_ml.sklearn_extensions import Flattener
from vampire_ml.sklearn_extensions import StaticTransformer
from vampire_ml.sklearn_extensions import get_feature_weights


@memory.cache
def run_generator_transform_one(self, problem):
    return self._transform_one(problem)


class RunGenerator(BaseEstimator, StaticTransformer):
    def __init__(self, runs_per_problem, random_predicates, random_functions):
        self.runs_per_problem = runs_per_problem
        self.random_predicates = random_predicates
        self.random_functions = random_functions

    dtype_execution_score = np.float
    dtype_precedence = vampyre.vampire.Problem.dtype_precedence

    def transform(self, problems):
        logging.info(f'Solving each of {len(problems)} problems {self.runs_per_problem} times')
        return Parallel(verbose=1)(delayed(self.transform_one)(problem) for problem in problems)

    def transform_one(self, problem):
        if memory.recompute:
            run_generator_transform_one.call_and_shelve(self, problem).clear()
        return run_generator_transform_one(self, problem)

    def _transform_one(self, problem):
        """Runs Vampire on the problem repeatedly and collects the results into arrays."""
        # TODO: Consider: Skip running if the signature is too large to learn from.
        try:
            # We use the type `np.float` to support `np.nan` values.
            base_scores = np.empty(self.runs_per_problem, dtype=self.dtype_execution_score)
            precedences = {
                symbol_type: np.empty((self.runs_per_problem, len(problem.get_symbols(symbol_type))),
                                      dtype=self.dtype_precedence) for
                symbol_type in self.symbol_types()}
            for i, execution in enumerate(self.get_executions(problem)):
                assert i < self.runs_per_problem
                base_scores[i] = execution.base_score()
                for symbol_type, precedence_matrix in precedences.items():
                    precedence_matrix[i] = execution.configuration.precedences[self.precedence_option(symbol_type)]
            return precedences, base_scores
        except RuntimeError:
            logging.debug(f'Failed to generate runs on problem {problem}.', exc_info=True)
            return None, None

    def get_executions(self, problem, progress_bar=True):
        # TODO: Exhaust all precedences on small problems.
        return problem.solve_with_random_precedences(solve_count=self.runs_per_problem,
                                                     random_predicates=self.random_predicates,
                                                     random_functions=self.random_functions,
                                                     progress_bar=progress_bar)

    def symbol_types(self):
        res = list()
        if self.random_predicates:
            res.append('predicate')
        if self.random_functions:
            res.append('function')
        return res

    @staticmethod
    def precedence_option(symbol_type):
        return {'predicate': 'predicate_precedence', 'function': 'function_precedence'}[symbol_type]


@memory.cache
def preference_matrix_transformer_transform_one(self, problem):
    return self._transform_one(problem)


class PreferenceMatrixTransformer(BaseEstimator, StaticTransformer):
    """
    Predicts a symbol preference matrix dictionary for each problem in isolation.
    Does not generalize across problems.
    """

    def __init__(self, run_generator, score_scaler, score_predictor, max_symbols=None):
        """
        :param run_generator: Run generator. Transforms problem into precedences and scores.
        :param score_scaler: Score scaler blueprint. Scales a batch of scores.
        :param score_predictor: Score predictor blueprint. Predicts score from symbol order matrix.
        Must be a linear model exposing `coef_`. The same predictor is used for all symbol types
        (predicates and functions).
        """
        self.run_generator = run_generator
        self.score_scaler = score_scaler
        self.score_predictor = score_predictor
        self.max_symbols = max_symbols

    def transform(self, problems):
        """Transforms each of the given problems into a dictionary of symbol preference matrices."""
        logging.info(f'Estimating preference matrices for {len(problems)} problems')
        return Parallel(verbose=1)(delayed(self.transform_one)(problem) for problem in problems)

    def transform_one(self, problem):
        if memory.recompute:
            preference_matrix_transformer_transform_one.call_and_shelve(self, problem).clear()
        return preference_matrix_transformer_transform_one(self, problem)

    def _transform_one(self, problem):
        if self.run_generator.random_predicates and len(problem.get_predicates()) > self.max_symbols:
            return None
        if self.run_generator.random_functions and len(problem.get_functions()) > self.max_symbols:
            return None
        precedences, scores = self.run_generator.transform_one(problem)
        if precedences is None or scores is None:
            return None
        # For each problem, we fit an independent copy of target transformer.
        scores = sklearn.base.clone(self.score_scaler).fit_transform(scores.reshape(-1, 1))[:, 0]
        return {symbol_type: self.get_preference_matrix(precedence_matrix, scores) for
                symbol_type, precedence_matrix in precedences.items()}

    dtype_preference = np.float

    def get_preference_matrix(self, precedences, scores):
        if precedences.shape[1] > self.max_symbols:
            return None
        try:
            return self.get_preference_matrix_or_raise(precedences, scores)
        except ValueError:
            logging.debug('Preference matrix fitting failed.', exc_info=True)
            return None

    def get_preference_matrix_or_raise(self, precedences, scores):
        valid_samples = ~np.isnan(scores)
        if not valid_samples.any():
            logging.debug('All the scores are nan. Assuming preference 0 for all symbol pairs.')
            return np.zeros((precedences.shape[1], precedences.shape[1]), dtype=self.dtype_preference)
        if not valid_samples.all():
            warnings.warn(
                f'Omitting {np.count_nonzero(~valid_samples)}/{len(scores)} samples because they are nan-scored when learning problem-specific preference matrix.')
        precedences = precedences[valid_samples]
        scores = scores[valid_samples]
        if (precedences == precedences[0]).all():
            logging.debug('All the valid precedences are identical. Assuming preference 0 for all symbol pairs.')
            return np.zeros((precedences.shape[1], precedences.shape[1]), dtype=self.dtype_preference)
        if (scores == scores[0]).all():
            logging.debug('All the valid scores are identical. Assuming preference 0 for all symbol pairs.')
            return np.zeros((precedences.shape[1], precedences.shape[1]), dtype=self.dtype_preference)
        preference_pipeline = pipeline.Pipeline([
            ('order_matrices', preprocessing.FunctionTransformer(func=self.order_matrices)),
            ('flattener', Flattener())
        ])
        preferences_flattened = preference_pipeline.fit_transform(precedences)
        score_predictor = sklearn.base.clone(self.score_predictor)
        assert not (preferences_flattened == preferences_flattened[0]).all()
        assert (~np.isnan(scores)).all()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=sklearn.exceptions.ConvergenceWarning)
            score_predictor.fit(preferences_flattened, scores)
            # Note: If fitting fails to converge, the coefficients are all zeros.
        # TODO: Consider using a sparse representation of the output.
        return preference_pipeline['flattener'].inverse_transform(score_predictor.coef_)[0]

    @staticmethod
    def order_matrices(permutations):
        permutations = np.asarray(permutations)
        m = permutations.shape[0]
        n = permutations.shape[1]
        assert permutations.shape == (m, n)
        logging.debug(f'Allocating a bool array of size {m * n * n}.')
        res = np.empty((m, n, n), np.bool)
        precedence_inverse = np.empty(n, permutations.dtype)
        # TODO: Consider optimizing this loop.
        for i, precedence in enumerate(permutations):
            # https://stackoverflow.com/a/25535723/4054250
            precedence_inverse[precedence] = np.arange(n)
            res[i] = np.tri(n, k=-1, dtype=np.bool).transpose()[precedence_inverse, :][:, precedence_inverse]
        return res


class PreferenceMatrixPredictor(BaseEstimator, TransformerMixin):
    """
    Predicts a symbol preference matrix dictionary for a problem.
    Generalizes to new problems by exploiting patterns shared by multiple problems.
    Learns from a batch of problems jointly.
    """

    def __init__(self, problem_matrix, pair_value, batch_size, weighted=True, incremental_epochs=None,
                 max_symbols=None, random_state=None):
        """
        :param problem_matrix: Transforms a problem into a preference matrix dictionary.
        :param pair_value: Symbol pair preference value predictor blueprint.
        Predicts preference value from an embedding of a symbol pair.
        :param batch_size: How many symbol pairs should we learn from in each training batch?
        :param weighted: If True, each of the samples is weighted by absolute target value when fitting `pair_value`.
        Moreover, each of the problems is weighted by mean absolute preference value.
        :param incremental_epochs: How many batches should we train on incrementally?
        If None, the training is performed in one batch.
        :param max_symbols: Maximum signature size to predict preference matrix for.
        :param random_state: If int, seed of a random number generator. If None, use `np.random`.
        Otherwise a `RandomState`.
        """
        self.problem_matrix = problem_matrix
        self.pair_value = pair_value
        self.batch_size = batch_size
        self.weighted = weighted
        self.incremental_epochs = incremental_epochs
        self.max_symbols = max_symbols
        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

    weight_dtype = np.float

    def fit(self, problems):
        preferences = self.get_preferences(problems)
        self.pair_value_fitted_ = dict()
        for symbol_type in self.symbol_types():
            reg = sklearn.base.clone(self.pair_value)
            self.pair_value_fitted_[symbol_type] = reg
            assert id(reg) == id(self.pair_value_fitted_[symbol_type])
            if self.incremental_epochs is not None:
                # TODO: Implement early stopping.
                for _ in ProgressBar(range(self.incremental_epochs),
                                     desc=f'Fitting general {symbol_type} preference regressor {type(reg).__name__}',
                                     unit='epoch'):
                    symbol_pair_embeddings, target_preference_values = self.generate_batch(problems, symbol_type,
                                                                                           preferences[symbol_type])
                    reg.partial_fit(symbol_pair_embeddings, target_preference_values)
            else:
                logging.debug(
                    f'General {symbol_type} preference regressor: Generating batch of {self.batch_size} samples...')
                symbol_pair_embeddings, target_preference_values = self.generate_batch(problems, symbol_type,
                                                                                       preferences[symbol_type])
                logging.debug(
                    f'General {symbol_type} preference regressor: Batch of {len(symbol_pair_embeddings)} samples generated.')
                logging.info(
                    f'General {symbol_type} preference regressor: Fitting on {len(symbol_pair_embeddings)} samples...')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=sklearn.exceptions.ConvergenceWarning)
                    reg.fit(symbol_pair_embeddings, target_preference_values)
                logging.info(f'General {symbol_type} preference regressor: Fitted.')
            with np.printoptions(suppress=True, precision=2, linewidth=sys.maxsize):
                try:
                    logging.info(f'Feature weights: {get_feature_weights(reg)}')
                except RuntimeError:
                    logging.debug('Failed to extract feature weights.', exc_info=True)
        return self

    def transform(self, problems):
        """Estimate symbol preference matrix for each problem."""
        # TODO: Predict the preferences for all problems in one call to `self.reg.predict`.
        for problem in problems:
            yield self.transform_one(problem)

    def transform_one(self, problem):
        return {symbol_type: self.predict_one(problem, symbol_type) for symbol_type in self.symbol_types()}

    def symbol_types(self):
        return self.problem_matrix.run_generator.symbol_types()

    def predict_one(self, problem, symbol_type):
        reg = self.pair_value_fitted_[symbol_type]
        try:
            n = len(problem.get_symbols(symbol_type))
            if self.max_symbols is not None and n > self.max_symbols:
                logging.debug(
                    f'{problem} has {n}>{self.max_symbols} {symbol_type} symbols. This is too many to predict a preference matrix.')
                return None
            l, r = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            symbol_indexes = np.concatenate((l.reshape(-1, 1), r.reshape(-1, 1)), axis=1)
            symbol_pair_embeddings = problem.get_symbol_pair_embeddings(symbol_type, symbol_indexes)
            return reg.predict(symbol_pair_embeddings).reshape(n, n)
        except RuntimeError:
            logging.debug(f'Failed to predict preference on problem {problem}.', exc_info=True)
            return None

    PreferenceRecord = collections.namedtuple('PreferenceRecord', ['matrices', 'weights'])

    def get_preferences(self, problems):
        preferences = {symbol_type: self.PreferenceRecord(list(), np.zeros(len(problems), dtype=self.weight_dtype))
                       for symbol_type in self.symbol_types()}
        for i, problem_preferences in enumerate(self.problem_matrix.fit_transform(problems)):
            for symbol_type in self.symbol_types():
                if problem_preferences is None:
                    preferences[symbol_type].matrices.append(None)
                    continue
                matrix = problem_preferences[symbol_type]
                preferences[symbol_type].matrices.append(matrix)
                if matrix is not None and matrix.size > 0:
                    preferences[symbol_type].weights[i] = np.mean(np.abs(matrix))
        return preferences

    def generate_batch(self, problems, symbol_type, preference_record):
        assert len(problems) == len(preference_record.matrices) == len(preference_record.weights)
        symbol_pair_embeddings = list()
        target_preference_values = list()
        p = None
        if self.weighted:
            assert np.all(preference_record.weights >= 0)
            if np.count_nonzero(preference_record.weights) == 0:
                warnings.warn('All of the problems have all-zero preference matrices. No data to learn from.')
                return list(), list()
            p = preference_record.weights / np.sum(preference_record.weights)
        problem_indexes = self.random_state.choice(len(problems), size=self.batch_size, p=p)
        for problem_i, n_samples in zip(*np.unique(problem_indexes, return_counts=True)):
            problem = problems[problem_i]
            try:
                symbol_pair_embedding, target_preference_value = self.generate_sample_from_preference_matrix(
                    problem, symbol_type, preference_record.matrices[problem_i], n_samples)
                symbol_pair_embeddings.append(symbol_pair_embedding)
                target_preference_values.append(target_preference_value)
            except RuntimeError as e:
                warnings.warn(f'Failed to generate samples from problem {problem}. Cause: {e}')
        return np.concatenate(symbol_pair_embeddings), np.concatenate(target_preference_values)

    def generate_sample_from_preference_matrix(self, problem, symbol_type, preference_matrix, n_samples):
        if preference_matrix is None:
            raise RuntimeError('Cannot learn without a preference matrix.')
        n = len(problem.get_symbols(symbol_type))
        l, r = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        all_pairs_index_pairs = np.concatenate((l.reshape(-1, 1), r.reshape(-1, 1)), axis=1)
        all_pairs_values = preference_matrix.flatten()
        p = None
        if self.weighted:
            if np.allclose(0, all_pairs_values):
                raise RuntimeError('Cannot learn from an all-zero preference matrix.')
            p = np.abs(all_pairs_values) / np.sum(np.abs(all_pairs_values))
        chosen_pairs_indexes = self.random_state.choice(len(all_pairs_index_pairs), size=n_samples, p=p)
        chosen_pairs_index_pairs = all_pairs_index_pairs[chosen_pairs_indexes]
        chosen_pairs_embeddings = problem.get_symbol_pair_embeddings(symbol_type, chosen_pairs_index_pairs)
        chosen_pairs_values = all_pairs_values[chosen_pairs_indexes]
        return chosen_pairs_embeddings, chosen_pairs_values
