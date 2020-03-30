import logging
import warnings

import numpy as np
import scipy
import sklearn.base
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

import vampire_ml.precedence
from utils import memory
from vampire_ml.sklearn_extensions import Flattener


class StaticTransformer(TransformerMixin):
    def fit(self, x, y=None):
        return self


class RunGenerator(BaseEstimator, StaticTransformer):
    def __init__(self, runs_per_problem, random_predicates, random_functions):
        self.runs_per_problem = runs_per_problem
        self.random_predicates = random_predicates
        self.random_functions = random_functions

    def transform(self, problem):
        return memory.cache(type(self)._transform)(self, problem)

    def _transform(self, problem):
        """Runs Vampire on the problem repeatedly and collects the results into arrays."""
        # TODO: Exhaust all precedences on small problems.
        executions = problem.solve_with_random_precedences(solve_count=self.runs_per_problem,
                                                           random_predicates=self.random_predicates,
                                                           random_functions=self.random_functions)
        # We use the type `np.float` to support `np.nan` values.
        base_scores = np.empty(self.runs_per_problem, dtype=np.float)
        precedences = {
            symbol_type: np.empty((self.runs_per_problem, len(problem.get_symbols(symbol_type))), dtype=np.uint) for
            symbol_type in self.symbol_types()}
        for i, execution in enumerate(executions):
            assert i < self.runs_per_problem
            base_scores[i] = execution.base_score()
            for symbol_type, precedence_matrix in precedences.items():
                precedence_matrix[i] = execution.configuration.precedences[self.precedence_option(symbol_type)]
        return precedences, base_scores

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


class PreferenceMatrixTransformer(BaseEstimator, StaticTransformer):
    """
    Predicts a symbol preference matrix dictionary for each problem in isolation.
    Does not generalize across problems.
    """

    def __init__(self, run_generator, score_scaler, score_predictor):
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

    def transform(self, problems):
        """Transforms each of the given problems into a dictionary of symbol preference matrices."""
        with tqdm(problems, desc='Calculating problem preferences', unit='problem', total=len(problems),
                  disable=len(problems) <= 1) as t:
            for problem in t:
                t.set_postfix_str(problem)
                yield self.transform_one(problem)

    def transform_one(self, problem):
        return memory.cache(type(self)._transform_one)(self, problem)

    def _transform_one(self, problem):
        precedences, scores = self.run_generator.transform(problem)
        if not np.all(np.isnan(scores)):
            # For each problem, we fit an independent copy of target transformer.
            scores = sklearn.base.clone(self.score_scaler).fit_transform(scores.reshape(-1, 1))[:, 0]
        else:
            logging.debug(f'All the scores are nans for problem {problem}.')
        return {symbol_type: self.get_preference_matrix(precedence_matrix, scores) for
                symbol_type, precedence_matrix in precedences.items()}

    def get_preference_matrix(self, precedences, scores):
        if precedences.shape[1] <= 1:
            return np.zeros((precedences.shape[1], precedences.shape[1]), dtype=np.float)
        preference_pipeline = pipeline.Pipeline([
            ('order_matrices', preprocessing.FunctionTransformer(func=self.order_matrices)),
            ('flattener', Flattener())
        ])
        preferences_flattened = preference_pipeline.fit_transform(precedences)
        preference_score_regressor = sklearn.base.clone(self.score_predictor)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            preference_score_regressor.fit(preferences_flattened, scores)
        return preference_pipeline['flattener'].inverse_transform(preference_score_regressor.coef_)[0]

    @staticmethod
    def order_matrices(permutations):
        permutations = np.asarray(permutations)
        m = permutations.shape[0]
        n = permutations.shape[1]
        assert permutations.shape == (m, n)
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

    def __init__(self, problem_matrix, pair_value, batch_size, weighted=False, incremental_epochs=None):
        """
        :param problem_matrix: Transforms a problem into a preference matrix dictionary.
        :param pair_value: Symbol pair preference value predictor blueprint.
        Predicts preference value from an embedding of a symbol pair.
        :param batch_size: How many symbol pairs should we learn from in each training batch?
        :param weighted: If True, each of the samples is weighted by the absolute of its target value values when
        fitting `pair_value`.
        :param incremental_epochs: How many batches should we train on incrementally?
        If None, the training is performed in one batch.
        """
        self.problem_matrix = problem_matrix
        self.pair_value = pair_value
        self.batch_size = batch_size
        self.weighted = weighted
        self.incremental_epochs = incremental_epochs

    def fit(self, problems):
        preferences = {symbol_type: list() for symbol_type in self.symbol_types()}
        for problem_preferences in self.problem_matrix.fit_transform(problems):
            for symbol_type in self.symbol_types():
                preferences[symbol_type].append(problem_preferences[symbol_type])
        self.pair_value_fitted_ = dict()
        for symbol_type in self.symbol_types():
            reg = sklearn.base.clone(self.pair_value)
            self.pair_value_fitted_[symbol_type] = reg
            assert id(reg) == id(self.pair_value_fitted_[symbol_type])
            if self.incremental_epochs is not None:
                # TODO: Implement early stopping.
                for _ in tqdm(range(self.incremental_epochs), desc=f'Fitting {symbol_type} precedence regressor',
                              unit='epoch'):
                    symbol_pair_embeddings, target_preference_values = self.generate_batch(problems, symbol_type,
                                                                                           preferences[symbol_type])
                    if self.weighted:
                        reg.partial_fit(symbol_pair_embeddings, target_preference_values,
                                        sample_weight=np.abs(target_preference_values))
                    else:
                        reg.partial_fit(symbol_pair_embeddings, target_preference_values)
            else:
                logging.info(f'Fitting {symbol_type} precedence regressor on {self.batch_size} samples...')
                symbol_pair_embeddings, target_preference_values = self.generate_batch(problems, symbol_type,
                                                                                       preferences[symbol_type])
                if self.weighted:
                    reg.fit(symbol_pair_embeddings, target_preference_values,
                            sample_weight=np.abs(target_preference_values))
                else:
                    reg.fit(symbol_pair_embeddings, target_preference_values)
        return self

    def transform(self, problems):
        """Estimate symbol preference matrix for each problem."""
        # TODO: Predict the preferences for all problems in one call to `self.reg.predict`.
        for problem in problems:
            yield {symbol_type: self.predict_one(problem, symbol_type) for symbol_type in self.symbol_types()}

    def symbol_types(self):
        return self.problem_matrix.run_generator.symbol_types()

    def predict_one(self, problem, symbol_type):
        reg = self.pair_value_fitted_[symbol_type]
        n = len(problem.get_symbols(symbol_type))
        l, r = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        symbol_indexes = np.concatenate((l.reshape(-1, 1), r.reshape(-1, 1)), axis=1)
        symbol_pair_embeddings = problem.get_symbol_pair_embeddings(symbol_type, symbol_indexes)
        return reg.predict(symbol_pair_embeddings).reshape(n, n)

    def generate_batch(self, problems, symbol_type, preferences):
        assert len(problems) == len(preferences)
        symbol_pair_embeddings = list()
        target_preference_values = list()
        problem_indexes = np.random.choice(len(problems), size=self.batch_size)
        for problem_i, n_samples in zip(*np.unique(problem_indexes, return_counts=True)):
            symbol_pair_embedding, target_preference_value = self.generate_sample_from_preference_matrix(
                problems[problem_i], symbol_type, preferences[problem_i], n_samples)
            symbol_pair_embeddings.append(symbol_pair_embedding)
            target_preference_values.append(target_preference_value)
        return np.concatenate(symbol_pair_embeddings), np.concatenate(target_preference_values)

    @classmethod
    def generate_sample_from_preference_matrix(cls, problem, symbol_type, preference_matrix, n_samples):
        symbol_indexes = np.random.choice(len(problem.get_symbols(symbol_type)), size=(n_samples, 2))
        symbol_pair_embedding = problem.get_symbol_pair_embeddings(symbol_type, symbol_indexes)
        target_preference_value = preference_matrix[symbol_indexes[:, 0], symbol_indexes[:, 1]]
        return symbol_pair_embedding, target_preference_value


class GreedyPrecedenceGenerator(BaseEstimator, StaticTransformer):
    @staticmethod
    def transform(preference_dicts):
        """For each preference dictionary yields a precedence dictionary."""
        for preference_dict in preference_dicts:
            yield {f'{symbol_type}_precedence': vampire_ml.precedence.learn_ltot(preference_matrix, symbol_type) for
                   symbol_type, preference_matrix in preference_dict.items()}
            # TODO: Experiment with hill climbing.


class RandomPrecedenceGenerator(BaseEstimator, StaticTransformer):
    def __init__(self, random_predicates, random_functions):
        self.random_predicates = random_predicates
        self.random_functions = random_functions

    def transform(self, problems):
        """For each problem yields a precedence dictionary."""
        for problem in problems:
            yield problem.random_precedences(self.random_predicates, self.random_functions, seed=0)


class ScorerMean:
    def __call__(self, estimator, problems, y=None):
        precedence_dicts = estimator.transform(problems)
        scores = list()
        with tqdm(zip(problems, precedence_dicts), desc=str(self), unit='problem', total=len(problems)) as t:
            stats = dict()
            for problem, precedence_dict in t:
                stats['problem'] = problem
                t.set_postfix(stats)
                try:
                    score_transformed = self.get_score(problem, precedence_dict)
                    assert not np.isnan(score_transformed)
                except FloatingPointError:
                    # FloatingPointError occurs when all the random runs on the problem fail
                    # so we have no baseline to normalize the score.
                    logging.debug(f'Failed to determine the score on problem {problem}.', exc_info=True)
                    score_transformed = np.nan
                scores.append(score_transformed)
                stats['score'] = np.nanmean(scores)
                t.set_postfix(stats)
        return np.nanmean(scores)

    def get_score(self, problem, precedence_dict):
        execution = problem.get_execution(precedences=precedence_dict)
        return self.get_execution_score(problem, execution)

    def get_execution_score(self, problem, execution):
        raise NotImplementedError


class ScorerSuccessRate(ScorerMean):
    def __str__(self):
        return type(self).__name__

    def get_execution_score(self, problem, execution):
        return execution['exit_code'] == 0


class ScorerSaturationIterations(ScorerMean):
    def __init__(self, run_generator, score_scaler):
        self.run_generator = run_generator
        self.score_scaler = score_scaler

    def __repr__(self):
        return f'{type(self).__name__}({self.run_generator}, {self.score_scaler})'

    def __str__(self):
        return type(self).__name__

    def get_execution_score(self, problem, execution):
        return -self.get_fitted_target_transformer(problem).transform([[execution.base_score()]])[0, 0]

    def get_fitted_target_transformer(self, problem):
        _, base_scores = self.run_generator.transform(problem)
        return sklearn.base.clone(self.score_scaler).fit(base_scores.reshape(-1, 1))


class ScorerPercentile(ScorerMean):
    def __init__(self, run_generator, kind='rank'):
        self.run_generator = run_generator
        self.kind = kind

    def __repr__(self):
        return f'{type(self).__name__}({self.run_generator}, {self.kind})'

    def __str__(self):
        return f'{type(self).__name__}({self.kind})'

    def get_execution_score(self, problem, execution):
        _, base_scores = self.run_generator.transform(problem)
        base_scores = np.nan_to_num(base_scores, nan=np.inf)
        execution_score = execution.base_score()
        if np.isnan(execution_score):
            execution_score = np.inf
        return scipy.stats.percentileofscore(base_scores, execution_score, kind=self.kind)
