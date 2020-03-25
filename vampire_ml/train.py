import logging
import warnings

import numpy as np
import sklearn.base
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

import vampire_ml.precedence
from utils import memory
from vampire_ml.sklearn_extensions import Flattener


class ProblemToResultsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, runs_per_problem, random_predicates, random_functions):
        self.runs_per_problem = runs_per_problem
        self.random_predicates = random_predicates
        self.random_functions = random_functions

    def fit(self, _):
        return self

    def transform(self, problem):
        return memory.cache(type(self)._transform)(self, problem)

    def _transform(self, problem):
        """Runs Vampire on the problem repeatedly and collects the results into arrays."""
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
        return base_scores, precedences

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


class IsolatedProblemToPreferencesTransformer(BaseEstimator, TransformerMixin):
    """Processes each problem independently."""

    def __init__(self, problem_to_results_transformer, target_transformer, preference_score_regressor):
        self.problem_to_results_transformer = problem_to_results_transformer
        self.target_transformer = target_transformer
        self.preference_score_regressor = preference_score_regressor

    def fit(self, _):
        return self

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
        scores, precedences = self.problem_to_results_transformer.transform(problem)
        if not np.all(np.isnan(scores)):
            # For each problem, we fit an independent copy of target transformer.
            scores = sklearn.base.clone(self.target_transformer).fit_transform(scores.reshape(-1, 1))[:, 0]
        else:
            logging.debug(f'All the scores are nans for problem {problem}.')
        return {symbol_type: self.get_preference_matrix(precedence_matrix, scores) for
                symbol_type, precedence_matrix in precedences.items()}

    def get_preference_matrix(self, precedences, scores):
        if precedences.shape[1] <= 1:
            return np.zeros((precedences.shape[1], precedences.shape[1]), dtype=np.float)
        preference_pipeline = pipeline.Pipeline([
            ('pairs_hit', preprocessing.FunctionTransformer(func=self.pairs_hit)),
            ('flattener', Flattener())
        ])
        preferences_flattened = preference_pipeline.fit_transform(precedences)
        preference_score_regressor = sklearn.base.clone(self.preference_score_regressor)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            preference_score_regressor.fit(preferences_flattened, scores)
        return preference_pipeline['flattener'].inverse_transform(preference_score_regressor.coef_)[0]

    @staticmethod
    def pairs_hit(precedences):
        precedences = np.asarray(precedences)
        m = precedences.shape[0]
        n = precedences.shape[1]
        assert precedences.shape == (m, n)
        res = np.empty((m, n, n), np.bool)
        precedence_inverse = np.empty(n, precedences.dtype)
        # TODO: Consider optimizing this loop.
        for i, precedence in enumerate(precedences):
            # https://stackoverflow.com/a/25535723/4054250
            precedence_inverse[precedence] = np.arange(n)
            res[i] = np.tri(n, k=-1, dtype=np.bool).transpose()[precedence_inverse, :][:, precedence_inverse]
        return res


class JointProblemToPreferencesTransformer(BaseEstimator, TransformerMixin):
    """Processes a batch of problems jointly."""

    def __init__(self, isolated_problem_to_preference, preference_regressors, batch_size, incremental_epochs=None):
        """
        :param preference_regressor: estimates preference of a pair of symbols based on the embeddings of the symbols.
        """
        self.isolated_problem_to_preference = isolated_problem_to_preference
        self.preference_regressors = preference_regressors
        self.batch_size = batch_size
        self.incremental_epochs = incremental_epochs

    def fit(self, problems):
        preferences = {symbol_type: list() for symbol_type in self.symbol_types()}
        for problem_preferences in self.isolated_problem_to_preference.fit_transform(problems):
            for symbol_type in self.symbol_types():
                preferences[symbol_type].append(problem_preferences[symbol_type])
        for symbol_type in self.symbol_types():
            reg = self.preference_regressors[symbol_type]
            if reg is None:
                continue
            if self.incremental_epochs is not None:
                # TODO: Implement early stopping.
                for _ in tqdm(range(self.incremental_epochs), desc=f'Fitting {symbol_type} precedence regressor',
                              unit='epoch'):
                    symbol_pair_embeddings, target_preference_values = self.generate_batch(problems, symbol_type,
                                                                                           preferences[symbol_type])
                    reg.partial_fit(symbol_pair_embeddings, target_preference_values)
            else:
                logging.info(f'Fitting {symbol_type} precedence regressor on {self.batch_size} samples...')
                symbol_pair_embeddings, target_preference_values = self.generate_batch(problems, symbol_type,
                                                                                       preferences[symbol_type])
                reg.fit(symbol_pair_embeddings, target_preference_values)
        return self

    def transform(self, problems):
        """Estimate symbol preference matrix for each problem."""
        # TODO: Predict the preferences for all problems in one call to `self.reg.predict`.
        for problem in problems:
            yield {symbol_type: self.predict_one(problem, symbol_type) for symbol_type in self.symbol_types()}

    def symbol_types(self):
        if self.preference_regressors is None:
            return frozenset()
        return frozenset(self.preference_regressors.keys()) & frozenset(
            self.isolated_problem_to_preference.problem_to_results_transformer.symbol_types())

    def predict_one(self, problem, symbol_type):
        reg = self.preference_regressors[symbol_type]
        if reg is None:
            return None
        n = len(problem.get_symbols(symbol_type))
        l, r = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        symbol_indexes = np.concatenate((l.reshape(-1, 1), r.reshape(-1, 1)), axis=1)
        symbol_pair_embeddings = self.get_symbol_pair_embeddings(problem, symbol_type, symbol_indexes)
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
        symbol_pair_embedding = cls.get_symbol_pair_embeddings(problem, symbol_type, symbol_indexes)
        target_preference_value = preference_matrix[symbol_indexes[:, 0], symbol_indexes[:, 1]]
        return symbol_pair_embedding, target_preference_value

    @staticmethod
    def get_symbol_pair_embeddings(problem, symbol_type, symbol_indexes):
        n_samples = len(symbol_indexes)
        assert symbol_indexes.shape == (n_samples, 2)
        problem_embeddings = np.asarray(problem.get_embedding()).reshape(1, -1).repeat(n_samples, axis=0)
        symbol_embeddings = problem.get_symbols_embedding(symbol_type, symbol_indexes.flatten()).reshape(n_samples,
                                                                                                         2, -1)
        return np.concatenate((problem_embeddings, symbol_embeddings[:, 0], symbol_embeddings[:, 1]), axis=1)


class PreferenceToPrecedenceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    @staticmethod
    def transform(preference_dicts):
        for preference_dict in preference_dicts:
            yield {f'{symbol_type}_precedence': vampire_ml.precedence.learn_ltot(preference_matrix, symbol_type) for
                   symbol_type, preference_matrix in preference_dict.items()}
            # TODO: Experiment with hill climbing.


class ScorerMean:
    def __init__(self, name):
        self.name = name

    def __call__(self, estimator, problems, y=None):
        precedence_dicts = estimator.transform(problems)
        scores = list()
        with tqdm(zip(problems, precedence_dicts), desc=f'Mean score ({self.name})', unit='problem',
                  total=len(problems)) as t:
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
    def __init__(self):
        super().__init__('success rate')

    def get_execution_score(self, problem, execution):
        return execution['exit_code'] == 0


class ScorerSaturationIterations(ScorerMean):
    def __init__(self, problem_to_results_transformer, target_transformer):
        super().__init__('saturation iterations')
        self.problem_to_results_transformer = problem_to_results_transformer
        self.target_transformer = target_transformer

    def __repr__(self):
        return f'{type(self)}({self.problem_to_results_transformer}, {self.target_transformer})'

    def get_execution_score(self, problem, execution):
        return -self.get_fitted_target_transformer(problem).transform([[execution.base_score()]])[0, 0]

    def get_fitted_target_transformer(self, problem):
        base_scores, _ = self.problem_to_results_transformer.transform(problem)
        return sklearn.base.clone(self.target_transformer).fit(base_scores.reshape(-1, 1))
