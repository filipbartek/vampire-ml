import numpy as np
from sklearn.base import BaseEstimator

import vampire_ml.precedence
from vampire_ml.sklearn_extensions import StaticTransformer


class GreedyPrecedenceGenerator(BaseEstimator, StaticTransformer):
    @staticmethod
    def transform(preference_dicts):
        """For each preference dictionary yields a precedence dictionary."""
        for preference_dict in preference_dicts:
            if preference_dict is None or any(v is None for v in preference_dict.values()):
                yield None
                continue
            yield {f'{symbol_type}_precedence': vampire_ml.precedence.learn_ltot(preference_matrix, symbol_type) for
                   symbol_type, preference_matrix in preference_dict.items()}


class RandomPrecedenceGenerator(BaseEstimator, StaticTransformer):
    def __init__(self, random_predicates, random_functions, seed=0):
        self.random_predicates = random_predicates
        self.random_functions = random_functions
        self.seed = seed

    def transform(self, problems):
        """For each problem yields a precedence dictionary."""
        for problem in problems:
            try:
                yield problem.random_precedences(self.random_predicates, self.random_functions, seed=self.seed)
            except RuntimeError:
                yield None


class BestPrecedenceGenerator(BaseEstimator, StaticTransformer):
    def __init__(self, run_generator):
        self.run_generator = run_generator

    def transform(self, problems):
        """For each problem yields a precedence dictionary."""
        for precedences, base_scores in self.run_generator.transform(problems):
            if base_scores is None:
                yield None
                continue
            base_scores = np.nan_to_num(base_scores, nan=np.inf)
            best_i = np.argmin(base_scores)
            yield {f'{symbol_type}_precedence': precedence_matrix[best_i] for symbol_type, precedence_matrix in
                   precedences.items()}
