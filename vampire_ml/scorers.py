import logging
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn.base

import vampyre
from utils import ProgressBar
from utils import memory
from .sklearn_extensions import get_feature_weights

dtype_score = np.float


def get_base_score(problem, precedence_dict):
    if precedence_dict is None:
        return np.nan
    else:
        return problem.get_execution(precedences=precedence_dict).base_score()


@memory.cache
def get_base_scores(estimator, problems):
    precedence_dicts = estimator.transform(problems)
    scores = np.empty(len(problems), dtype=dtype_score)
    with ProgressBar(zip(problems, precedence_dicts), desc='Computing base scores', unit='problem',
                     total=len(problems)) as t:
        for i, (problem, precedence_dict) in enumerate(t):
            logging.debug(f'Computing base score on problem {problem}.')
            t.set_postfix({'problem': problem})
            score = get_base_score(problem, precedence_dict)
            problem.cache_clear()
            scores[i] = score
    return scores


class ScoreAggregator:
    def __init__(self, aggregate=np.mean):
        self.aggregate = aggregate

    def __call__(self, estimator, problems, y=None):
        res = np.nan
        if len(problems) > 0:
            res = self.aggregate(list(self.get_scores(estimator, problems)))
        logging.info(f'{self}: {res}')
        return res

    def get_scores(self, estimator, problems):
        for problem, base_score in zip(problems, get_base_scores(estimator, problems)):
            yield self.transform_score(base_score, problem)

    def transform_score(self, base_score, problem):
        raise NotImplementedError


class ScorerSuccess(ScoreAggregator):
    def __init__(self, aggregate=np.mean):
        super().__init__(aggregate=aggregate)

    def __str__(self):
        return f'{type(self).__name__}({self.aggregate.__name__})'

    def transform_score(self, base_score, problem):
        return not np.isnan(base_score)


class ScorerSuccessRelative(ScoreAggregator):
    def __init__(self, baseline_estimator, mode='better', aggregate=np.sum):
        super().__init__(aggregate=aggregate)
        self.baseline_estimator = baseline_estimator
        self.mode = mode

    def __str__(self):
        return f'{type(self).__name__}({self.mode})'

    def get_scores(self, estimator, problems):
        baseline_scores = get_base_scores(self.baseline_estimator, problems)
        current_scores = get_base_scores(estimator, problems)
        for baseline_score, current_score in zip(baseline_scores, current_scores):
            yield self.get_score(baseline_score, current_score)

    def get_score(self, baseline_score, current_score):
        baseline_success = not np.isnan(baseline_score)
        this_success = not np.isnan(current_score)
        if self.mode == 'better':
            return this_success and not baseline_success
        if self.mode == 'worse':
            return -(baseline_success and not this_success)
        assert False


class ScorerSaturationIterations(ScoreAggregator):
    def __init__(self, run_generator, score_scaler):
        super().__init__()
        self.run_generator = run_generator
        self.score_scaler = score_scaler

    def __repr__(self):
        return f'{type(self).__name__}({self.run_generator}, {self.score_scaler})'

    def __str__(self):
        return type(self).__name__

    def transform_score(self, base_score, problem):
        _, base_scores = self.run_generator.transform_one(problem)
        if base_scores is None or np.isnan(base_scores).all():
            # The problem is too difficult to scale the scores. All predictions score 0.
            return 0
        transformer = sklearn.base.clone(self.score_scaler).fit(base_scores.reshape(-1, 1))
        return -transformer.transform([[base_score]])[0, 0]


class ScorerPercentile(ScoreAggregator):
    def __init__(self, run_generator, kind='rank'):
        super().__init__()
        self.run_generator = run_generator
        self.kind = kind

    def __repr__(self):
        return f'{type(self).__name__}({self.run_generator}, {self.kind})'

    def __str__(self):
        return f'{type(self).__name__}({self.kind})'

    def transform_score(self, execution_score, problem):
        _, base_scores = self.run_generator.transform_one(problem)
        if base_scores is None:
            # We assume that all the runs would fail.
            base_scores = np.repeat(np.inf, self.run_generator.runs_per_problem)
        base_scores = np.nan_to_num(base_scores, nan=np.inf)
        if np.isnan(execution_score):
            execution_score = np.inf
        percentile = scipy.stats.percentileofscore(base_scores, execution_score, kind=self.kind)
        assert 0 <= percentile <= 100
        return 100 - percentile


class ScorerExplainer():
    def __init__(self, symbol_types=('predicate', 'function')):
        self.symbol_types = symbol_types
        self.weights = dict()

    def __call__(self, estimator, problems, y=None):
        assert estimator not in self.weights
        self.weights[estimator] = dict()
        for symbol_type in self.symbol_types:
            try:
                weights = get_feature_weights(estimator['precedence']['preference'].pair_value_fitted_[symbol_type])
                with np.printoptions(suppress=True, precision=2, linewidth=sys.maxsize):
                    logging.debug(f'{symbol_type} feature weights: {weights}')
            except (TypeError, AttributeError, KeyError, RuntimeError):
                logging.debug(f'Failed to extract {symbol_type} feature weights.', exc_info=True)
                weights = None
            self.weights[estimator][symbol_type] = weights
        return np.nan

    def get_dataframe(self, symbol_type):
        column_names = vampyre.vampire.Problem.get_symbol_pair_embedding_column_names(symbol_type)
        data = {column_name: list() for column_name in column_names}
        for weights_dict in self.weights.values():
            weights = weights_dict[symbol_type]
            for i, column_name in enumerate(column_names):
                try:
                    data[column_name].append(weights[i])
                except TypeError:
                    data[column_name].append(None)
        return pd.DataFrame(data, index=self.weights.keys())
