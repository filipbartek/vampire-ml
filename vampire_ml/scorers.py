import itertools
import logging
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn.base
from joblib import Parallel, delayed

import vampyre
from utils import ProgressBar
from utils import memory
from .sklearn_extensions import get_feature_weights
from .train import PreferenceMatrixTransformer

dtype_score = np.float


def get_base_score(problem, precedence_dict):
    logging.debug(f'Computing base score on problem {problem}.')
    if precedence_dict is None:
        return np.nan, False
    else:
        return problem.get_execution(precedences=precedence_dict).base_score(), True


@memory.cache
def get_base_scores(estimator, problems):
    logging.info(f'Computing base scores on {len(problems)} problems')
    precedence_dicts = estimator.transform(problems)
    r = Parallel(verbose=1)(delayed(get_base_score)(problem, precedence_dict) for problem, precedence_dict in
                            ProgressBar(zip(problems, precedence_dicts), total=len(problems), unit='problem',
                                        desc='Computing base scores'))
    scores, predictions = zip(*r)
    return np.asarray(scores, dtype=dtype_score), np.asarray(predictions, dtype=np.bool)


class ScoreAggregator:
    def __init__(self, aggregate=np.mean):
        self.aggregate = aggregate

    def __str__(self):
        return f'{type(self).__name__}({self.aggregate.__name__})'

    def __call__(self, estimator, problems, y=None):
        res = np.nan
        if len(problems) > 0:
            try:
                res = self.aggregate(self.get_scores(estimator, problems))
            except RuntimeError:
                logging.debug('Failed to get scores.', exc_info=True)
        logging.info(f'{self}: {res}')
        return res

    def get_scores(self, estimator, problems):
        return np.fromiter(self.generate_scores(estimator, problems), dtype=dtype_score, count=len(problems))

    def generate_scores(self, estimator, problems):
        for problem, base_score in zip(problems, get_base_scores(estimator, problems)[0]):
            yield self.transform_score(base_score, problem)

    def transform_score(self, base_score, problem):
        raise NotImplementedError


class ScorerSuccess(ScoreAggregator):
    def __init__(self, aggregate=np.mean):
        super().__init__(aggregate=aggregate)

    def transform_score(self, base_score, problem):
        return not np.isnan(base_score)


class ScorerSuccessRelative(ScoreAggregator):
    def __init__(self, baseline_estimator, mode='better', aggregate=np.sum):
        super().__init__(aggregate=aggregate)
        self.baseline_estimator = baseline_estimator
        self.mode = mode

    def __str__(self):
        return f'{type(self).__name__}({self.mode})'

    def generate_scores(self, estimator, problems):
        baseline_scores = get_base_scores(self.baseline_estimator, problems)[0]
        current_scores = get_base_scores(estimator, problems)[0]
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
        base_scores = self.run_generator.transform_one(problem)[1]
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
        base_scores = self.run_generator.transform_one(problem)[1]
        if base_scores is None:
            # We assume that all the runs would fail.
            base_scores = np.repeat(np.inf, self.run_generator.runs_per_problem)
        base_scores = np.nan_to_num(base_scores, nan=np.inf)
        if np.isnan(execution_score):
            execution_score = np.inf
        percentile = scipy.stats.percentileofscore(base_scores, execution_score, kind=self.kind)
        assert 0 <= percentile <= 100
        return 100 - percentile


def compare_score_vectors(measured, predicted):
    n = len(predicted)
    assert len(measured) == n
    if n <= 1:
        return np.nan
    measured = np.nan_to_num(measured, nan=np.inf)
    predicted = np.nan_to_num(predicted, nan=np.inf)
    data = {('saturation_iterations', 'total'): 0,
            ('saturation_iterations', 'strict'): 0,
            ('saturation_iterations', 'weak'): 0,
            ('success', 'total'): 0,
            ('success', 'strict'): 0,
            ('success', 'weak'): 0,
            'total_precedences': n}
    # TODO: Optimize.
    for l, r in itertools.product(range(n), range(n)):
        if measured[l] < measured[r]:
            data[('saturation_iterations', 'total')] += 1
            if predicted[l] < predicted[r]:
                data[('saturation_iterations', 'strict')] += 1
            if predicted[l] <= predicted[r]:
                data[('saturation_iterations', 'weak')] += 1
            if np.isinf(measured[r]):
                assert not np.isinf(measured[l])
                data[('success', 'total')] += 1
                if predicted[l] < predicted[r]:
                    data[('success', 'strict')] += 1
                if predicted[l] <= predicted[r]:
                    data[('success', 'weak')] += 1
    return data


@memory.cache
def get_ordering_scores(preference_predictor, problems, run_generator, max_symbols=10000):
    records = {}
    # TODO: Parallelize.
    for problem in ProgressBar(problems, unit='problem', desc='Computing ordering scores'):
        precedences, base_scores = run_generator.transform_one(problem)
        if precedences is None:
            logging.debug(f'{problem}: Failed to generate runs. Skipping.')
            continue
        assert base_scores is not None
        for symbol_type, precedence_matrix in precedences.items():
            if precedence_matrix.shape[1] > max_symbols:
                logging.debug(
                    f'{problem}: {symbol_type}: Too many symbols: {precedence_matrix.shape[1]} > {max_symbols}. Skipping.')
                continue
            order_matrices = PreferenceMatrixTransformer.order_matrices(precedence_matrix)
            preference_matrix = preference_predictor.predict_one(problem, symbol_type)
            if preference_matrix is None:
                # If we got this far, preference matrix prediction should always pass.
                logging.debug(f'{problem}: {symbol_type}: Failed to predict preference matrix. Skipping.')
                continue
            predicted_scores = np.sum(order_matrices * preference_matrix, axis=(1, 2))
            record = compare_score_vectors(base_scores, predicted_scores)
            record['problem'] = problem.path
            if symbol_type not in records:
                records[symbol_type] = []
            records[symbol_type].append(record)
    return {symbol_type: pd.DataFrame.from_records(scores, index='problem') for symbol_type, scores in records.items()}


class ScorerOrdering:
    def __init__(self, run_generator, symbol_type, aggregation='problems', measure='saturation_iterations',
                 comparison='strict', max_symbols=10000):
        self.run_generator = run_generator
        self.symbol_type = symbol_type
        assert aggregation in {'problems', 'samples'}
        self.aggregation = aggregation
        assert measure in {'saturation_iterations', 'success'}
        self.measure = measure
        assert comparison in {'strict', 'weak', 'mean'}
        self.comparison = comparison
        self.max_symbols = max_symbols

    def __str__(self):
        return f'{type(self).__name__}({self.symbol_type}, {self.aggregation}, {self.measure}, {self.comparison})'

    def __call__(self, estimator, problems, y=None):
        res = np.nan
        if len(problems) > 0:
            try:
                hits, totals = self.get_scores(estimator, problems)
                assert np.all(hits >= 0)
                assert np.all(totals >= 0)
                if self.aggregation == 'problems':
                    rates = hits / totals
                    res = np.nanmean(rates)
                else:
                    assert self.aggregation == 'samples'
                    total = np.nansum(totals)
                    if total != 0:
                        res = np.nansum(hits) / total
            except RuntimeError:
                logging.debug('Failed to get scores.', exc_info=True)
        logging.info(f'{self}: {res}')
        return res

    def get_scores(self, estimator, problems):
        try:
            preference_predictor = estimator['precedence']['preference']
        except TypeError as e:
            raise RuntimeError from e
        df = get_ordering_scores(preference_predictor, problems, self.run_generator, max_symbols=self.max_symbols)[
            self.symbol_type]
        if self.comparison == 'mean':
            hits = (df[(self.measure, 'strict')] + df[(self.measure, 'weak')]) / 2
        else:
            hits = df[(self.measure, self.comparison)]
        totals = df[(self.measure, 'total')]
        return hits, totals


class ScorerExplainer:
    def __init__(self, symbol_types=('predicate', 'function')):
        self.symbol_types = symbol_types
        self.weights = dict()

    def __call__(self, estimator, problems=None, y=None):
        if estimator in self.weights:
            return np.nan
        self.weights[estimator] = dict()
        for symbol_type in self.symbol_types:
            try:
                weights = get_feature_weights(estimator['precedence']['preference'].pair_value_fitted_[symbol_type])
                scale = np.abs(weights).sum()
                if scale != 0:
                    weights_normalized = weights / scale
                else:
                    weights_normalized = weights
                with np.printoptions(suppress=True, precision=2, linewidth=sys.maxsize):
                    logging.info(f'{symbol_type} feature weights: {scale} * {weights_normalized}')
            except (TypeError, AttributeError, KeyError, RuntimeError):
                logging.debug(f'Failed to extract {symbol_type} feature weights.', exc_info=True)
                weights_normalized = None
            self.weights[estimator][symbol_type] = weights_normalized
        return np.nan

    def get_dataframe(self):
        records = list()
        for estimator, weights_dict in self.weights.items():
            records.append(np.concatenate(
                [self.get_weights_vector(weights_dict, symbol_type) for symbol_type in self.symbol_types]))
        column_names = list()
        for symbol_type in self.symbol_types:
            column_names.extend(
                (symbol_type, s) for s in
                vampyre.vampire.Problem.get_symbol_pair_embedding_column_names(symbol_type))
        return pd.DataFrame.from_records(records, index=pd.Index(self.weights.keys(), name='estimator'),
                                         columns=pd.MultiIndex.from_tuples(column_names))

    @staticmethod
    def get_weights_vector(weights_dict, symbol_type):
        if weights_dict[symbol_type] is None:
            return np.full(len(vampyre.vampire.Problem.get_symbol_pair_embedding_column_names(symbol_type)), np.nan)
        assert len(weights_dict[symbol_type]) == len(
            vampyre.vampire.Problem.get_symbol_pair_embedding_column_names(symbol_type))
        return weights_dict[symbol_type]


class ScorerPrediction(ScoreAggregator):
    def __init__(self, aggregate=np.sum):
        super().__init__(aggregate)

    def get_scores(self, estimator, problems):
        return get_base_scores(estimator, problems)[1]
