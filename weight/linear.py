import functools
import logging

import clogistic
import cvxpy
import numpy as np
import pandas as pd
import scipy
import sklearn

from questions.utils import timer


def analyze(samples_aggregated, common_clause_features=None, **kwargs):
    def coef_dict(coef):
        if coef is None:
            return {}
        coef = coef.squeeze(axis=0)
        coef_symbol = coef[len(common_clause_features):]
        res = {
            **{k: coef[i] for i, k in enumerate(common_clause_features)},
            'pc': {k: scipy.stats.percentileofscore(coef_symbol, coef[i], kind='strict') for i, k in
                   enumerate(common_clause_features)},
            'symbol': {k: getattr(np, k)(coef_symbol) for k in ['mean', 'min', 'max', 'median']}
        }
        return res

    clause_features = samples_aggregated['token_counts'].toarray()
    clause_proof = samples_aggregated['proof'].toarray().squeeze(axis=1)

    records = []
    for case_name, analyze in {'clause': analyze_clause_classifier,
                               'clause_pair': analyze_clause_pair_classifier}.items():
        try:
            res = analyze(clause_features, clause_proof, **kwargs)
        except (ValueError, cvxpy.error.SolverError) as e:
            logging.warning(str(e))
            continue
        record = {'task': case_name, **{k: v for k, v in res.items() if k not in ['intercept', 'coef']}}
        assert res['intercept'].shape == (1,)
        record['intercept'] = res['intercept'][0]
        if clause_features is not None:
            record['coef'] = {k: coef_dict(v) for k, v in res['coef'].items()}
        records.append(record)
    df = pd.json_normalize(records, sep='_')
    return df


def analyze_clause_classifier(X, y, **kwargs):
    num_true = np.count_nonzero(y)
    num_false = np.count_nonzero(~y)
    if num_true == 0 or num_false == 0:
        raise ValueError('Purely positive or negative clauses.')
    sample_weight = np.where(y, 0.5 / num_true, 0.5 / num_false)

    return fit_all(X, y, sample_weight=sample_weight, fit_intercept=True, **kwargs)


def analyze_clause_pair_classifier(X, y, **kwargs):
    # Coding of y: True corresponds to proof clause.
    # proof, nonproof
    pair_clause_indices = np.meshgrid(np.where(y), np.where(~y), indexing='ij')
    # nonproof - proof
    pair_X = X[pair_clause_indices[1]].astype(np.int32) - X[pair_clause_indices[0]]
    pair_X = pair_X.reshape((-1, pair_X.shape[-1]))
    pair_y = np.ones(pair_X.shape[:-1], dtype=bool)

    # We train without intercept and we need two training classes.
    # We adjust the data by flipping half of the samples.
    pair_y[1::2] = False
    pair_X *= np.expand_dims(np.where(pair_y, 1, -1), 1)

    return fit_all(pair_X, pair_y, fit_intercept=False, **kwargs)


def fit_all(X, y, sample_weight=None, **kwargs):
    logging.debug('Fitting dataset of shape %s...' % str(X.shape))
    with timer() as t_fit:
        model = fit_clogistic(X, y, sample_weight=sample_weight, lb_value=1, **kwargs)
    with timer() as t_score:
        score = model.score(X, y, sample_weight=sample_weight)
    res = {
        **kwargs,
        'score': score,
        'intercept': model.intercept_,
        'coef': {
            'raw': model.coef_,
            'normalized': normalize_min(model.coef_)
        },
        'time_elapsed': {
            'fit': t_fit.elapsed,
            'score': t_score.elapsed
        }
    }
    return res


def fit_sklearn(X, y, sample_weight=None, **kwargs):
    model = sklearn.linear_model.LogisticRegression(**kwargs)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def fit_clogistic(X, y, sample_weight=None, lb_value=None, fit_intercept=False, **kwargs):
    if X.shape[0] == 0:
        raise ValueError('Attempting to fit on an empty dataset.')
    model = clogistic.LogisticRegression(fit_intercept=fit_intercept, **kwargs)
    bounds = None
    if lb_value is not None:
        lb = np.full(X.shape[1], lb_value)
        if fit_intercept:
            lb = np.r_[lb, -np.inf]
        bounds = scipy.optimize.Bounds(lb=lb)
    model.fit(X, y, sample_weight=sample_weight, bounds=bounds)
    return model


def normalize_norm(coef, ord=1, **kwargs):
    # To make the normalized coefficients comparable across problems,
    # they are scaled by the number of features (`coef.shape[1]`).
    return coef / np.linalg.norm(coef, ord=ord, **kwargs) * coef.shape[1]


def normalize_sum(coef):
    # This normalization ensures that the minimum coefficient is `lb_value`.
    m = np.abs(coef).sum()
    return coef / m * coef.shape[1]


def normalize_min(coef, lb_value=1):
    # This normalization ensures that the minimum coefficient is `lb_value`.
    m = coef.min()
    if m <= 0:
        return None
    return coef / m * lb_value


fits = {
    #'sklearn': (fit_sklearn, normalize_norm),
    #'none': (fit_clogistic, normalize_sum),
    '0+': (functools.partial(fit_clogistic, lb_value=0), normalize_sum),
    '1+': (functools.partial(fit_clogistic, lb_value=1), normalize_min)
}
