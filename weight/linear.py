import logging

import numpy as np
import scipy
import sklearn

from questions.utils import timer

log = logging.getLogger(__name__)


def analyze_pair(model, samples_aggregated, common_clause_features=None, evaluate_weights=None):
    X = samples_aggregated['token_counts'].toarray()
    y = samples_aggregated['proof'].toarray().squeeze(axis=1)

    pair_X, pair_y = construct_pair_dataset(X, y)

    logging.debug(f'Fitting dataset of shape {pair_X.shape}...')
    with timer() as t_fit:
        model.fit(pair_X, pair_y)

    coef = model.coef_.squeeze(axis=0)
    coef_symbol = coef[len(common_clause_features):]

    eval_res = []
    if evaluate_weights is not None:
        eval_res = evaluate_weights(coef)

    def feature_record(value):
        return {
            'value': value,
            'pc': {kind: scipy.stats.percentileofscore(coef_symbol, value, kind=kind) for kind in ['strict', 'weak']}
        }

    return {
        'score': {
            'pair_accuracy': model.score(pair_X, pair_y),
            'single_roc_auc': sklearn.metrics.roc_auc_score(~y, model.predict_proba(X)[:, 1])
        },
        'coef': {
            'feature': {k: feature_record(v) for k, v in zip(common_clause_features, coef)},
            'symbol': {k: getattr(np, k)(coef_symbol) for k in ['mean', 'std', 'min', 'max', 'median']}
        },
        'time_fit': t_fit.elapsed,
        'empirical': eval_res
    }


def construct_pair_dataset(X, y):
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

    return pair_X, pair_y
