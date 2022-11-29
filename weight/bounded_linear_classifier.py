import logging

import numpy as np
import scipy
import sklearn


log = logging.getLogger(__name__)


class BoundedLinearClassifier(sklearn.base.BaseEstimator):
    def __init__(self, model, coef_lb=None, coef_ub=None, keep_feasible=False, normalize_coef=True):
        self.model = model
        self.coef_lb = coef_lb
        self.coef_ub = coef_ub
        self.keep_feasible = keep_feasible
        self.normalize_coef = normalize_coef

    def __getattr__(self, name):
        # Delegate everything to `self.model`.
        return getattr(self.model, name)

    def fit(self, X, y, *args, **kwargs):
        if len(y) == 0:
            raise ValueError('Cannot fit to an empty dataset.')
        # Raises `cvxpy.error.SolverError` if the fitting fails.
        self.model.fit(X, y, *args, bounds=self.bounds(X.shape[1]), **kwargs)
        assert self.valid()
        if self.normalize_coef:
            score_before = self.score(X, y)
            log.debug(f'Before normalization: score={score_before}, residual_min={self.residual_min()}')
            self.normalize()
            score_after = self.score(X, y)
            log.debug(f'After normalization: score={score_after}, residual_min={self.residual_min()}')
            assert score_before == score_after
        assert self.valid()
        return self

    def normalize(self):
        if self.coef_lb > 0:
            log.debug('Normalizing by minimum.')
            factor = self.coef_lb / self.coef_.min()
        elif self.coef_lb is None or self.coef_lb == 0:
            log.debug('Normalizing by sum.')
            factor = self.n_features / np.abs(self.coef_).sum()
        else:
            log.debug(f'Skipping normalization for coef_lb={self.coef_lb}.')
            return
        log.debug(f'Normalization factor: {factor}')
        self.coef_ *= factor
        assert self.fit_intercept or (self.intercept_ == 0).all()
        if self.fit_intercept:
            self.intercept_ *= factor

    def valid(self, tol=1e-05):
        return self.residual_min() >= -tol

    def residual_min(self):
        return min(r.min() for r in self.residual())

    def residual(self):
        return self.bounds(self.n_features).residual(self.coef_intercept)

    @property
    def coef_intercept(self):
        res = self.coef_.squeeze(axis=0)
        if self.fit_intercept:
            res = np.r_[res, self.intercept_]
        return res

    @property
    def n_features(self):
        return self.coef_.shape[1]

    def bounds(self, n_features=None):
        if n_features is None:
            n_features = self.n_features
        return scipy.optimize.Bounds(lb=self.lb(n_features), ub=self.ub(n_features), keep_feasible=self.keep_feasible)

    def lb(self, n_features):
        return self.bound(n_features, -np.inf, coef=self.coef_lb)

    def ub(self, n_features):
        return self.bound(n_features, np.inf, coef=self.coef_ub)

    def bound(self, n_features, default, coef=None, intercept=None):
        if coef is None:
            coef = default
        if intercept is None:
            intercept = default
        res = np.full(n_features, coef)
        if self.fit_intercept:
            res = np.r_[np.full(n_features, coef), intercept]
        return res
