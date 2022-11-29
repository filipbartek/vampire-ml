import warnings

import numpy as np
import sklearn


class ConstantLinearClassifier(sklearn.base.BaseEstimator):
    def __init__(self, coef=0, intercept=0):
        self.coef = coef
        self.intercept = intercept
        # We only call `fit` to initialize all the properties set by `fit`, such as `coef_` or `classes_`.
        # For this reason we fit with `max_iter=0`.
        self.model = sklearn.linear_model.LogisticRegression(penalty='none', fit_intercept=False, max_iter=0)

    def __getattr__(self, name):
        # Delegate everything to `self.model`.
        return getattr(self.model, name)

    def fit(self, X, *args, **kwargs):
        with warnings.catch_warnings():
            # Fitting with insufficient iterations emits a warning. We suppress the warning.
            warnings.simplefilter('ignore', category=sklearn.exceptions.ConvergenceWarning)
            self.model.fit(X, *args, **kwargs)
        self.model.coef_ = np.full_like(self.model.coef_, self.coef)
        self.model.intercept_ = np.full_like(self.model.intercept_, self.intercept)
        return self
