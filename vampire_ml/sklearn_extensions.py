import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model._base import LinearModel


class MeanRegression(RegressorMixin, LinearModel):
    def fit(self, X, y):
        assert X.dtype == np.bool
        self.coef_ = np.zeros(X.shape[1], dtype=np.float)
        for i in range(X.shape[1]):
            y_selected = y[X[:, i]]
            if len(y_selected) == 0:
                continue
            self.coef_[i] = y_selected.mean()
        mean_positive_feature_count_per_sample = np.mean(np.sum(X, axis=1))
        assert mean_positive_feature_count_per_sample >= 0
        if mean_positive_feature_count_per_sample != 0:
            self.coef_ /= mean_positive_feature_count_per_sample
        self.intercept_ = 0.
        return self


class QuantileImputer(SimpleImputer):
    def __init__(self, missing_values=np.nan, copy=True, quantile=0.5, factor=1, divide_by_success_rate=False):
        super().__init__(missing_values=missing_values, strategy='constant', copy=copy)
        self.quantile = quantile
        self.factor = factor
        self.divide_by_success_rate = divide_by_success_rate

    def fit(self, X, y=None):
        self.fill_value = np.nanquantile(X, self.quantile) * self.factor
        if self.divide_by_success_rate:
            self.fill_value *= X.size / np.count_nonzero(~np.isnan(X))
        return super().fit(X, y)


class Flattener(TransformerMixin, BaseEstimator):
    def __init__(self, order='F'):
        self.order = order

    def fit(self, X, y=None):
        assert len(X.shape) >= 2
        self.shape_ = X.shape[1:]
        return self

    def transform(self, X):
        assert X.shape[1:] == self.shape_
        return X.reshape((X.shape[0], -1), order=self.order)

    def inverse_transform(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        assert len(X.shape) == 2
        assert X.shape[1] == np.prod(self.shape_)
        return X.reshape((X.shape[0],) + self.shape_, order=self.order)
