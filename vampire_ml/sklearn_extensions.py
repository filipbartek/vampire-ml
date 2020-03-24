import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model._base import LinearModel
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_random_state


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
        self.success_rate = None

    def fit(self, X, y=None):
        self.fill_value = np.nanquantile(X, self.quantile) * self.factor
        if self.divide_by_success_rate:
            self.success_rate = np.count_nonzero(~np.isnan(X)) / X.size
            assert 0 <= self.success_rate <= 1
            if self.success_rate != 0:
                self.fill_value /= self.success_rate
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


class StableShuffleSplit(ShuffleSplit):
    def _iter_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(
            n_samples, self.test_size, self.train_size,
            default_test_size=self._default_test_size)

        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            # random partition
            permutation = rng.permutation(n_samples)
            assert n_train + n_test <= len(permutation)
            ind_train = permutation[:n_train]
            ind_test = permutation[:-n_test - 1:-1]
            assert len(ind_test) == n_test
            yield ind_train, ind_test


class EstimatorDict(BaseEstimator):
    def __init__(self, **params):
        self.__dict__.update(params)

    def get_params(self, deep=True):
        res = self.__dict__.copy()
        if deep:
            for key, value in self.__dict__.items():
                if value is None:
                    continue
                res.update({f'{key}__{k}': v for k, v in value.get_params().items()})
        return res

    def __getitem__(self, item):
        return self.__dict__[item]

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()
