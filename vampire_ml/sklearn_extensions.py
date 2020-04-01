import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model._base import LinearModel
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_random_state


class MeanRegression(RegressorMixin, LinearModel):
    def fit(self, X, y):
        self.intercept_ = 0.
        self.coef_ = np.zeros(X.shape[1], dtype=np.float)
        for i in range(X.shape[1]):
            if np.count_nonzero(X[:, i]) == 0:
                continue
            self.coef_[i] = np.average(y, weights=X[:, i])
        # TODO: Generalize for non-bool features.
        mean_positive_feature_count_per_sample = np.mean(np.sum(X, axis=1))
        assert mean_positive_feature_count_per_sample >= 0
        if mean_positive_feature_count_per_sample != 0:
            self.coef_ /= mean_positive_feature_count_per_sample
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
            self.success_rate_ = np.count_nonzero(~np.isnan(X)) / X.size
            assert 0 <= self.success_rate_ <= 1
            if self.success_rate_ != 0:
                self.fill_value /= self.success_rate_
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


class StableStandardScaler(StandardScaler):
    """Handles constant features in a robust manner."""

    def fit(self, X, y=None):
        res = super().fit(X, y=y)
        self.scale_[np.isclose(0, self.scale_)] = 1
        assert not self.with_mean or self.mean_ == np.nanmean(X)
        assert not self.with_std or self.scale_ == np.nanstd(X) or self.scale_ == 1
        assert not np.isclose(0, self.scale_)
        return res

    def transform(self, X, copy=None):
        assert not np.isclose(0, self.scale_)
        res = super().transform(X, copy=copy)
        res[np.isclose(0, res)] = 0
        return res


class StableShuffleSplit(ShuffleSplit):
    def _iter_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        if groups is None:
            groups = np.ones(n_samples, dtype=np.bool)
        assert groups.dtype == np.bool
        assert groups.shape == (n_samples,)
        n_samples_train = np.count_nonzero(groups)
        n_train = self._subset_size_to_n(self.train_size, n_samples_train)
        n_test = self._subset_size_to_n(self.test_size, n_samples)
        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            permutation = rng.permutation(n_samples)
            assert n_train + n_test <= len(permutation)
            permutation_train = [p for p in permutation if groups[p]]
            ind_train = permutation_train[:n_train]
            ind_train_set = set(ind_train)
            permutation_test = [p for p in permutation if p not in ind_train_set]
            ind_test = permutation_test[:-n_test - 1:-1]
            assert len(ind_test) == n_test
            yield ind_train, ind_test

    def _subset_size_to_n(self, subset_size, total_samples):
        if subset_size is None:
            subset_size = 0.5
        if isinstance(subset_size, float):
            assert 0 <= subset_size <= 1
            return int(subset_size * total_samples)
        if isinstance(subset_size, int):
            return subset_size
        assert False


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
