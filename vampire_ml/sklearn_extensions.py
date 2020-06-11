import logging
import warnings

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_random_state


class StaticTransformer(TransformerMixin):
    def fit(self, x, y=None):
        return self


class FrozenLinearModel(RegressorMixin, LinearModel):
    def __init__(self, coef, intercept=0):
        self.coef = coef
        self.intercept = intercept

    def fit(self, X, y):
        assert self.coef.shape == (X.shape[1],)
        self.coef_ = self.coef
        self.intercept_ = self.intercept
        return self


class MeanRegression(RegressorMixin, LinearModel):
    def fit(self, X, y):
        self.intercept_ = 0
        self.coef_ = np.zeros(X.shape[1], dtype=y.dtype)
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


class InterceptRegression(RegressorMixin, LinearModel):
    def fit(self, X, y):
        self.intercept_ = np.nanmean(y)
        self.coef_ = np.zeros(X.shape[1], dtype=y.dtype)
        return self


class BayesRegression(RegressorMixin):
    def fit(self, X, y):
        self.values_ = {}
        for row, indices in zip(*np.unique(X, return_inverse=True)):
            self.values_[row] = np.mean(y[indices])
        return self

    def predict(self, X):
        return [self.values_[row] for row in X]


class QuantileImputer(SimpleImputer):
    def __init__(self, missing_values=np.nan, copy=True, quantile=0.5, factor=1, divide_by_success_rate=False,
                 default_fill_value=np.nan):
        super().__init__(missing_values=missing_values, strategy='constant', fill_value=np.nan, copy=copy)
        self.quantile = quantile
        self.factor = factor
        self.divide_by_success_rate = divide_by_success_rate
        self.default_fill_value = default_fill_value

    def fit(self, X, y=None):
        self.fill_value = self.default_fill_value
        if not np.isnan(X).all():
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
        self.mean_ = None
        if self.with_mean:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                self.mean_ = np.nan_to_num(np.nanmean(X, axis=0), copy=False, nan=0)
        self.var_ = None
        self.scale_ = None
        if self.with_std:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                self.var_ = np.nan_to_num(np.nanvar(X, axis=0), copy=False, nan=1)
            self.var_[np.isclose(0, self.var_)] = 1
            self.scale_ = np.sqrt(self.var_)
            assert not np.isclose(0, self.scale_).any()
        self.n_samples_seen_ = X.shape[0]
        return self

    def transform(self, X, copy=None):
        assert not np.isclose(0, self.scale_).any()
        res = super().transform(X, copy=copy)
        res[np.isclose(0, res)] = 0
        return res


class StableShuffleSplit(ShuffleSplit):
    dtype = np.uint32

    def _iter_indices(self, X, y=None, groups=None):
        n_available_total = _num_samples(X)
        if groups is None:
            available_train = np.ones(n_available_total, dtype=np.bool)
            available_test = np.ones(n_available_total, dtype=np.bool)
        else:
            assert groups.shape == (n_available_total,)
            if np.count_nonzero(groups == 'train') > 0:
                available_train = groups == 'train'
            else:
                available_train = np.ones(n_available_total, dtype=np.bool)
            if np.count_nonzero(groups == 'test') > 0:
                available_test = groups == 'test'
            else:
                available_test = np.ones(n_available_total, dtype=np.bool)
        n_available_train = np.count_nonzero(available_train)
        n_available_test = np.count_nonzero(available_test)
        n_train = self.train_samples(n_available_train)
        n_test = self.test_samples(n_available_test)
        logging.info('Generating %s splits. Train: %s / %s. Test: %s / %s. Total: %s.', self.n_splits, n_train,
                     n_available_train, n_test, n_available_test, n_available_total)
        if n_train + n_test > n_available_total:
            raise ValueError('Reduce test size or train size.')
        # We omit shuffling if all samples go into one output set with only one split.
        if self.n_splits == 1 and n_available_total == n_train:
            assert n_test == 0
            yield np.arange(n_available_total, dtype=self.dtype), np.empty((0,), dtype=self.dtype)
            return
        if self.n_splits == 1 and n_available_total == n_test:
            assert n_train == 0
            yield np.empty((0,), dtype=self.dtype), np.arange(n_available_total, dtype=self.dtype)
            return
        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            permutation = rng.permutation(n_available_total).astype(self.dtype)
            ind_train = np.fromiter((p for p in permutation if available_train[p]), dtype=self.dtype, count=n_train)
            ind_train_set = set(ind_train)
            ind_test = np.fromiter((p for p in reversed(permutation) if available_test[p] and p not in ind_train_set),
                                   dtype=self.dtype, count=n_test)
            assert len(ind_train) == n_train
            assert len(ind_test) == n_test
            yield ind_train, ind_test

    def train_samples(self, train_candidates):
        return self._subset_size_to_n(self.train_size, train_candidates)

    def test_samples(self, test_candidates):
        return self._subset_size_to_n(self.test_size, test_candidates)

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


def get_feature_weights(estimator):
    weights = None
    try:
        weights = estimator.coef_
    except AttributeError:
        pass
    try:
        weights = estimator.feature_importances_
    except AttributeError:
        pass
    if weights is None:
        raise RuntimeError('The estimator does not expose feature weights in `coef_` or `feature_importances_`.')
    return weights


def get_hyperparameters(estimator):
    res = {}
    try:
        res['alpha'] = estimator.alpha_
    except AttributeError:
        pass
    try:
        res['l1_ratio'] = estimator.l1_ratio_
    except AttributeError:
        pass
    return res


def fit(estimator, x, y, cross_validate):
    assert len(x) == len(y)
    record = {'sample_count': x.shape[0], 'feature_count': x.shape[1]}
    if cross_validate is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=sklearn.exceptions.ConvergenceWarning)
            # Throws ValueError if y contains a nan.
            scores = cross_validate(estimator, x, y)
        for score_name, a in scores.items():
            record['cv', score_name, 'mean'] = np.mean(a)
            record['cv', score_name, 'std'] = np.std(a)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=sklearn.exceptions.ConvergenceWarning)
        # Throws ValueError if y contains a nan.
        estimator.fit(x, y)
        # Note: If fitting fails to converge, the coefficients are all zeros.
        record.update(predictor_properties(estimator, x, y))
    return record


def predictor_properties(predictor, x, y_true):
    assert len(x) == len(y_true)
    data = {}
    y_pred = predictor.predict(x)
    data.update({('score', key): value for key, value in scores(y_true, y_pred).items()})
    try:
        data.update({('coef_', 'count'): len(predictor.coef_),
                     ('coef_', 'nonzero'): np.count_nonzero(predictor.coef_)})
    except AttributeError:
        pass
    for name in ['intercept_', 'n_iter_', 'alpha_', 'l1_ratio_']:
        try:
            data[name] = getattr(predictor, name)
        except AttributeError:
            pass
    return data


def scores(y_true, y_pred):
    return {'mse': mean_squared_error(y_true, y_pred), 'r2': r2_score(y_true, y_pred)}
