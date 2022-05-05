import numpy as np
import warnings, pdb

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


class scale:
    def __init__(self, X, a=None, b=None):
        self.X = np.atleast_2d(X)
        self.dim = self.X.shape[1]  # each column is a dimension or feature
        if a is None and b is None:
            # default bounds
            self.a = -1 * np.ones(self.dim)
            self.b = 1 * np.ones(self.dim)
            self.__scaled = self.X  # already scaled
        elif len(a) == 0 and len(b) == 0:
            self.a = -1 * np.ones(self.dim)
            self.b = 1 * np.ones(self.dim)
            self.__scaled = self.X  # already scaled
        else:
            self.a = a
            self.b = b
            assert self.dim == len(a), "dim mismatch in a!"
            assert self.dim == len(b), "dim mismatch in b!"

            self.w = self.b - self.a
            assert self.X.shape[1] == len(self.w), "X and a,b mismatch!"
            # scale to -1 -> 1
            self.__scaled = 2 * (self.X - self.a) / self.w - 1.0
        # integration factor ([a,b] -> [-1,1])
        # self.intf = np.prod(.5*(self.b - self.a))
        # self.intf = np.prod(.5*(self.b - self.a)/(self.b - self.a)) # (b-a) canceled by prod prob weight

    def __getitem__(self, key):
        return self.__scaled[key]


# domain test
class DomainScaler:
    def __init__(self, dim, input_range, output_range=(-1, 1)):
        self.dim = dim
        self.input_range = input_range
        self.output_range = output_range

    def _get_bound_list(self, input_range):
        if isinstance(input_range, list) or isinstance(input_range, np.ndarray):
            assert len(input_range) == self.dim, "input range must be a list of tuples"
            input_bounds = input_range
        elif isinstance(input_range, tuple):
            input_bounds = [(input_range[0], input_range[1]) for i in range(self.dim)]
        a = np.array([ab[0] for ab in input_bounds])  # lower bounds
        b = np.array([ab[1] for ab in input_bounds])  # upper bounds
        return input_bounds, a, b

    def _range_check(self, X, B):
        if X.ndim == 1:
            X = np.atleast_2d(X)
        n, d = X.shape
        assert d == self.dim, "columns of X must be the same as dimensions"
        assert len(B) == self.dim, "length of bounds list must be same as dimensions"
        dim_check = [
            (X[:, i] >= B[i][0]).all() and (X[:, i] <= B[i][1]).all() for i in range(d)
        ]
        assert all(dim_check), "X is not in the range of the input range."
        return X

    def fit_transform(self, X):
        self.input_bounds, a, b = self._get_bound_list(self.input_range)
        self.output_bounds, c, d = self._get_bound_list(self.output_range)
        X = self._range_check(X, self.input_bounds)
        # transform to [0,1] first for ease
        X_unit_scaled = (X - a) / (b - a)
        # transform to output bounds
        X_scaled = (d - c) * X_unit_scaled + c
        X_scaled = self._range_check(X_scaled, self.output_bounds)
        return X_scaled

    def inverse_transform(self, Xhat):
        self.input_bounds, a, b = self._get_bound_list(self.input_range)
        self.output_bounds, c, d = self._get_bound_list(self.output_range)
        Xhat = self._range_check(Xhat, self.output_bounds)
        Xhat_unit_scaled = (Xhat - c) / (d - c)
        X_inv = (b - a) * Xhat_unit_scaled + a
        X_inv = self._range_check(X_inv, self.input_bounds)
        return X_inv


class MinMaxTargetScaler:
    """Custom target scaler which uses one min and max for all elements

                    Description:

    A Transformer class that scales and shifts by the min/max

    This transformer transforms the target Y by the absolute max and minimum to
    the unit hypercube, i.e., [0,1]^dim.  It does not scale each target column
    differently as in sklearn's preprocessing toolbox. The transform is simply

    Y = (max(Y)-min(Y))*Yhat + min(Y)

    Todo:
                    To do list

    Note:
                    This is just a test

    Parameters:
                    min_ (float):   minimum of Y (over all columns and rows)
                    max_ (float):   maximum of Y (over all columns and rows)
                    w_ (float):     diff btwn min and max of Y (over all columns and rows)
                    ab_w_ (float):      width of target range

    Attributes:
                    min_ (float):   minimum of Y (over all columns and rows)
                    max_ (float):   maximum of Y (over all columns and rows)
                    w_ (float):     diff btwn min and max of Y (over all columns and rows)
                    ab_w_ (float):      width of target range

    Returns:
                    The return value.

    Raises:
                    Attribute Errors

    Examples:
                    Examples should be written in doctest format

                    >>> print("Hello World!")
    """

    def __init__(self, target_range=(0, 1)):
        """This is a description of the constructor"""
        self.a, self.b = target_range

    def fit(self, Y, target=None):
        """
        Fit method

        This function compute the min, max and width, which are used in the above
        transform and inverse transform formulas.

        Parameters
        ----------
        Y : numpy.ndarray, required
                        ndarray of shape (n_samples ,n_features)
                        This is the (possibly multi) target output.

        """
        self.min_ = np.amin(Y)
        self.max_ = np.amax(Y)
        self.w_ = self.max_ - self.min_
        self.ab_w_ = self.b - self.a

    def transform(self, Y):
        assert self.min_ is not None and self.max_ is not None, "Need to fit first."
        # scale to 0 -> 1 first
        Yhat = (Y - self.min_) / self.w_
        # scale to target_range
        Yhat = Yhat * self.ab_w_ + self.a
        return Yhat

    def fit_transform(self, Y, target=None):
        self.fit(Y)
        return self.transform(Y)

    def inverse_transform(self, Yhat):
        # first transform back to [0,1]
        # Y = (Yhat - self.a)/self.ab_w_
        # back to true range
        Y = (self.w_ / self.ab_w_) * (Yhat - self.a) + self.min_
        return Y


class PCATargetTransform(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_components=2, exp_var_cutoff=0.90, svd_solver="arpack", whiten=False
    ):
        self.n_components = n_components
        self.exp_var_cutoff = exp_var_cutoff
        self.svd_solver = svd_solver
        self.K = None
        self.whiten = whiten  # return scaled projections w var=1

    def fit(self, Y, target=None):
        self.n, self.d = Y.shape
        if self.n_components == "auto":
            self._compute_K(Y)
        if isinstance(self.n_components, int):
            self._compute_K(Y)  # compute K regardless to get cumulative error
            self.K = self.n_components
            if self.n_components > self.d:
                warnings.warn(
                    "No of components greater than dimension. Setting to maximum allowable."
                )
                self.n_components = self.d
        assert self.K is not None, "K components is not defined or being set properly."

        self.pca = PCA(
            n_components=self.K, svd_solver=self.svd_solver, whiten=self.whiten
        )
        self.pca.fit(Y)
        self.components_ = self.pca.components_
        self.n_components_ = self.pca.n_components_
        self.singular_values_ = self.pca.singular_values_
        self.variances_ = self.singular_values_ ** 2 / self.n
        self.cumulative_error = self.cumulative_error_full[: self.K]
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return self

    def transform(self, Y):
        assert hasattr(self, "pca"), "Perform fit first."
        return self.pca.transform(Y)

    def inverse_transform(self, Yhat):
        return self.pca.inverse_transform(Yhat)

    def _compute_K(self, Y, max_n_components=50):
        """automatically compute number of PCA terms"""
        max_n_components = min(50, min(Y.shape[0], Y.shape[1]))
        skpca = PCA(n_components=min(max_n_components, self.d), svd_solver="auto")
        skpca.fit(Y)
        self.cumulative_error_full = np.cumsum(skpca.explained_variance_ratio_)
        # print(cumulative_error)
        # need to check whether to use + 1 or not
        loc = np.where(1 - self.cumulative_error_full <= 1 - self.exp_var_cutoff)[0] + 1
        if loc.size == 0:
            if self.n_components == "auto":
                warnings.warn(
                    "Exp var cutoff may be too strict. Setting K to at most 50 components"
                )
            self.K = max_n_components
        elif loc.size > 0:
            self.K = loc[0]


class LogTransform:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            assert np.all(X > 0), "X has negative values."
        return self

    def transform(self, X):
        self.fit(X)
        return np.log(X)

    def inverse_transform(self, Xhat):
        return np.exp(Xhat)


def gen_scale_pipeline(log=True, scale=True, pca=True, **kwargs):
    target_range = kwargs.get("target_range", (0, 1))
    n_components = kwargs.get("n_components", 4)
    svd_solver = kwargs.get("svd_solver", "arpack")
    estimators = []
    if log:
        estimators.append(("log", LogTransform()))
    if scale:
        estimators.append(("scaler", MinMaxTargetScaler(target_range=target_range)))
    if pca:
        estimators.append(
            (
                "pca",
                PCATargetTransform(n_components=n_components, svd_solver=svd_solver),
            )
        )
    pipe = Pipeline(estimators)
    return pipe


def target_pipeline(log=False, scale=False, pca=True, **kwargs):
    target_range = kwargs.get("target_range", (0, 1))
    n_components = kwargs.get("n_components", 4)
    svd_solver = kwargs.get("svd_solver", "arpack")
    whiten = kwargs.get("whiten", False)
    estimators = []
    if log:
        estimators.append(("log", LogTransform()))
    if scale:
        estimators.append(("scaler", MinMaxTargetScaler(target_range=target_range)))
    if pca:
        estimators.append(
            (
                "pca",
                PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten),
            )
        )
    pipe = Pipeline(estimators)
    return pipe


def target_pipeline_custom(log=False, scale=False, pca=True, **kwargs):
    target_range = kwargs.get("target_range", (0, 1))
    n_components = kwargs.get("n_components", 4)
    svd_solver = kwargs.get("svd_solver", "arpack")
    whiten = kwargs.get("whiten", False)
    exp_var_cutoff = kwargs.get("exp_var_cutoff", 0.9)
    estimators = []
    if log:
        estimators.append(("log", LogTransform()))
    if scale:
        estimators.append(("scaler", MinMaxTargetScaler(target_range=target_range)))
    if pca:
        estimators.append(
            (
                "pca",
                PCATargetTransform(
                    n_components=n_components,
                    svd_solver=svd_solver,
                    whiten=whiten,
                    exp_var_cutoff=exp_var_cutoff,
                ),
            )
        )
    pipe = Pipeline(estimators)
    return pipe
