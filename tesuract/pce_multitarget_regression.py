import numpy as np
import tesuract
from collections import defaultdict
import time as T
import matplotlib.pyplot as mpl
import warnings, pdb

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    KFold,
    cross_val_score,
    cross_validate,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# from alive_progress import alive_bar
from tqdm import tqdm


class RegressionWrapperCV(BaseEstimator):
    def __init__(
        self,
        regressor="pce",
        reg_params={},
        scorer="neg_root_mean_squared_error",
        n_jobs=1,
        verbose=0,
        cv=None,
    ):
        self.regressor = regressor
        self.reg_params = reg_params
        self.scorer = scorer
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cv = cv
        # self.__dict__.update(reg_params)

    def _setup_cv(self):
        if self.cv == None:
            self.cv = KFold(n_splits=5)

    def _reformat_grid(self, params):
        if isinstance(self.regressor, str):
            self.regressor = [self.regressor]
        try:
            # if parameter grid is in the form of grid search cv
            ParameterGrid(params)
            params_cv = params
        except:
            # convert reg_params to form that can be used by grid search
            # for a single grid point
            params_cv = defaultdict(list)
            for key, item in params.items():
                params_cv[key].append(item)
        return params_cv

    def _model_factory(self, regressor):
        if regressor == "pce":
            return tesuract.PCEReg()
        if regressor == "rf":
            return RandomForestRegressor()
        if regressor == "mlp":
            return MLPRegressor()
        if regressor == "gpr":
            return GaussianProcessRegressor()
        if regressor == "krr":
            return KernelRidge()
        if regressor == "knn":
            return KNeighborsRegressor()
        if regressor == "svr":
            return SVR()

    def fit(self, X, y):
        self._setup_cv()
        if isinstance(self.regressor, str):
            model = self._model_factory(regressor=self.regressor)
            reg_params_cv = self._reformat_grid(self.reg_params)
            GridSCV = GridSearchCV(
                model,
                reg_params_cv,
                scoring=self.scorer,
                n_jobs=self.n_jobs,
                cv=self.cv,
                verbose=self.verbose,
                return_train_score=True,
            )
            GridSCV.fit(X, y)
            self.best_estimator_ = GridSCV.best_estimator_
            self.best_params_ = GridSCV.best_params_
            self.best_score_ = GridSCV.best_score_
            self.best_overfit_error_ = self.overfit_score(GridSCV)
            self.cv_results_ = GridSCV.cv_results_
        if isinstance(self.regressor, list):
            self.fit_multiple_reg(X, y)
        return self

    def fit_multiple_reg(self, X, y):
        if isinstance(self.regressor, list):
            n_regressors = len(self.regressor)
            self.best_estimators_ = [None] * n_regressors
            self.all_best_params_ = [None] * n_regressors
            self.best_scores_ = np.zeros(n_regressors)
            self.best_overfit_errors_ = [None] * n_regressors
            self.all_cv_results_ = [None] * n_regressors
            # potentially depricated
            # assert isinstance(self.reg_params,list), "parameters must also be a list"
            if isinstance(self.reg_params, list) is False:
                self.reg_params = [self.reg_params]
            # print(len(self.reg_params), n_regressors)
            assert (
                len(self.reg_params) == n_regressors
            ), "length of parameters and regressors must match."
            for i, R in enumerate(self.regressor):
                # print("Fitting regressor ", R)
                model = self._model_factory(regressor=R)
                reg_params_cv = self._reformat_grid(self.reg_params[i])
                GridSCV = GridSearchCV(
                    model,
                    reg_params_cv,
                    scoring=self.scorer,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    verbose=self.verbose,
                    return_train_score=True,
                )
                GridSCV.fit(X, y)
                self.best_estimators_[i] = GridSCV.best_estimator_
                self.all_best_params_[i] = GridSCV.best_params_
                self.best_scores_[i] = GridSCV.best_score_
                self.best_overfit_errors_[i] = self.overfit_score(GridSCV)
                self.all_cv_results_[i] = GridSCV.cv_results_
            self.best_index_ = np.argmax(self.best_scores_)
            self.best_estimator_ = self.best_estimators_[self.best_index_]
            self.best_params_ = self.all_best_params_[self.best_index_]
            self.best_score_ = self.best_scores_[self.best_index_]
            self.best_overfit_error_ = self.best_overfit_errors_[self.best_index_]
            self.cv_results_ = self.all_cv_results_[self.best_index_]
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def overfit_score(self, GridSCV):
        """Takes a grid search cv object and returns overfit score"""
        # assert hasattr(self,'GridSCV'), "Must run fit function."
        best_index_ = GridSCV.best_index_
        mean_train_score = GridSCV.cv_results_["mean_train_score"]
        mean_test_score = GridSCV.cv_results_["mean_test_score"]
        return mean_train_score[best_index_] - mean_test_score[best_index_]

class MRegressionWrapperCV(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        regressor="pce",
        reg_params={"order": 2},
        custom_params=False,
        target_transform=None,
        target_transform_params={},
        scorer="neg_root_mean_squared_error",
        n_jobs=-1,
        cv=None,
        mixed=True,
        verbose=0,
    ):
        self.regressor = regressor
        self.reg_params = reg_params
        self.custom_params = custom_params
        self.scorer = scorer
        self.n_jobs = n_jobs
        self.cv = cv
        self.target_transform = target_transform
        self.target_transform_params = target_transform_params
        self.mixed = mixed  # to control whether mixed surrogate for each target TBD
        self.verbose = verbose

    def _setupCV(self, shuffle=False, randstate=13):
        if self.cv == None:
            self.cv = KFold(n_splits=5)  # ,shuffle=True,random_state=13)

    def _fit_target_transform(self, Y):
        if self.target_transform is None:
            self.TT = FunctionTransformer(lambda Y: Y)
        else:
            if isinstance(self.target_transform, Pipeline):
                # print("Target Transform is a pipeline object. Cannot set internal parameters just yet.")
                self.TT = self.target_transform
            # elif isinstance(self.target_transform,PCA):
            # 	# if instantiated PCA object, ok to set to target transform
            # 	self.TT = self.target_transform
            else:
                if isinstance(self.target_transform, type):
                    # Allow user to pass uninstantiated class
                    # for uninstantiated input to target transform
                    self.TT = self.target_transform(**self.target_transform_params)
                else:
                    # Allow user to pass an instantiated class
                    self.TT = self.target_transform
                assert hasattr(
                    self.target_transform, "fit"
                ), "transform object is instantiated and must have a fit method."
                assert hasattr(
                    self.target_transform, "transform"
                ), "transform object is instantiated and must have a transform method."
                assert hasattr(
                    self.target_transform, "inverse_transform"
                ), "transform object is instantiated and must have an inverse_transform method."

        self.TT.fit(Y)
        return self

    def fit(self, X, Y):
        self.n, self.dim = X.shape
        self._setupCV()
        self._fit_target_transform(Y)
        Yhat = self.TT.transform(Y)
        # save explained variance ratio if using PCA
        # otherwise, return equal weights
        try:
            self.explained_variance_ratio_ = (
                self.TT.named_steps.pca.explained_variance_ratio_
            )
            self.cumsum_explained_variance_ratio_ = (
                self.TT.named_steps.pca.cumulative_error_full
            )
        except:
            self.explained_variance_ratio_ = np.ones(Yhat.shape[1]) / Yhat.shape[1]
        # compute total variance for feature importances
        self.TotVar = np.var(
            Y
        )  # .var(axis=0).mean()# total variance averaged over all space
        if Yhat.ndim == 1:
            Yhat = np.atleast_2d(Yhat).T
        assert len(Yhat) == self.n, "mistmatch in no of samples btwn X and Y."
        self.ntargets = Yhat.shape[1]
        self._setupCV()
        if isinstance(self.regressor, str):
            self.res = self.fit_single_reg(X, Yhat)
        elif isinstance(self.regressor, list) and self.mixed == False:
            self.res = self.fit_multiple_reg(X, Yhat)
        elif isinstance(self.regressor, list) and self.mixed == True:
            self.res = self.fit_single_reg(X, Yhat)
        return self

    def fit_single_reg(self, X, Y, regressor=None, reg_params=None):
        if regressor == None:
            regressor = self.regressor
            if isinstance(regressor, str):
                regressor = [regressor]
            reg_params = self.reg_params
        # elif len(reg_params) == 1:
        # 	reg_params = [[reg_params] for i in range(self.ntargets)]
        # if isinstance(regressor,list):
        # 	# if regressor is a list and mixed is True, repeat regressor params
        # 	# this is to keep consistent the case when we custom input regressors for each target
        # 	reg_params = [reg_params for i in range(len(regressor))]
        # 	print(reg_params, regressor)
        res = defaultdict(list)
        # with alive_bar(self.ntargets) as bar:
        for i in tqdm(range(self.ntargets)):
            if self.custom_params:
                # fit a single regressor to each target
                assert len(regressor) == len(reg_params)
                assert len(regressor) == self.ntargets
                reg = RegressionWrapperCV(
                    regressor=[regressor[i]],
                    reg_params=[reg_params[i]],
                    n_jobs=self.n_jobs,
                    scorer=self.scorer,
                    cv=self.cv,
                    verbose=self.verbose,
                )
                reg.fit(X, Y[:, i])
                res["best_estimators_"].append(reg.best_estimator_)
                res["best_params_"].append(reg.best_params_)
                res["best_scores_"].append(reg.best_score_)
                dict_scores_ = {regressor[i]: reg.best_scores_}
                res["best_scores_all_"].append(dict_scores_)
                res["best_overfit_error_"].append(reg.best_overfit_error_)
                res["cv_results_"].append(
                    reg.cv_results_
                )  # cv results from best regressor in list
                res["best_index_"].append(reg.best_index_)  # index of best score
                # res['all_cv_results_'].append(reg.all_cv_results_) # all cv results from all regressor in list
            else:
                reg = RegressionWrapperCV(
                    regressor=regressor,
                    reg_params=reg_params,
                    n_jobs=self.n_jobs,
                    scorer=self.scorer,
                    cv=self.cv,
                    verbose=self.verbose,
                )
                reg.fit(X, Y[:, i])
                res["best_estimators_"].append(reg.best_estimator_)
                res["best_params_"].append(reg.best_params_)
                res["best_scores_"].append(reg.best_score_)
                dict_scores_ = {
                    regressor[j]: reg.best_scores_[j] for j in range(len(regressor))
                }
                res["best_scores_all_"].append(dict_scores_)
                res["best_overfit_error_"].append(reg.best_overfit_error_)
                res["cv_results_"].append(
                    reg.cv_results_
                )  # cv results from best regressor in list
                res["best_index_"].append(reg.best_index_)  # index of best score
                # res['all_cv_results_'].append(reg.all_cv_results_) # all cv results from all regressor in list
            # bar()
        self.__dict__.update(res)
        return res

    def fit_multiple_reg(self, X, Y):
        # wont execute if mixed is True
        if isinstance(self.regressor, list):
            mres = []
            for i, r in enumerate(self.regressor):
                res = self.fit_single_reg(
                    X, Y, regressor=r, reg_params=self.reg_params[i]
                )
                mres.append(res)
        return mres

    def predict(self, X):
        assert hasattr(self, "res"), "Must run fit."
        if isinstance(self.res, dict):
            Ypred = self._predict_single(X, res=self.res)
        elif isinstance(self.res, list):
            Ypred = self._predict_multiple(X, res=self.res)
        return Ypred

    def _predict_single(self, X, res=None):
        # assert isinstance(self.res,dict), "Must pass string as regressor OR list with mixed=True, otherwise, predict is ambiguous"
        assert isinstance(
            res, dict
        ), "for single prediction, results must be a dictionary."
        Yhatpred_list = []
        for estimator in self.best_estimators_:
            if X.ndim == 1:
                X = np.atleast_2d(X)
            Yhatpred_list.append(estimator.predict(X))
        Yhatpred = np.array(Yhatpred_list).T
        Ypred = self.TT.inverse_transform(Yhatpred)
        return Ypred

    def _predict_multiple(self, X, res=None):
        # assert isinstance(self.res,dict), "Must pass string as regressor OR list with mixed=True, otherwise, predict is ambiguous"
        assert isinstance(
            res, list
        ), "for multiple predictions, results must be a list of dictionaries."
        predictions = []
        for r in res:
            predictions.append(self._predict_single(X, r))
        return np.squeeze(np.array(predictions))

    @property
    def feature_importances_(self):
        assert hasattr(self, "res"), "Must run .fit() first!"
        FI_ = []
        self.SI_ = []
        for estimator in self.best_estimators_:
            fi = estimator.feature_importances_
            FI_.append(fi)
            self.SI_.append(estimator.computeSobol())
        FI_ = np.array(FI_)
        self.SI_ = np.array(self.SI_)

        # weighted average of sobol weighted by error AND explained variance
        # formula for each dim = sum(sobol weight fraction of each component estimator * explain variance ratio for each component)
        try:
            var_weight = self.TT.named_steps.pca.explained_variance_ratio_
        except:
            var_weight = np.ones(self.ntargets) / self.ntargets
        var_weight = var_weight[:, np.newaxis]
        self.sobol_weighted_ = np.sum(self.SI_ * var_weight, axis=0)
        # ***array doesn't have to sum to 1 since the total variance is not captured by the finite set of PCA compoents**. Moreover, it's sum is doubly counted since the sobol indices can sum to more than 1. Thus, the total variance is potentiall greater than the 1-cutoff and less than 1.
        # self.sobol_weighted /= np.sum(self.sobol_weighted)
        return FI_  # return feature importance for each component pce

    def score(self, X, Y):
        Ypred = self.predict(X)
        assert Ypred.shape == Y.shape, "predict and Y shape do not match."
        # MSPREs = [np.mean((1.0 - Ypred[t]/Y[t])**2) for t in range(Y.shape[0])]
        MSPREs = [
            np.mean((Y[t] - Ypred[t]) ** 2) / np.mean(Y[t] ** 2)
            for t in range(Y.shape[0])
        ]
        return -1 * np.mean(MSPREs)

    # add def for fitting multiple for each component for faster fitting

class MPCEReg(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        regressor="pce",
        reg_params={"order": 2},
        target_transform=None,
        target_transform_params={},
        scorer="neg_root_mean_squared_error",
        n_jobs=4,
        cv=None,
        mixed=True,
        verbose=1,
    ):
        self.regressor = regressor
        self.reg_params = reg_params
        self.scorer = scorer
        self.n_jobs = n_jobs
        self.cv = cv
        self.target_transform = target_transform
        self.target_transform_params = target_transform_params
        self.mixed = mixed  # to control whether mixed surrogate for each target TBD
        self.verbose = verbose

    def _setupCV(self, shuffle=False, randstate=13):
        if self.cv == None:
            self.cv = KFold(n_splits=5)  # ,shuffle=True,random_state=13)

    def _fit_target_transform(self, Y):
        if self.target_transform is None:
            self.TT = FunctionTransformer(lambda Y: Y)
        else:
            if isinstance(self.target_transform, Pipeline):
                # print("Target Transform is a pipeline object. Cannot set internal parameters just yet.")
                self.TT = self.target_transform
            else:
                self.TT = self.target_transform(**self.target_transform_params)
        self.TT.fit(Y)
        return self

    def fit(self, X, Y):
        self.n, self.dim = X.shape
        self._setupCV()
        self._fit_target_transform(Y)
        Yhat = self.TT.transform(Y)
        if Yhat.ndim == 1:
            Yhat = np.atleast_2d(Yhat).T
        assert len(Yhat) == self.n, "mistmatch in no of samples btwn X and Y."
        self.ntargets = Yhat.shape[1]
        self._setupCV()
        if isinstance(self.regressor, str):
            # fits the same regressor to each latent var
            self.res = self.fit_single_reg(X, Yhat)
        elif isinstance(self.regressor, list) and self.mixed == True:
            # fits different regressor to each latent var
            assert (
                len(self.regressor) == self.ntargets
            ), "number of regressors must be same as the targets."
            assert (
                len(self.reg_params) == self.ntargets
            ), "number of regressors must be same as the targets."
            self.res = self.fit_single_reg(X, Yhat)
        # elif isinstance(self.regressor,list) and self.mixed == True:
        # 	self.res = self.fit_single_reg(X,Yhat)
        return self

    def fit_single_reg(self, X, Y, regressor=None, reg_params=None):
        if regressor == None:
            regressor = self.regressor
            reg_params = self.reg_params
        if isinstance(regressor, str):
            # if single string is provided, repeat it for each target
            regressor = [regressor for i in range(self.ntargets)]
            reg_params = [reg_params for i in range(self.ntargets)]
        res = defaultdict(list)
        # with alive_bar(self.ntargets) as bar:
        for i in tqdm(range(self.ntargets)):
            reg = RegressionWrapperCV(
                regressor=[regressor[i]],
                reg_params=[reg_params[i]],
                n_jobs=self.n_jobs,
                scorer=self.scorer,
                cv=self.cv,
                verbose=self.verbose,
            )
            reg.fit(X, Y[:, i])
            res["best_estimators_"].append(reg.best_estimator_)
            res["best_params_"].append(reg.best_params_)
            res["best_scores_"].append(reg.best_score_)
            res["best_overfit_error_"].append(reg.best_overfit_error_)
            res["cv_results_"].append(
                reg.cv_results_
            )  # cv results from best regressor in list
            res["best_index_"].append(reg.best_index_)  # index of best score
            # res['all_cv_results_'].append(reg.all_cv_results_) # all cv results from all regressor in list
            # bar()
        self.__dict__.update(res)
        return res

    def fit_multiple_reg(self, X, Y):
        # wont execute if mixed is True
        if isinstance(self.regressor, list):
            mres = []
            for i, r in enumerate(self.regressor):
                res = self.fit_single_reg(
                    X, Y, regressor=r, reg_params=self.reg_params[i]
                )
                mres.append(res)
        return mres

    def predict(self, X):
        assert hasattr(self, "res"), "Must run fit."
        if isinstance(self.res, dict):
            Ypred = self._predict_single(X, res=self.res)
        elif isinstance(self.res, list):
            Ypred = self._predict_multiple(X, res=self.res)
        return Ypred

    def _predict_single(self, X, res=None):
        # assert isinstance(self.res,dict), "Must pass string as regressor OR list with mixed=True, otherwise, predict is ambiguous"
        assert isinstance(
            res, dict
        ), "for single prediction, results must be a dictionary."
        Yhatpred_list = []
        for estimator in self.best_estimators_:
            if X.ndim == 1:
                X = np.atleast_2d(X)
            Yhatpred_list.append(estimator.predict(X))
        Yhatpred = np.array(Yhatpred_list).T
        Ypred = self.TT.inverse_transform(Yhatpred)
        return Ypred

    def _predict_multiple(self, X, res=None):
        # assert isinstance(self.res,dict), "Must pass string as regressor OR list with mixed=True, otherwise, predict is ambiguous"
        assert isinstance(
            res, list
        ), "for multiple predictions, results must be a list of dictionaries."
        predictions = []
        for r in res:
            predictions.append(self._predict_single(X, r))
        return np.squeeze(np.array(predictions))

    def feature_importances_(self):
        assert hasattr(self, "res"), "Must run .fit() first!"
        FI_ = []
        for estimator in self.best_estimators_:
            fi = estimator.feature_importances_
            FI_.append(fi)
        return np.array(FI_)

    def score(self, X, Y):
        Ypred = self.predict(X)
        assert Ypred.shape == Y.shape, "predict and Y shape do not match."
        MSPREs = [np.mean((1.0 - Ypred[t] / Y[t]) ** 2) for t in range(Y.shape[0])]
        return np.mean(MSPREs)

    # add def for fitting multiple for each component for faster fitting
