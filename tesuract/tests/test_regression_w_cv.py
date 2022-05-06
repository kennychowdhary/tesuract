import tesuract
import unittest
import numpy as np
import warnings, pdb
import time as T

from sklearn.model_selection import RepeatedKFold
import pytest

relpath = tesuract.__file__[:-11]  # ignore the __init__.py specification
print(relpath)


def mse(a, b):
    return mean_squared_error(a, b, squared=False)


import numpy as np
import tesuract
from sklearn.datasets import make_friedman1

# generate test data
rn = np.random.RandomState(44)
dim = 10
X, y = make_friedman1(n_samples=100, n_features=dim, random_state=rn)

# transform data to [-1,1] for polynomials
dScaler = tesuract.preprocessing.DomainScaler(dim=dim, input_range=(0, 1))
Xs = dScaler.fit_transform(X)

# fit using PCE CV wrapper
pce_param_grid = {
    "order": [1],
    "mindex_type": ["total_order"],
    "fit_type": ["LassoCV", "linear"],
}

rf_param_grid = {"n_estimators": [100], "max_features": ["auto"], "max_depth": [5, 10]}

svr_param_grid = {
    "kernel": ["linear", "poly"],
    "degree": [4],
    "gamma": ["auto"],
    "C": [1],
}


@pytest.mark.regression
class TestPCERegression(unittest.TestCase):
    def test_pce_cv_regression(self):
        pceCV = tesuract.RegressionWrapperCV(
            regressor="pce", reg_params=pce_param_grid, n_jobs=-1
        )
        pceCV.fit(Xs, y)
        print("best pce score:", pceCV.best_score_)
        print(
            "best pce faeture importances:", pceCV.best_estimator_.feature_importances_
        )

    def test_pce_repeated_kfold_cv_regression(self):
        rkcv = RepeatedKFold(n_splits=5, n_repeats=2)
        pceCV2 = tesuract.RegressionWrapperCV(
            regressor="pce", reg_params=pce_param_grid, cv=rkcv, n_jobs=-1
        )
        pceCV2.fit(Xs, y)
        print("best pce error: {0:.5f}".format(-pceCV2.best_score_))
        print(
            "best pce faeture importances:", pceCV2.best_estimator_.feature_importances_
        )

    def test_rf_cv_regression(self):
        rfCV = tesuract.RegressionWrapperCV(
            regressor="rf", reg_params=rf_param_grid, n_jobs=-1
        )
        rfCV.fit(Xs, y)
        print("best rf score:", rfCV.best_score_)

    def test_svr_cv_regression(self):
        svrCV = tesuract.RegressionWrapperCV(
            regressor="svr", reg_params=svr_param_grid, n_jobs=-1
        )
        svrCV.fit(Xs, y)
        print("best svr score:", svrCV.best_score_)

    def test_multimodel_cv_regression(self):
        multimodelCV = tesuract.RegressionWrapperCV(
            regressor=["pce", "rf", "svr"],
            reg_params=[pce_param_grid, rf_param_grid, svr_param_grid],
            n_jobs=-1,
        )
        multimodelCV.fit(Xs, y)
        print("best pce/rf score:", multimodelCV.best_score_)
