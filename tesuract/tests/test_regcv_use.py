import tesuract
import unittest
import numpy as np
import warnings, pdb
from time import time
import pytest

import sklearn
from sklearn.datasets import make_friedman1
from sklearn.model_selection import cross_val_score

relpath = tesuract.__file__[:-11]  # ignore the __init__.py specification
print(relpath)

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

# regression test for multi output pca regressor
@pytest.mark.regression
class TestRegressionWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # import a data set for our regression problem
        self.X, self.y = make_friedman1(n_samples=100, n_features=6, random_state=1239)

    def test_simple_pce_scalar_fit(self):
        print("Fitting 8th order polynomial...")
        pce_model = tesuract.PCEReg(order=6)
        pce_model.fit(self.X, self.y)
        print("done fitting!")

    def test_cross_val_score_of_pce_reg_class(self):
        # compute the cv score
        pce = tesuract.PCEReg(order=7)
        pce_score = cross_val_score(
            pce, self.X, self.y, scoring="r2", verbose=1, n_jobs=1
        ).mean()
        self.pce_score_1 = pce_score
        print("PCE score is {0:.3f}".format(pce_score))

    def test_mregcv_interface_w_single_parameter_choice(self):
        start = time()
        pce_grid = {
            "order": 6,
            "mindex_type": "total_order",
            "fit_type": "ElasticNetCV",
        }
        # hyper-parameter tune the PCE regression class using all available cores
        pce = tesuract.RegressionWrapperCV(
            regressor="pce", reg_params=pce_grid, n_jobs=1, scorer="r2", verbose=1
        )
        pce.fit(self.X, self.y)
        print("Hyper-parameter CV PCE score is {0:.3f}".format(pce.best_score_))
        print("Total time is ", time() - start)

    def test_mregcv_w_simple_parameter_grid(self):
        start = time()
        pce_grid = {
            "order": list(range(1, 8)),
            "mindex_type": ["total_order"],
            "fit_type": ["linear", "LassoCV"],
        }
        # hyper-parameter tune the PCE regression class using all available cores
        pce = tesuract.RegressionWrapperCV(
            regressor="pce", reg_params=pce_grid, n_jobs=4, scorer="r2", verbose=1
        )
        pce.fit(self.X, self.y)
        print("Hyper-parameter CV PCE score is {0:.3f}".format(pce.best_score_))
        print("Total time is ", time() - start)

    def test_mregcv_w_advanced_param_grid(self):
        start = time()
        pce_grid = [
            {
                "order": list(range(1, 8)),
                "mindex_type": ["total_order"],
                "fit_type": ["linear", "ElasticNetCV"],
                "fit_params": [
                    {"alphas": np.logspace(-8, 4, 20), "max_iter": 10000, "tol": 1e-4}
                ],
            },
            {
                "order": list(range(1, 8)),
                "mindex_type": ["total_order", "hyperbolic"],
                "fit_type": ["LassoCV"],
                "fit_params": [
                    {"alphas": np.logspace(-8, 4, 20), "max_iter": 10000, "tol": 1e-4}
                ],
            },
        ]
        pce = tesuract.RegressionWrapperCV(
            regressor="pce", reg_params=pce_grid, n_jobs=-1, scorer="r2", verbose=1
        )
        pce.fit(self.X, self.y)
        print("Hyper-parameter CV PCE score is {0:.3f}".format(pce.best_score_))
        print("Total time is ", time() - start)


# regression test for multi output pca regressor
@pytest.mark.regression
class TestMRegressionWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X = np.loadtxt(relpath + "/tests/data/X_train_ESM.txt")
        self.Y = np.loadtxt(relpath + "/tests/data/Y_train_ESM.txt")

    def test_mregcv_with_esm_data(self):

        pce_grid = [
            {
                "order": list(range(1, 2)),
                "mindex_type": ["total_order"],
                "fit_type": ["linear"],
                "fit_params": [
                    {"alphas": np.logspace(-8, 4, 10), "max_iter": 10000, "tol": 1e-1}
                ],
            }
        ]

        target_transform = tesuract.preprocessing.PCATargetTransform(
            n_components=2,
            whiten=True,
            exp_var_cutoff=0.5,
        )
        regmodel = tesuract.MRegressionWrapperCV(
            regressor="pce",
            reg_params=pce_grid,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=-1,
            verbose=1,
        )
        regmodel.fit(self.X, self.Y)

    def test_mregcv_with_list_regressor_initiation(self):

        pce_grid = [
            {
                "order": list(range(1, 2)),
                "mindex_type": ["total_order"],
                "fit_type": ["linear"],
            }
        ]

        target_transform = tesuract.preprocessing.PCATargetTransform(
            n_components=2,
            whiten=True,
            exp_var_cutoff=0.5,
        )
        regmodel = tesuract.MRegressionWrapperCV(
            regressor=["pce"],
            reg_params=[pce_grid],
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=-1,
            verbose=1,
        )
        regmodel.fit(self.X, self.Y)

    def test_mregcv_with_auto_pca_target_transform_w_cv_score(self):

        pce_grid = [
            {
                "order": list(range(1, 2)),
                "mindex_type": ["total_order"],
                "fit_type": ["LassoCV"],
            }
        ]

        target_transform = tesuract.preprocessing.PCATargetTransform(
            n_components="auto",
            whiten=True,
            exp_var_cutoff=0.5,
        )
        regmodel = tesuract.MRegressionWrapperCV(
            regressor="pce",
            reg_params=pce_grid,
            target_transform=target_transform,
            target_transform_params={},
            n_jobs=-1,
            verbose=1,
        )
        regmodel.fit(self.X, self.Y)

        # Clone and compute the cv score of full model
        cv_score, surrogate_clone = regmodel.compute_cv_score(
            X=self.X, y=self.Y, scoring="r2"
        )
        print("Mean CV score:", cv_score)
