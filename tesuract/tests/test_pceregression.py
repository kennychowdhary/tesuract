import tesuract
import unittest
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error
import pytest

relpath = tesuract.__file__[:-11]  # ignore the __init__.py specification
print(relpath)


def mse(a, b):
    return mean_squared_error(a, b, squared=False)


@pytest.mark.unit
class TestPCERegressionArbDomain(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dim = 2
        self.order = 3
        self.nsamples = 27
        rn = np.random.RandomState(123)
        X = 2 * rn.rand(self.nsamples, self.dim) - 1
        self.ibounds = [(-1, 1) for r in range(self.dim)]
        y = (
            3.14
            + 1.74 * 0.5 * (3 * X[:, 0] ** 2 - 1)
            + X[:, 1] * 0.5 * (3 * X[:, 0] ** 2 - 1)
        )
        self.c_true = np.array(
            [
                3.14,
                0.0,
                0.0,
                1.74,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ]
        )
        self.X, self.y = X, y
        self.rn = rn

    def domain_setup(self, a=0, b=5):
        # on (0,5) domain
        X = (b - a) * self.rn.rand(self.nsamples, self.dim) + a
        ibounds = [(a, b) for r in range(self.dim)]
        y = (
            3
            + 2 * 0.5 * (3 * X[:, 0] ** 2 - 1)
            + X[:, 1] * 0.5 * (3 * X[:, 0] ** 2 - 1)
        )
        # self.c_true = np.array([3.14,0.,0.,1.74,0.,0.,0.,1.,0.,0.,])
        return X, y, ibounds

    def test_pcereg_w_canonical_domain(self):
        X, y, ibounds = self.X, self.y, self.ibounds
        pce = tesuract.PCEReg(order=3, input_range=ibounds)
        pce.fit(X, y)
        assert mse(y, pce.predict(X)) <= 1e-14
        # print(np.around(pce.coef_,3))
        assert mse(np.around(pce.coef_, 3), self.c_true) <= 1e-16

    def test_pcereg_w_non_unit_domain(self):
        X, y, ibounds = self.domain_setup()
        pce = tesuract.PCEReg(order=3, input_range=ibounds)
        pce.fit(X, y)
        assert (
            mse(y, pce.predict(X)) <= 1e-12
        ), "predict is not working properly for input ranges that are not on [-1,1]."

    def test_pcereg_w_unit_domain(self):
        X, y, ibounds = self.domain_setup(a=0, b=5)
        pce = tesuract.PCEReg(order=3, input_range=ibounds)
        pce.fit(X, y)
        assert (
            mse(y, pce.predict(X)) <= 1e-12
        ), "predict is not working properly for input ranges that are not on [-1,1]."


@pytest.mark.unit
class TestPCERegression(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dim = 2
        self.order = 3
        self.nsamples = 30
        rn = np.random.RandomState(123)
        X = 2 * rn.rand(self.nsamples, self.dim) - 1
        y = (
            3
            + 2 * 0.5 * (3 * X[:, 0] ** 2 - 1)
            + X[:, 1] * 0.5 * (3 * X[:, 0] ** 2 - 1)
        )
        self.c_true = np.array(
            [
                3.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ]
        )
        self.X, self.y = X, y
        self.rn = rn

    def test_linear_fit(self):
        p = tesuract.PCEReg(self.order)
        p.fit(self.X, self.y)
        assert (
            np.mean(np.abs(p.coef - self.c_true)) <= 1e-12
        ), "Fit coefficient did not match truth."

    def test_linear_fit_w_normalization(self):
        p = tesuract.PCEReg(self.order, normalized=True)
        p.fit(self.X, self.y)
        normsq = p.computeNormSq()
        assert (
            mse(p.coef / np.sqrt(normsq), self.c_true) <= 5e-12
        ), "Fit coefficient did not match truth."

    def test_linear_fit_prediction_w_normalization(self):
        p = tesuract.PCEReg(self.order, normalized=True)
        p.fit(self.X, self.y)
        assert (
            mse(p.predict(self.X), self.y) <= 5e-12
        ), "linear fit with normalization is broken."

    def test_multindex(self):
        p = tesuract.PCEReg(self.order)
        p.fit(self.X, self.y)
        M = p._M
        assert np.sum(M - p.mindex) == 0, "Multindex does not match."

    def test_custom_index(self):
        p0 = tesuract.PCEReg(self.order)
        p0.fit(self.X, self.y)
        p = tesuract.PCEReg(order=self.order, customM=p0._M)
        p.fit(self.X, self.y)
        assert (
            np.mean(np.abs(p.coef - self.c_true)) <= 1e-12
        ), "Fit coefficient did not match truth."

    def test_custom_feature_importance(self):
        rn = np.random.RandomState(123)
        X = 2 * rn.rand(100, 2) - 1
        y = X[:, 0] + 0.5 * (3 * X[:, 1] ** 2 - 1)
        p0 = tesuract.PCEReg(self.order)
        p0.fit(X, y)
        fi0 = p0.feature_importances_
        p = tesuract.PCEReg(customM=p0.mindex[1:])
        p.fit(X, y)
        fi = p.feature_importances_
        assert (
            np.sum((fi - fi0) ** 2) <= 1e-16
        ), "feature importance for custom multiindex failed."
        # assert np.mean(np.abs(p.coef - self.c_true)) <= 1e-12, "Fit coefficient did not match truth."

    def test_LassoCV_fit(self):
        p = tesuract.PCEReg(self.order, fit_type="LassoCV")
        y = self.y + 0.001 * self.rn.rand(len(self.y))  # add noise
        p.fit(self.X, y)
        assert (
            np.mean(np.abs(p.coef - self.c_true)) <= 2e-2
        ), "LassoCV not accurate enough."

    def test_OmpCV_fit(self):
        p = tesuract.PCEReg(self.order, fit_type="OmpCV")
        y = self.y + 0.001 * self.rn.rand(len(self.y))  # add noise
        p.fit(self.X, y)
        assert (
            np.mean(np.abs(p.coef - self.c_true)) <= 2e-1
        ), "OmpCV not accurate enough."

    def test_ElasticNetCV_fit(self):
        p = tesuract.PCEReg(self.order, fit_type="ElasticNetCV")
        p.fit(self.X, self.y)
        assert (
            np.mean(np.abs(p.coef - self.c_true)) <= 2e-2
        ), "ElasticNetCV error is not small enough."

    def test_predict_in_base_vs_child_class(self):
        p = tesuract.PCEReg(self.order)
        p.fit(self.X, self.y)
        ypred1 = p.predict(self.X)
        ypred2 = p.polyeval(self.X)
        assert (
            np.mean(np.abs(ypred1 - ypred2)) == 0
        ), "Mismatch between eval and prediction."

    def test_predict_in_base_vs_child_with_normalization(self):
        p = tesuract.PCEReg(self.order, normalized=True)
        p.fit(self.X, self.y)
        ypred1 = p.predict(self.X)
        ypred2 = p.polyeval(self.X)
        assert (
            np.mean(np.abs(ypred1 - ypred2)) == 0
        ), "Mismatch between eval and prediction."

    def test_single_predict(self):
        p = tesuract.PCEReg(self.order)
        p.fit(self.X, self.y)
        ypred1 = p.predict(self.X[0])
        ypred2 = p.polyeval(self.X[0])
        assert (
            np.mean(np.abs(ypred1 - ypred2)) == 0
        ), "Mismatch between eval and prediction."

    def test_set_custom_multiindex(self):
        M = np.array([[0, 0], [2, 0], [2, 1]])
        p = tesuract.PCEReg(order=self.order, customM=M)
        p.fit(self.X, self.y)
        assert (
            np.mean(p.coef - np.array([3.0, 2.0, 1.0])) <= 1e-10
        ), "coefficients did not converge for custom multindex."
        assert (
            np.mean(np.abs(self.y - p.predict(self.X))) <= 1e-10
        ), "Fit did not converge and/or predict did not match pce.eval."

    def test_grid_search_cv(self):
        X = 2 * self.rn.rand(self.nsamples, self.dim) - 1
        y = (
            3
            + 2 * 0.5 * (3 * X[:, 0] ** 2 - 1)
            + X[:, 1] * 0.5 * (3 * X[:, 0] ** 2 - 1)
        )
        param_grid = [
            {
                "order": [1, 3],
                "mindex_type": [
                    "total_order",
                    "hyperbolic",
                ],
                "fit_type": ["LassoCV", "linear"],
            }
        ]
        from sklearn.model_selection import GridSearchCV

        pceCV = GridSearchCV(
            tesuract.PCEReg(), param_grid, scoring="neg_root_mean_squared_error"
        )
        pceCV.fit(X, y)
        assert mse(self.c_true.shape, pceCV.best_estimator_.coef.shape) < 1e-12

    def test_fit_with_2d_quadrature(self):
        X = np.loadtxt(relpath + "/tests/data/X_2dquad_points.txt")
        y = (
            3
            + 2 * 0.5 * (3 * X[:, 0] ** 2 - 1)
            + X[:, 1] * 0.5 * (3 * X[:, 0] ** 2 - 1)
        )
        c0 = np.array(
            [
                3.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ]
        )
        w = np.loadtxt(relpath + "/tests/data/w_2dquad_weights.txt")
        p = tesuract.PCEReg(
            order=3, normalized=False, fit_type="quadrature", fit_params={"w": w}
        )
        p.fit(X, y)
        assert (
            mse(p.coef, c0) <= 1e-14
        ), "quadratue with non normalized basis did not work. "

    def test_fit_with_2d_quadrature_and_normalized_basis(self):
        X = np.loadtxt(relpath + "/tests/data/X_2dquad_points.txt")
        y = (
            3
            + 2 * 0.5 * (3 * X[:, 0] ** 2 - 1)
            + X[:, 1] * 0.5 * (3 * X[:, 0] ** 2 - 1)
        )
        c0 = np.array(
            [
                3.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ]
        )
        w = np.loadtxt(relpath + "/tests/data/w_2dquad_weights.txt")
        p = tesuract.PCEReg(
            order=3, normalized=False, fit_type="quadrature", fit_params={"w": w}
        )
        p.fit(X, y)
        assert (
            mse(p.coef, c0) <= 1e-14
        ), "quadratue with non normalized basis did not work. "

    def test_quadrature_assert_error_wo_no_weights(self):
        X = np.loadtxt(relpath + "/tests/data/X_2dquad_points.txt")
        y = (
            3
            + 2 * 0.5 * (3 * X[:, 0] ** 2 - 1)
            + X[:, 1] * 0.5 * (3 * X[:, 0] ** 2 - 1)
        )
        c0 = np.array(
            [
                3.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ]
        )
        w = np.loadtxt(relpath + "/tests/data/w_2dquad_weights.txt")
        p = tesuract.PCEReg(order=3, normalized=False, fit_type="quadrature")
        with self.assertRaises(AssertionError):
            p.fit(X, y)

    def test_fit_predict_w_store_false(self):
        p2 = tesuract.PCEReg(self.order, fit_type="linear")
        y2 = self.y + 0 * 0.001 * self.rn.rand(self.nsamples)  # add noise
        p2.fit(self.X, y2)
        # print(np.around(p2.coef_))
        assert mse(p2.predict(self.X), y2) < 1e-12, "accuracy loss. check. "
        self.assertRaises(AttributeError, getattr, p2, "Phi")

    def test_fit_w_store_true(self):
        p2 = tesuract.PCEReg(self.order, fit_type="linear", store_phi=True)
        y2 = self.y + 0 * 0.001 * self.rn.rand(self.nsamples)  # add noise
        p2.fit(self.X, y2)
        assert hasattr(
            p2, "Phi"
        ), "object does not have Phi, but it should if store_phi=True"

    def test_Xhat_w_store_true(self):
        p2 = tesuract.PCEReg(self.order, fit_type="linear", store_phi=True)
        y2 = self.y + 0 * 0.001 * self.rn.rand(self.nsamples)  # add noise
        p2.fit(self.X, y2)
        assert mse(p2.predict(self.X), y2) < 1e-12, "accuracy loss. check. "
        assert hasattr(
            p2, "Xhat"
        ), "object does not have Xhat, but it should if store_phi=True"

    def test_Xhat_w_store_false(self):
        p2 = tesuract.PCEReg(self.order, fit_type="linear")
        y2 = self.y + 0 * 0.001 * self.rn.rand(self.nsamples)  # add noise
        p2.fit(self.X, y2)
        assert mse(p2.predict(self.X), y2) < 1e-12, "accuracy loss. check. "
        self.assertRaises(AttributeError, getattr, p2, "Xhat")
