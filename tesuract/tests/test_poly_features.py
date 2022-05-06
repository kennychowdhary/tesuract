from audioop import cross
import tesuract
import unittest
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pytest
import time

relpath = tesuract.__file__[:-11]  # ignore the __init__.py specification
print(relpath)


def mse(a, b):
    return mean_squared_error(a, b, squared=False)


from sklearn.datasets import make_friedman1


@pytest.mark.unit
class TestPCERegression(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # import a data set for our simple regression problem
        X, y = make_friedman1(n_samples=100, n_features=8, random_state=777)

        # scale the data
        self.X = 2 * X - 1  # X[0,1] -> [-1,1]
        self.y = (y - y.mean()) / np.sqrt(y.var())

    def test_pcebuilder_poly_features(self):
        X, y = self.X, self.y

        # create an 8th order PCE polynomial
        pceb = tesuract.PCEBuilder(order=8)
        start = time.time()
        Xhat = pceb.fit_transform(X)
        print(Xhat.shape)
        print("Total time is:", time.time() - start)

        # create an 8th order using poly features
        pceb = tesuract.PCEBuilder(order=8, use_sklearn_poly_features=True)
        start = time.time()
        Xhat2 = pceb.fit_transform(X)
        print(Xhat2.shape)
        print("Total time is:", time.time() - start)

        assert Xhat.shape == Xhat2.shape, "poly features should match pce features."

    def test_pcereg_poly_features(self):
        X, y = self.X, self.y

        # create an 8th order PCE polynomial
        pce = tesuract.PCEReg(order=8)
        start = time.time()
        pce._compile(X)
        print(pce.Xhat.shape)
        print("Total time is:", time.time() - start)

        # create an 8th order using poly features
        pce2 = tesuract.PCEReg(order=8, use_sklearn_poly_features=True)
        start = time.time()
        pce2._compile(X)
        print(pce2.Xhat.shape)
        print("Total time is:", time.time() - start)

        assert (
            pce.Xhat.shape == pce2.Xhat.shape
        ), "poly features should match pce features."

    def test_pcereg_poly_features_fit(self):
        X, y = self.X, self.y

        # create an 8th order PCE polynomial
        pce = tesuract.PCEReg(order=8)
        start = time.time()
        scores = cross_val_score(pce, X, y, cv=5)
        print("score:", scores.mean())
        print("Total time is:", time.time() - start)

        # create an 8th order using poly features
        pce2 = tesuract.PCEReg(order=8, use_sklearn_poly_features=True)
        start = time.time()
        # pce2.fit(X, y)
        # print(pce2.coef_)
        scores2 = cross_val_score(pce2, X, y, cv=5)
        print("score2:", scores2.mean())
        print("Total time is:", time.time() - start)

        # assert (
        #     pce.Xhat.shape == pce2.Xhat.shape
        # ), "poly features should match pce features."
