import tesuract
import unittest
import numpy as np
import warnings, pdb
import time as T
import sklearn

relpath = tesuract.__file__[:-11]  # ignore the __init__.py specification
print(relpath)

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

# regression test for multi output pca regressor
class TestMRegressionWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # import a data set for our regression problem
        from sklearn.datasets import make_friedman1

        self.X, self.y = make_friedman1(n_samples=100, n_features=5, random_state=58399)

    def test_simple_pce_scalar_fit(self):
        print("Fitting 8th order polynomial...")
        pce_model = tesuract.PCEReg(order=8)
        pce_model.fit(self.X, self.y)
        print("done fitting!")
