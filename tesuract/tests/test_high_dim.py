import tesuract
import unittest
import numpy as np
import warnings, pdb
import time as T
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

relpath = tesuract.__file__[:-11] # ignore the __init__.py specification
print(relpath)



class TestPCERegression(unittest.TestCase):
	...
	# def test_efficient_poly_transform_customM(self):
	# 	# this test will ensure the model selector selector works 
	# 	X, y = make_friedman1(n_samples=50, n_features=68, random_state=0)
	# 	customM = np.load("tests/customM_test.npy")
	# 	ptest = tesuract.PCEReg(customM=customM)
	# 	start = T.time()
	# 	ptest._compile(X)
	# 	print("time = ", T.time() - start)
	# def test_efficient_poly_transform(self):
	# 	# this test will ensure the model selector selector works 
	# 	X, y = make_friedman1(n_samples=150, n_features=38, random_state=0)
	# 	# customM = np.load("tests/customM_test.npy")
	# 	ptest = tesuract.PCEReg(order=4)
	# 	start = T.time()
	# 	ptest._compile(X)
	# 	end = T.time()
	# 	print("Total time is %.5f sec" %(end-start))


		