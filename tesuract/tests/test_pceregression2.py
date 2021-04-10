import pypce
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

relpath = pypce.__file__[:-11] # ignore the __init__.py specification
print(relpath)

def mse(a,b):
	return mean_squared_error(a,b,squared=False)

class TestPCERegression(unittest.TestCase):
	def test_checking_predict_does_recompute_mindex(self):
		# this test will ensure the model selector selector works 
		X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
		p = pypce.PCEReg(order=2)
		p.fit(X,y)
		p.predict(X)
		Xnew = X[:,:-2]
		p.fit(Xnew,y)
		p.predict(Xnew)
		print("# times computing mindex:", p._mindex_compute_count_)
		assert p._mindex_compute_count_ == 2, "mindex should be recomputed when X dim changes. "
	def test_checking_predict_does_not_recompute_mindex(self):
		# this is a test to count the number of times mindex is computed
		X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
		p = pypce.PCEReg(order=2)
		p.fit(X,y)
		p.predict(X)
		Xnew, ynew = X[10:], y[10:]
		p.fit(Xnew,ynew)
		p.predict(Xnew)
		print("# times mindex is computed: ", p._mindex_compute_count_)
		assert p._mindex_compute_count_ == 1, "mindex should NOT be recomputed when X row size changes."
	def test_k_fold_mindex_count(self):
		# this is a test to count the number of times mindex is computed
		X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
		X = 2*X - 1 # scale to [-1,1]
		kf = KFold(n_splits=5)
		p = pypce.PCEReg()
		for train_index, test_index in kf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			p.fit(X_train,y_train)
			score = np.sum(1 - p.predict(X_test)**2/y_test**2)
			# print(score)
		print("# times mindex is computed: ", p._mindex_compute_count_)
		assert p._mindex_compute_count_ == 1, "mindex should NOT be recomputed when X row size changes."
	def test_k_fold_mindex_count(self):
		# this is a test to count the number of times mindex is computed
		X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
		X = 2*X - 1 # scale to [-1,1]
		params = {'order':[1,2,3]}
		estimator = pypce.PCEReg(fit_type="LassoCV")
		grid = GridSearchCV(estimator, params)
		grid.fit(X,y)
		# print(grid.cv_results_)
		print("# times computed...", grid.best_estimator_._mindex_compute_count_)
		assert grid.best_estimator_._mindex_compute_count_ == 1, "Best estimator should only compute mindex once!"
	def test_feature_importance_friendman_example(self):
		# this is a test to count the number of times mindex is computed
		X, y = make_friedman1(n_samples=100, n_features=10, random_state=0)
		X = 2*X - 1 # scale to [-1,1]
		estimator = pypce.PCEReg(order=4,fit_type="LassoCV")
		estimator.fit(X,y)
		fi = estimator.feature_importances_
		print(fi > .05)
		assert np.sum(fi>.05) == 5, "friedman function should have 5 important features only."
	def test_grid_search_cv_picks_right_order(self):
		# 
		rn = np.random.RandomState(99)
		X = 2*rn.rand(50,5)-1
		fi_index = 3
		x = X[:,3]
		# y = (1./8)*(35.*x**4 - 30*x**4 + 3)
		y = (1./2)*(5*x**3 - 3*x)
		params = {'order':[1,2,3,4]}
		estimator = pypce.PCEReg(fit_type="LassoCV")
		start = T.time()
		grid = GridSearchCV(estimator, params, cv=KFold(n_splits=5))
		grid.fit(X,y)
		end = T.time()
		print("Total grid search time is {0:.2f} seconds.".format(end-start))
		# print(grid.cv_results_)
		print("\nbest parameters:", grid.best_params_)
		fi = grid.best_estimator_.feature_importances_
		# print("\nfeatures:", grid.best_estimator_.feature_importances_)
		assert grid.best_params_['order'] == 3, "Grid search should find the best order 4, but it is not. Something is wrong."
		assert np.argmax(fi) == fi_index, "Not finding the right feature importance index."
	def test_grid_search_cv_picks_right_order_and_features(self):
		# 
		rn = np.random.RandomState(99)
		X = 2*rn.rand(50,5)-1
		fi_index = 1
		x = X[:,fi_index]
		y = (1./8)*(35.*x**4 - 30*x**4 + 3)
		# y = (1./2)*(5*x**3 - 3*x)
		params = {'order':[1,2,3,4]}
		estimator = pypce.PCEReg(fit_type="LassoCV")
		start = T.time()
		grid = GridSearchCV(estimator, params, cv=KFold(n_splits=5))
		grid.fit(X,y)
		end = T.time()
		print("Total grid search time is {0:.2f} seconds.".format(end-start))
		print(grid.cv_results_)
		print("\nbest parameters:", grid.best_params_)
		fi = grid.best_estimator_.feature_importances_
		# print("\nfeatures:", fi)
		assert grid.best_params_['order'] == 3, "Grid search should find the best order 4, but it is not. Something is wrong."
		assert np.argmax(fi) == fi_index, "Not finding the right feature importance index."
		# print("# times computed...", grid.best_estimator_._mindex_compute_count_)


		