import numpy as np 
import pypce
from collections import defaultdict

from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.estimator_checks import check_estimator 
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.model_selection import ParameterGrid 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer

# data
n, d = 30, 2
rn = np.random.RandomState(45)
X = 2 * rn.rand(n, d) - 1
y = 3 + 2 * .5 * (3 * X[:, 0] ** 2 - 1) + X[:, 1] * .5 * (3 * X[:, 0] ** 2 - 1)
y2 = 1 + 2 * .5 * (3 * X[:, 1] ** 2 - 1) + 3 * X[:, 0] * .5 * (3 * X[:, 1] ** 2 - 1)
Y = np.vstack([y, y2]).T  # test for multioutput regressor
c_true = np.array([3., 0., 0., 2., 0., 0., 0., 1., 0., 0., ])


# domain test
class DomainScaler2:
	def __init__(self,dim,input_range,output_range=(-1,1)):
		self.dim = dim
		self.input_range = input_range
		self.output_range = output_range
	def _get_bound_list(self,input_range):
		if isinstance(input_range,list):
			assert len(input_range) == self.dim, "input range must be a list of tuples"
			input_bounds = input_range
		if isinstance(input_range,tuple):
			input_bounds = [(input_range[0],input_range[1]) for i in range(self.dim)]
		a = np.array([ab[0] for ab in input_bounds]) # lower bounds
		b = np.array([ab[1] for ab in input_bounds]) # upper bounds
		return input_bounds,a,b
	def _range_check(self,X,B):
		n,d = X.shape
		assert d == self.dim, "columns of X must be the same as dimensions"
		assert len(B) == self.dim, "length of bounds list must be same as dimensions"
		dim_check = [(X[:,i] >= B[i][0]).all() and (X[:,i] <= B[i][1]).all() for i in range(d)]
		assert all(dim_check), "X is not in the range of the input range."
		return X
	def fit(self,X):
		self.input_bounds,a,b = self._get_bound_list(self.input_range)
		self.output_bounds,c,d = self._get_bound_list(self.output_range)
		X = self._range_check(X,self.input_bounds)
		# transform to [0,1] first for ease
		X_unit_scaled = (X - a)/(b-a)
		# transform to output bounds
		X_scaled = (d-c)*X_unit_scaled + c
		X_scaled = self._range_check(X_scaled,self.output_bounds)
		return X_scaled
	def fit_transform(self,X):
		self.fit(X)
		return self
	def inverse_transform(self,Xhat):
		self.input_bounds,a,b = self._get_bound_list(self.input_range)
		self.output_bounds,c,d = self._get_bound_list(self.output_range)
		Xhat = self._range_check(Xhat,self.output_bounds)
		Xhat_unit_scaled = (Xhat - c)/(d-c)
		X_inv = (b-a)*Xhat_unit_scaled + a
		X_inv = self._range_check(X_inv,self.input_bounds)
		return X_inv 

X_1 = X.copy() # [-1,1] for both dimensions
X_2 = .5*(X+1) # [0,1] for both dimensions
X_3 = np.array([X.T[0]+1,3+.5*X.T[1]]).T # [0,2]x[2.5,3.5] 

D = DomainScaler2(dim=2,input_range=(-1,1),output_range=(-1,1))
X_scaled = D.fit(X_1) # difference should be zero
assert (X_scaled - X_1).sum() == 0

D = DomainScaler2(dim=2,input_range=(-1,1),output_range=(0,1))
X_scaled = D.fit(X_1) # difference should be zero
assert (np.amin(X_scaled,axis=0)>=0).all() and (np.amax(X_scaled,axis=0)<=1).all()

D = DomainScaler2(dim=2,input_range=(-1,1),output_range=[(0,1),(3,15)])
X_scaled = D.fit(X_1) # difference should be zero

D = DomainScaler2(dim=2,input_range=[(0,1),(3,15)],output_range=(-1,1))
X_scaled = D.fit(X_scaled) # difference should be zero

D = DomainScaler2(dim=2,input_range=(0,1),output_range=(-1,1))
X_scaled = D.fit(X_2) # difference should be zero

D = DomainScaler2(dim=2,input_range=[(0,2),(2.5,3.5)],output_range=(-1,1))
X_scaled = D.fit(X_3) # difference should be zero

D = DomainScaler2(dim=2,input_range=(-1,1),output_range=[(0,2),(2.5,3.5)])
X_scaled_new = D.fit(X_scaled) # difference should be zero
assert (X_scaled_new - X_3).sum() == 0

X_inv = D.inverse_transform(X_scaled_new)
assert (X_inv - X_scaled).sum() == 0
