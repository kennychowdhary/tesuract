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

class RegressionWrapper(BaseEstimator):
	def __init__(self,regressor='pce',reg_params={},scorer='neg_root_mean_squared_error',n_jobs=1,verbosity=1):
		self.regressor = regressor
		self.reg_params = reg_params
		self.scorer = scorer
		self.n_jobs = n_jobs
		self.verbosity = verbosity
		# self.__dict__.update(reg_params)
	def _setup_cv(self):
		self.cv = KFold(n_splits=5)
	def _reformat_grid(self,params):
		try: 
			# if parameter grid is in the form of grid search cv
			ParameterGrid(params)
			params_cv = params
		except:
			# convert reg_params to form that can be used by grid search
			# for a single grid point
			params_cv = defaultdict(list)
			for key,item in params.items():
				params_cv[key].append(item)
		return params_cv
	def _model_factory(self,regressor):
		if regressor == 'pce':
			return pypce.pcereg()
		if regressor == 'randforests':
			return RandomForestRegressor()
	def fit(self,X,y):
		self._setup_cv()
		if isinstance(self.regressor,str):
			model = self._model_factory(regressor=self.regressor)
			reg_params_cv = self._reformat_grid(self.reg_params)
			GridSCV = GridSearchCV(model, reg_params_cv, scoring=self.scorer, n_jobs=self.n_jobs, cv=self.cv, verbose=self.verbosity, return_train_score=True)
			GridSCV.fit(X,y)
			self.best_estimator_ = GridSCV.best_estimator_
			self.best_params_ = GridSCV.best_params_
			self.best_score_ = GridSCV.best_score_
			self.best_overfit_error_ = self.overfit_score(GridSCV)
			self.cv_results_ = GridSCV.cv_results_
		if isinstance(self.regressor,list):
			self.fit_multiple_reg(X,y)
		return self
	def fit_multiple_reg(self,X,y):
		if isinstance(self.regressor,list):
			n_regressors = len(self.regressor)
			self.best_estimators_ = [None]*n_regressors
			self.all_best_params_ = [None]*n_regressors
			self.best_scores_ = np.zeros(n_regressors)
			self.best_overfit_errors_ = [None]*n_regressors
			self.all_cv_results_ = [None]*n_regressors
			assert isinstance(self.reg_params,list), "parameters must also be a list"
			assert len(self.reg_params) == n_regressors, "length of parameters and regressors must match."
			for i,R in enumerate(self.regressor):
				print("Fitting regressor ", R)
				model = self._model_factory(regressor=R)
				reg_params_cv = self._reformat_grid(self.reg_params[i])
				GridSCV = GridSearchCV(model, reg_params_cv, scoring=self.scorer, n_jobs=self.n_jobs, cv=self.cv, verbose=self.verbosity, return_train_score=True)
				GridSCV.fit(X,y)
				self.best_estimators_[i] = GridSCV.best_estimator_
				self.all_best_params_[i] = GridSCV.best_params_
				self.best_scores_[i] = GridSCV.best_score_
				self.best_overfit_errors_[i] = self.overfit_score(GridSCV)
				self.all_cv_results_[i] = GridSCV.cv_results_
			best_index_ = np.argmax(best_score_)
			self.best_estimator_ = self.best_estimators_[best_index_]
			self.best_params_ = self.all_best_params_[best_index_]
			self.best_score_ = self.best_scores_[best_index_]
			self.best_overfit_error_ = self.best_overfit_errors_[best_index_]
			self.cv_results_ = self.all_cv_results_[best_index_]
		return self
	def predict(self,X):
		return self.best_estimator_.predict(X)
	def overfit_score(self,GridSCV):
		''' Takes a grid search cv object and returns overfit score '''
		# assert hasattr(self,'GridSCV'), "Must run fit function."
		best_index_ = GridSCV.best_index_
		mean_train_score = GridSCV.cv_results_['mean_train_score']
		mean_test_score = GridSCV.cv_results_['mean_test_score']
		return mean_train_score[best_index_] - mean_test_score[best_index_]


# data
n, d = 30, 2
rn = np.random.RandomState(45)
X = 2 * rn.rand(n, d) - 1
y = 3 + 2 * .5 * (3 * X[:, 0] ** 2 - 1) + X[:, 1] * .5 * (3 * X[:, 0] ** 2 - 1)
y2 = 1 + 2 * .5 * (3 * X[:, 1] ** 2 - 1) + 3 * X[:, 0] * .5 * (3 * X[:, 1] ** 2 - 1)
Y = np.vstack([y, y2]).T  # test for multioutput regressor
c_true = np.array([3., 0., 0., 2., 0., 0., 0., 1., 0., 0., ])

# single parameter test
reg_params = {'order': 3, 'mindex_type': 'total_order', 'fit_type': 'linear'}
RW = RegressionWrapper(regressor='pce',reg_params=reg_params)
# RW.fit(X,y)

# pce grid search cv test
pce_param_grid = [
    {'order': [1, 2, 3],
     'mindex_type': ['total_order', 'hyperbolic', ],
     'fit_type': ['LassoCV', 'linear', 'ElasticNetCV']}
]
RW2 = RegressionWrapper(regressor='pce',reg_params=pce_param_grid,n_jobs=6)


# random forest fit
rf_param_grid = {'n_estimators': [1000],
                 'max_features': ['auto','log2'],
                 'max_depth': [5,10,15]
                 }
RW3 = RegressionWrapper(regressor='randforests', reg_params=rf_param_grid,n_jobs=6)

# compute cv score for list of regressors
multreg = RegressionWrapper(regressor=['pce','pce','randforests'],reg_params=[pce_param_grid,reg_params,rf_param_grid])


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




