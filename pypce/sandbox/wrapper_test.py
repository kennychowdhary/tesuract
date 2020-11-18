import numpy as np 
import pypce
from collections import defaultdict
import time as T
import matplotlib.pyplot as mpl
import warnings, pdb
 
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.estimator_checks import check_estimator 
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
		
class RegressionWrapperCV(BaseEstimator):
	def __init__(self,regressor='pce',reg_params={},scorer='neg_root_mean_squared_error',n_jobs=1,verbose=1,cv=None):
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
			return pypce.PCEReg()
		if regressor == 'randforests':
			return RandomForestRegressor()
	def fit(self,X,y):
		self._setup_cv()
		if isinstance(self.regressor,str):
			model = self._model_factory(regressor=self.regressor)
			reg_params_cv = self._reformat_grid(self.reg_params)
			GridSCV = GridSearchCV(model, reg_params_cv, scoring=self.scorer, n_jobs=self.n_jobs, cv=self.cv, verbose=self.verbose, return_train_score=True)
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
				GridSCV = GridSearchCV(model, reg_params_cv, scoring=self.scorer, n_jobs=self.n_jobs, cv=self.cv, verbose=self.verbose, return_train_score=True)
				GridSCV.fit(X,y)
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
	def predict(self,X):
		return self.best_estimator_.predict(X)
	def overfit_score(self,GridSCV):
		''' Takes a grid search cv object and returns overfit score '''
		# assert hasattr(self,'GridSCV'), "Must run fit function."
		best_index_ = GridSCV.best_index_
		mean_train_score = GridSCV.cv_results_['mean_train_score']
		mean_test_score = GridSCV.cv_results_['mean_test_score']
		return mean_train_score[best_index_] - mean_test_score[best_index_]

from tqdm import tqdm
class MRegressionWrapperCV(BaseEstimator, RegressorMixin):
	def __init__(self, 
				regressor='pce', reg_params={'order':2},
				target_transform=None,
				target_transform_params={},
				scorer='neg_root_mean_squared_error',
				n_jobs=4,cv=None,
				mixed=True,
				verbose=1):
		self.regressor = regressor
		self.reg_params = reg_params
		self.scorer = scorer
		self.n_jobs = n_jobs
		self.cv = cv
		self.target_transform = target_transform
		self.target_transform_params = target_transform_params
		self.mixed = mixed # to control whether mixed surrogate for each target TBD
		self.verbose = verbose
	def _setupCV(self, shuffle=False, randstate=13):
		if self.cv == None:
			self.cv = KFold(n_splits=5)  # ,shuffle=True,random_state=13)
	def _fit_target_transform(self,Y):
		if self.target_transform is None:
			self.TT = FunctionTransformer(lambda Y: Y)
		else:
			self.TT = self.target_transform(**self.target_transform_params)
		self.TT.fit(Y)
		return self
	def fit(self,X,Y):
		self.n, self.dim = X.shape
		self._setupCV()
		self._fit_target_transform(Y)
		Yhat = self.TT.transform(Y)
		if Yhat.ndim == 1: Yhat = np.atleast_2d(Yhat).T
		assert len(Yhat) == self.n, "mistmatch in no of samples btwn X and Y."
		self.ntargets = Yhat.shape[1]
		self._setupCV()
		if isinstance(self.regressor,str):
			self.res = self.fit_single_reg(X,Yhat)
		elif isinstance(self.regressor,list) and self.mixed == False:
			self.res = self.fit_multiple_reg(X,Yhat)
		elif isinstance(self.regressor,list) and self.mixed == True:
			self.res = self.fit_single_reg(X,Yhat)
		return self
	def fit_single_reg(self, X, Y, regressor=None,reg_params=None):
		if regressor == None:
			regressor = self.regressor
			reg_params = self.reg_params
		res = defaultdict(list)
		for i in tqdm(range(self.ntargets)):
			reg = RegressionWrapperCV(
				regressor=regressor,reg_params=reg_params,
				n_jobs=self.n_jobs, scorer=self.scorer, cv=self.cv, verbose=self.verbose)
			reg.fit(X, Y[:, i])
			res['best_estimators_'].append(reg.best_estimator_)
			res['best_params_'].append(reg.best_params_)
			res['best_scores_'].append(reg.best_score_)
			res['best_overfit_error_'].append(reg.best_overfit_error_)
		self.__dict__.update(res)
		return res
	def fit_multiple_reg(self, X, Y):
		# wont execute if mixed is True
		if isinstance(self.regressor,list):
			mres = []
			for i,r in enumerate(self.regressor):
				res = self.fit_single_reg(X,Y,
								regressor=r,
								reg_params=self.reg_params[i])
				mres.append(res)
		return mres
	def predict(self,X):
		assert hasattr(self,'res'), "Must run fit."
		if isinstance(self.res,dict):
			Ypred = self._predict_single(X,res=self.res)
		elif isinstance(self.res,list):
			Ypred = self._predict_multiple(X,res=self.res)
		return Ypred
	def _predict_single(self,X,res=None):
		# assert isinstance(self.res,dict), "Must pass string as regressor OR list with mixed=True, otherwise, predict is ambiguous"
		assert isinstance(res,dict), "for single prediction, results must be a dictionary."
		Yhatpred_list = []
		for estimator in self.best_estimators_:
			if X.ndim == 1:
				X = np.atleast_2d(X)
			Yhatpred_list.append(estimator.predict(X))
		Yhatpred = np.array(Yhatpred_list).T
		Ypred = self.TT.inverse_transform(Yhatpred)
		return Ypred
	def _predict_multiple(self,X,res=None):
		# assert isinstance(self.res,dict), "Must pass string as regressor OR list with mixed=True, otherwise, predict is ambiguous"
		assert isinstance(res,list), "for multiple predictions, results must be a list of dictionaries."
		predictions = []
		for r in res:
			predictions.append(self._predict_single(X,r))
		return np.squeeze(np.array(predictions))
	def feature_importances_(self):
		assert hasattr(self,'res'), "Must run .fit() first!"
		FI_ = []
		for estimator in self.best_estimators_:
			fi = estimator.feature_importances_
			FI_.append(fi)
		return np.array(FI_)

class PCATargetTransform(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, cutoff=1e-2, svd_solver='arpack'):
        self.n_components = n_components
        self.cutoff = cutoff
        self.svd_solver = svd_solver
        self.K = None
    def fit(self, Y):
        self.n, self.d = Y.shape
        if self.n_components == 'auto':
            self._compute_K(Y)
        if isinstance(self.n_components, int):
            self.K = self.n_components
            if self.n_components > self.d:
                warnings.warn("No of components greater than dimension. Setting to maximum allowable.")
                self.n_components = self.d
        assert self.K is not None, "K components is not defined or being set properly."
        self.pca = PCA(n_components=self.K, svd_solver=self.svd_solver)
        self.pca.fit(Y)
        self.cumulative_error = np.cumsum(self.pca.explained_variance_ratio_)
        return self
    def fit_transform(self, Y):
        self.fit(Y)
        return self.pca.transform(Y)
    def transform(self, Y):
        assert hasattr(self, 'pca'), "Perform fit first."
        return self.pca.transform(Y)
    def inverse_transform(self, Yhat):
        return self.pca.inverse_transform(Yhat)
    def _compute_K(self, Y, max_n_components=50):
        ''' automatically compute number of PCA terms '''
        pca = PCA(n_components=min(max_n_components, self.d), svd_solver=self.svd_solver)
        pca.fit(Y)
        cumulative_error = np.cumsum(pca.explained_variance_ratio_)
        print(cumulative_error)
        loc = np.where(1 - cumulative_error <= self.cutoff)[0]
        if loc.size == 0:
        	warnings.warn("Cutoff may be too big. Setting K = 16")
        	self.K = 16
        elif loc.size > 0:
	        self.K = loc[0]


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
RW = RegressionWrapperCV(regressor='pce',reg_params=reg_params)
# RW.fit(X,y)

# pce grid search cv test
pce_param_grid = [{'order': [1, 2, 3, 4],
	 'mindex_type': ['total_order', 'hyperbolic', ],
	 'fit_type': ['LassoCV', 'linear', 'ElasticNetCV']}]
RW2 = RegressionWrapperCV(regressor='pce',reg_params=pce_param_grid,n_jobs=6)


# random forest fit
rf_param_grid = {'n_estimators': [100,300],
				 'max_features': ['auto'],
				 'max_depth': [5,10]
				 }
RW3 = RegressionWrapperCV(regressor='randforests', reg_params=rf_param_grid,n_jobs=8)

# random forest fit
rf_param_grid2 = {'n_estimators': [1000],
				 'max_features': ['log2'],
				 'max_depth': [5]
				 }
RW4 = RegressionWrapperCV(regressor='randforests',reg_params=rf_param_grid2,n_jobs=8)

# compute cv score for single target and list of regressors
multreg = RegressionWrapperCV(
		regressor=['pce','pce','randforests','randforests'],
		reg_params=[pce_param_grid,reg_params,rf_param_grid,rf_param_grid2],
		n_jobs=6)

# fit multiple target with single type
mtarg1a = MRegressionWrapperCV(
			regressor='pce',
			reg_params=pce_param_grid,
			n_jobs=8)

# fit multiple target with single type
mtarg1b = MRegressionWrapperCV(
			regressor='randforests',
			reg_params=rf_param_grid,
			n_jobs=8)

# fit multiple target with type as single element list
mtarg2 = MRegressionWrapperCV(
			regressor=['pce'],
			reg_params=[pce_param_grid],
			n_jobs=8)


# fit multiple target with more than one type and mixed basis
mtarg3 = MRegressionWrapperCV(
			regressor=['pce','randforests'],
			reg_params=[pce_param_grid,rf_param_grid],
			n_jobs=8,
			mixed=True)

# fit multiple target with more than one type
# returns a list of models, not suitable for cross-validation
# probably won't include because we can't use validation score from sklearn
mtarg4 = MRegressionWrapperCV(
			regressor=['pce','randforests'],
			reg_params=[pce_param_grid,rf_param_grid],
			n_jobs=4,
			mixed=False)

# cross validation test
def multi_target_metric(y_true,y_pred):
	if y_true.ndim == 2 and y_true.ndim == 2: 
	    error_array =  np.linalg.norm(y_true - y_pred,axis=1)/np.linalg.norm(y_true,axis=1)
	    return error_array.mean()
	else:
		return np.linalg.norm(y_true - y_pred)/np.linalg.norm(y_true)
scorer = make_scorer(multi_target_metric,greater_is_better=False)
cv_outer = KFold(n_splits=5)
# cv_results = cross_validate(mtarg1a,X,Y,scoring=scorer,cv=cv_outer,verbose=1,return_train_score=True)

# grid search cv
param_grid = [
			  {'regressor': ['pce'], 'reg_params': [pce_param_grid]},
			  {'regressor': ['randforests'], 'reg_params': [rf_param_grid]}
			 ]
gsCV = GridSearchCV(MRegressionWrapperCV(n_jobs=8,verbose=0), param_grid, 
					scoring=scorer, cv=cv_outer, 
					verbose=1, return_train_score=True)
start = T.time()
# gsCV.fit(X,Y)
end = T.time()
print("Total time is %.5f seconds" %(end-start))

################################################################
# PCA Target Transform
################################################################
print("Performing PCA transform...")
# test target transform
# Load the data
data_fields = ['heat_flux','df_pressure','df_wall-heat-flux',
			   'df_wall-C_f','df_wall-C_h','df_wall-friction-velocity',
			   'df_wall-tau-mag']
# for d in data_fields[1:]:
# 	test = np.load(datadir + 'data/' + d + '.npy')
# 	np.save('Y_' + d + '.npy', test)
data_field = data_fields[0]
# for data_field in data_fields:
print(data_field)
datadir = '/Users/kchowdh/Research/sparc_tests/'
if data_field == 'heat_flux':
	x_domain = np.load(datadir + 'data/x_heat_flux.npy')
	X = np.load(datadir + 'data/LHS_' + data_field + '.npy')
else:
	x_domain = np.load(datadir + 'data/x_df_domain.npy')
	X = np.load(datadir + 'data/LHS_df_samples.npy')
Y = np.load(datadir + 'data/Y_' + data_field + '.npy')

Q = [.025,.5,.975]

# Scale target Y (use inverse transform for new data)
# Each row is a functional output, so we should scale each row
target_scaler = pypce.preprocessing.MinMaxTargetScaler(target_range=(0, 1))
Y_scaled = target_scaler.fit_transform(Y)
Y_q = np.quantile(Y_scaled,Q,axis=0)

# Lower and Upper Bound for features in X (for future scaling)
X_LB = np.array([.85, .85, .85, 0.7, 0.8, 0.3, 0.7, 0.0784, 0.38, 0.31, 1.19, 1.05])
X_UB = np.array([1.15, 1.15, 1.15, 1.0, 1.2, 0.7, 1.0, 0.1024, 0.42, 0.40, 1.31, 1.45])
X_bounds = list(zip(X_LB,X_UB))

X_col_names = ['dens_sc', 'vel_sc', 'temp_sc',
               'sigk1', 'sigk2', 'sigw1', 'sigw2',
               'beta_s', 'kap', 'a1', 'beta1r', 'beta2r']

# Scale features X (use inverse transform for new data)
feature_scaler = pypce.preprocessing.DomainScaler(dim=12, input_range=X_bounds, output_range=(-1, 1))
X_scaled = feature_scaler.fit_transform(X)

# pca_params={'n_components':'auto','cutoff':1e-3}
pca_params={'n_components':8}
pca = PCATargetTransform(**pca_params)
pca.fit(Y_scaled)
Yhat_scaled = pca.transform(Y_scaled)
print('n components = ', Yhat_scaled.shape[1])
Y_pca_recon0 = pca.inverse_transform(Yhat_scaled)
Y_pca_q = np.quantile(Y_pca_recon0,Q,axis=0)

# fit multiple target with type as single element list
# HFreg = MRegressionWrapperCV(
# 			regressor=['pce'],
# 			reg_params=[pce_param_grid],
# 			n_jobs=8)
# HFreg.fit(X_scaled,Yhat_scaled)
# Yhat_pred = HFreg.predict(X_scaled)
# Y_pce_recon = pca.inverse_transform(Yhat_pred)
# Y_pce_q = np.quantile(Y_pce_recon,Q,axis=0)


# fit multiple target with type as single element list
start = T.time()
HFreg2 = MRegressionWrapperCV(
			regressor=['pce'],
			reg_params=[pce_param_grid],
			target_transform=PCATargetTransform,
			target_transform_params=pca_params,
			n_jobs=8)
# HFreg2.fit(X_scaled,Y_scaled)
end = T.time()
print("Total time is %.5f seconds" %(end-start))
# Y_pce_recon2 = HFreg2.predict(X_scaled)
# Y_pce_q = np.quantile(Y_pce_recon2,Q,axis=0)
# print(HFreg2.best_scores_)

#######################
## Save and calibrate
#######################
def calibrate():

	# save model
	from joblib import dump, load
	dump(HFreg2,'regmodel.joblib')
	regmodel = load('regmodel.joblib')
	# regmodel = HFreg2

	# load observations for heat flux only
	# interpolate predictions for obersvations
	# if data_field == 'heat_flux_wall':
	x_heat_flux = x_domain #np.load(datadir + 'data/x_heat_flux.npy')
	obs = np.loadtxt(datadir + 'data/mks_hifire1_run30_q.dat')
	x_obs, y_obs = obs.T
	# transform y_obs using 
	y_obs_scaled = target_scaler.transform(y_obs)

	# interpolate
	from scipy.interpolate import interp1d
	def interp_heat_flux(y):
		# yi = np.interp(x_obs,x_heat_flux,y)
		fi = interp1d(x_heat_flux,y)
		yi = fi(x_obs)
		return yi

	def objfun(x_uq):
		y = regmodel.predict(x_uq)
		yi = interp_heat_flux(y)
		yi = np.atleast_2d(yi)
		y_true = np.atleast_2d(y_obs_scaled)
		ii = (x_obs >= 0) & (x_obs <= 0.65)
		error = np.mean((y_true[:,ii] - yi[:,ii])**2, axis=1)/np.mean(y_true[:,ii]**2)
		return error, yi


	def func(x):
		dim = 12
		xref = np.zeros(dim)
		# xref[[0,1,9]] = x
		xref[:] = x
		error, yi = objfun(xref)
		return error

	dim = len(X_col_names)
	opt_dim = 12
	X0 = []
	x0 = np.zeros(opt_dim) #X_scaled[0]
	X0.append(x0)
	for i in range(100):
		X0.append(np.random.rand(opt_dim))
	xref = np.zeros(dim)
	error0, ypred0 = objfun(xref)

	# optimize
	from scipy.optimize import fmin_l_bfgs_b, fmin_tnc, fmin_slsqp
	dim = X_scaled.shape[1]
	bounds = [(-1.0,1.0) for d in range(opt_dim)]

	Nfeval = 1
	def minimize_call(xstart,factr=1):
		def callbackF(Xi):
			global Nfeval
			if Nfeval % 10 == 0: print('Iteration %i' %Nfeval)
			Nfeval += 1
		# neglogpost = lambda x: -1*logpost_emcee(x,model_info,LB,UB)
		# lbfgs_options = {} #{"factr": 1e2, "pgtol": 1e-10, "maxiter": 200}
		res = fmin_l_bfgs_b(func,xstart,approx_grad=True,bounds=bounds,
								factr=factr,callback=callbackF,pgtol=1e-13,disp=2)
		return res
	# res = fmin_l_bfgs_b(func,x0,approx_grad=True,bounds=bounds,
	# 							factr=1e2,pgtol=1e-8,disp=2)
	# res = fmin_tnc(func,x0,approx_grad=True,bounds=bounds)
	# res = fmin_slsqp(func, x0, iter=1000, bounds=bounds, acc=1e-9, iprint=2, full_output=True)
	# x_opt = res[0]
	# y_opt = regmodel.predict(x_opt)

	from functools import partial
	myfun = partial(minimize_call, factr=1e2)
	# from multiprocessing import Pool
	# nprocs = 1
	# p = Pool(nprocs) 
	# start = T.time()
	# res = p.map(myfun,[x for x in X0[:nprocs]])
	# end = T.time()
	# p.close()
	# p.terminate()

	# fmins = [r['fun'] for r in res]
	# # fmins = [r[1] for r in res]
	# min_index = np.nanargmin(fmins)
	# print('fmin is ', fmins[min_index])
	# xstart = res[min_index]['x']
	# print('xstart is ', xstart)

#############################
############ OLD PLOTTING
#############################
# # PLOTTING
# mpl.plot(x_heat_flux,y_opt.T,'--.b',alpha=.25)
# mpl.plot(x_obs,y_obs_scaled,'*r')
# mpl.ylim([-.1,.6])

# X_col_names = ['dens_sc', 'vel_sc', 'temp_sc',
#                'sigk1', 'sigk2', 'sigw1', 'sigw2',
#                'beta_s', 'kap', 'a1', 'beta1r', 'beta2r']


# # soln unscaled over entire observation domain
# array([[1.15, 0.94, 1.15, 
# 	    1., 1.2, 0.7, 0.7, 
# 	    0.1024, 0.41135679, 0.38735689, 1.22172531, 1.05]])
# # soln unscaled over observation domain [.5,1.6]
# array([[1.07626664, 0.96534047, 1.15 , 
# 	    0.89227786, 0.8, 0.42324062, 1., 
# 	    0.09781106, 0.42, 0.37125262, 1.19, 1.45]])

# def plot_feature_importance(S,feature_labels,extra_line_plot=None):
#     assert isinstance(S,np.ndarray), "S must be a numpy array"
#     if S.ndim == 1:
#         ntargets = 1
#         ndim = len(S)
#         S = np.atleast_2d(S)

#     ntargets, ndim = S.shape
#     # normalize across columns (if not already)
#     S = S / np.atleast_2d(S.sum(axis=1)).T

#     # plot sobol indices as stacked bar charts

#     import matplotlib.pyplot as plt
#     import matplotlib._color_data as mcd
#     import seaborn as sns

#     xkcd_colors = []
#     xkcd = {name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS}
#     for j, n in enumerate(xkcd):
#         xkcd = mcd.XKCD_COLORS["xkcd:" + n].upper()
#         xkcd_colors.append(xkcd)

#     Ps = []
#     sns.set_palette(sns.color_palette("Paired", 12))
#     plt.figure(figsize=(20, 9))
#     bottom = np.zeros(ntargets)
#     for ii in range(ndim):
#         ptemp = plt.bar(np.arange(1, 1 + ntargets), S[:, ii], bottom=bottom, width=.25)
#         bottom = bottom + S[:, ii]  # reset bottom to new height
#         Ps.append(ptemp)
#     plt.ylabel('Sobol Scores')
#     plt.ylim([0, 1.1])
#     # plt.title('Sobol indices by pca mode')
#     # plt.xticks(t, ('PCA1','PCA2','PCA3','PCA4','PCA5','PCA6'))
#     # X_col_names = ['dens_sc', 'vel_sc', 'temp_sc',
#     #            'sigk1', 'sigk2', 'sigw1', 'sigw2',
#     #            'beta_s', 'kap', 'a1', 'beta1r', 'beta2r']
#     plt.legend(((p[0] for p in Ps)), (l for l in feature_labels),
#            fancybox=True, shadow=True,
#            loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=ndim)

#     # plot explained variance
#     if extra_line_plot is not None:
#         assert len(extra_line_plot) >= ntargets, "extra info must be the same length as the targets."
#         plt.plot(np.arange(1, 1 + ntargets), extra_line_plot[:ntargets], '--ok')
#     return plt

# plt = plot_feature_importance(HFreg2.feature_importances_(),X_col_names,pca.cumulative_error)
# plt.savefig(data_field + '_features.png')

# fig, ax = mpl.subplots(1,1,figsize=(20,9))
# x = 1.0*np.arange(len(Y_pca_q[0]))
# ax.fill_between(x=x/np.amax(x),y1=Y_pca_q[0],y2=Y_pca_q[2],color='b',alpha=.5)
# ax.plot(x/np.amax(x),Y_pca_q[1],'-w',alpha=.25)
# ax.fill_between(x=x/np.amax(x),y1=Y_pce_q[0],y2=Y_pce_q[2],color='r',alpha=.15)
# ax.plot(x/np.amax(x),Y_pce_q[1],'--r',alpha=.25)
# ax.set_title(data_field)

# fig.savefig(data_field + '_quantiles.png')
# # plt.show()
# # fig.show()





