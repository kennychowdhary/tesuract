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
from sklearn.decomposition import PCA, FastICA
		
from pceregression import MRegressionWrapperCV, RegressionWrapperCV, PCATargetTransform

# pce grid search cv test
pce_param_grid = [{'order': [1, 2, 3],
	 'mindex_type': ['total_order', 'hyperbolic', ],
	 'fit_type': ['LassoCV', 'ElasticNetCV']}]

# random forest fit
rf_param_grid = {'n_estimators': [100,300],
				 'max_features': ['auto'],
				 'max_depth': [5,10]
				 }

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
	obs = np.loadtxt(datadir + 'data/mks_hifire1_run30_q.dat')
	xobs,yobs = obs.T
else:
	x_domain = np.load(datadir + 'data/x_df_domain.npy')
	X = np.load(datadir + 'data/LHS_df_samples.npy')
Y = np.load(datadir + 'data/Y_' + data_field + '.npy')


Q = [.01,.5,.99]

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

# pca_params={'n_components':'auto','cutoff':1e-2}
pca_params={'n_components':8}
# pca = FastICA(**pca_params)
# pca = PCA(**pca_params)
pca = PCATargetTransform(**pca_params)
pca.fit(Y_scaled)
Yhat_scaled = pca.transform(Y_scaled)

print('n components = ', Yhat_scaled.shape[1])
Y_pca_recon0 = pca.inverse_transform(Yhat_scaled)
Y_pca_q = np.quantile(Y_pca_recon0,Q,axis=0)

# # compute cv score for single target and list of regressors
# regmodel1 = RegressionWrapperCV(
# 		regressor=['pce','randforests'],
# 		reg_params=[pce_param_grid,rf_param_grid],
# 		n_jobs=8)

# # plot best fit lines
# for i in range(12):
#     # plot(X_scaled[:,i],Yhat_scaled[:,0],'.',alpha=.1)
#     m,b,_,_,_ = linregress(X_scaled[:,i],Yhat_scaled[:,2])
#     x = linspace(-1,1,1000)
#     plot(x,m*x+b,'--',alpha=.7)

# fit multiple target with type as single element list
start = T.time()
regmodel2 = MRegressionWrapperCV(
			regressor=['pce'],
			reg_params=[pce_param_grid],
			target_transform=PCATargetTransform,
			target_transform_params=pca_params,
			n_jobs=8)
# regmodel2.fit(X_scaled,Y_scaled)
end = T.time()
print("Total time is %.5f seconds" %(end-start))
regmodel = regmodel2
# print(regmodel.best_params_)

# # load model
from joblib import dump, load
# dump(regmodel,'regmodel_test.joblib')
# regmodel = load(datadir + '/heat_flux_pce_model_n2500.joblib')
regmodel = load('regmodel_test.joblib')
print(regmodel.best_params_)

# get quantiles
rn = np.random.RandomState(2383)
X_test = 2*rn.rand(10000,12)-1
Y_pce_recon2 = regmodel.predict(X_test)
Y_pce_q = np.quantile(Y_pce_recon2,Q,axis=0)
# print(HFreg2.best_scores_)


#######################
## Save and calibrate
#######################
# def calibrate():

# save model
# regmodel = HFreg2

# load observations for heat flux only
# interpolate predictions for obersvations
# if data_field == 'heat_flux_wall':
x_heat_flux = x_domain #np.load(datadir + 'data/x_heat_flux.npy')
# obs = np.loadtxt(datadir + 'data/mks_hifire1_run30_q.dat')
# xobs, y_obs = obs.T
# # transform y_obs using 
yobs_scaled = target_scaler.transform(yobs)
obs_cutoff = 1.5

nv = [1.0,1.0,1.0,0.85,1.0,0.5,0.856,0.09,0.41,.31,1.20,1.0870]
nv = dict(zip(X_col_names,nv))
LB = np.array([.85, .85, .85, 0.7, 0.8, 0.3, 0.7, 0.0784, 0.38, 0.31, 1.19, 1.05])
UB = np.array([1.15, 1.15, 1.15, 1.0, 1.2, 0.7, 1.0, 0.1024, 0.42, 0.40, 1.31, 1.45])
LB[0],UB[0] = (.98*nv['dens_sc'],1.02*nv['dens_sc'])	# dens
LB[1],UB[1] = (.98*nv['vel_sc'],1.02*nv['vel_sc']) 		# vel	
LB[-3],UB[-3] = (.31,.4) 								# a1
UB_scaled = feature_scaler.fit_transform(UB)[0]
LB_scaled = feature_scaler.fit_transform(LB)[0]

# interpolate
from scipy.interpolate import interp1d
def interp_heat_flux(y):
	# yi = np.interp(xobs,x_heat_flux,y)
	fi = interp1d(x_heat_flux,y)
	yi = fi(xobs)
	return yi

def objfun(x_uq):
	y = regmodel.predict(x_uq)
	yi = interp_heat_flux(y)
	yi = np.atleast_2d(yi)
	y_true = np.atleast_2d(yobs_scaled)
	ii = (xobs >= 0) & (xobs <= obs_cutoff)
	error = np.mean((y_true[:,ii] - yi[:,ii])**2, axis=1)/np.mean(y_true[:,ii]**2)
	return error, yi


def func(x):
	dim = 12
	xref = np.zeros(dim)
	# xref[[0,1,9]] = x
	xref[:] = x
	error, yi = objfun(xref)
	return error

# Needs to be reformulated to be in the correct range
dim = len(X_col_names)
opt_dim = 12
X0 = []
x0 = np.zeros(opt_dim) #X_scaled[0]
X0.append(x0)
# for i in range(100):
# 	X0.append(np.random.rand(opt_dim))
# xref = np.zeros(dim)
# error0, ypred0 = objfun(xref)

# optimize
from scipy.optimize import fmin_l_bfgs_b, fmin_tnc, fmin_slsqp
dim = X_scaled.shape[1]
bounds = [(LB_scaled[i],UB_scaled[i]) for i,d in enumerate(range(opt_dim))]

Nfeval = 1
def minimize_call(xstart,factr=1e2):
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
myfun = partial(minimize_call, factr=1e7)
from multiprocessing import Pool
nprocs = 1
p = Pool(nprocs) 
start = T.time()
res = p.map(myfun,[x for x in X0])
end = T.time()
p.close()
p.terminate()

# fmins = [r['fun'] for r in res]
# # fmins = [r[1] for r in res]
# min_index = np.nanargmin(fmins)
# print('fmin is ', fmins[min_index])
# xstart = res[min_index]['x']
# print('xstart is ', xstart)

xopt_scaled = res[0][0]
# xopt_scaled = -1 + 0*x0
xopt = feature_scaler.inverse_transform(xopt_scaled)
xopt_dict = dict(zip(X_col_names,xopt[0]))

yopt_scaled = regmodel.predict(xopt_scaled)
yopt = target_scaler.inverse_transform(yopt_scaled)


#######################
# PLOT
######################
# plot

QY_pca = target_scaler.inverse_transform(Y_pca_q)
QY_pce = target_scaler.inverse_transform(Y_pce_q)

fig = mpl.figure()
ax = fig.add_subplot(111)
ax.fill_between
ax.fill_between(x_domain,QY_pca[0],QY_pca[2],alpha=.15,label='1-50-99% PCA')
ax.plot(x_domain,QY_pca[1],'--b',alpha=.35)
# ax.fill_between(x_domain,QY_pce[0],QY_pce[2],color='g',alpha=.05,label='2.5-50-97.5% PCE')
ax.plot(x_domain,QY_pce[1],'--g',alpha=.35)
ax.plot(xobs,yobs,'.r',alpha=.6,label='Observations')
ax.plot(x_domain,yopt.T,'-k',alpha=.5,label='opt')


# add details to plot
ax.set_yscale("log")
ax.set_xlabel("x")
ax.set_ylabel("heat flux")
ax.grid(True,which='both',alpha=.5)
ax.legend(fancybox=True, framealpha=0.25)
ax.set_title('a1 = ' + str(xopt_dict['a1']) + "; cutoff = " + str(obs_cutoff))




