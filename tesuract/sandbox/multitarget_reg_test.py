import numpy as np 
import tesuract
from sklearn.datasets import make_friedman1
import time as T

# generate test data
rn = np.random.RandomState(44)
dim = 10
X,y = make_friedman1(n_samples=100,n_features=dim,random_state=rn)


# transform data to [-1,1] for polynomials
dScaler = tesuract.preprocessing.DomainScaler(dim=dim,input_range=(0,1))
Xs = dScaler.fit_transform(X)

# fit using PCE CV wrapper
pce_param_grid = \
	{'order': [1,2,3,4],
	 'mindex_type': ['total_order', 'hyperbolic', ],
	 'fit_type': ['LassoCV', 'linear', 'ElasticNetCV']}

rf_param_grid = \
	{'n_estimators': [100,300,750],
	 'max_features': ['sqrt','auto'],
	 'max_depth': [5,10,25]}

rf_param_grid = \
	{'n_estimators': [100,300,750],
	 'max_features': ['sqrt','auto'],
	 'max_depth': [5,10,25]}

svr_param_grid = {
	'kernel': ('linear','poly', 'rbf', 'sigmoid'),
	'degree': (2,4,8),
	'gamma': ('scale','auto'),
	'C': (1,5,10)
} 

# start = T.time()
# pceCV = tesuract.RegressionWrapperCV(regressor='pce',reg_params=pce_param_grid,n_jobs=-1)
# pceCV.fit(Xs,y)
# end = T.time()
# print("best pce error: {0:.5f}, {1:.5f} sec".format(-pceCV.best_score_,end-start))

# PCA test
dim = 250
samples = 500
mean = np.zeros(dim)
t = np.linspace(0,1,dim)
tx,ty = np.meshgrid(t,t)
C = np.array([min(pair) for pair in zip(tx.reshape(-1),ty.reshape(-1))]).reshape(dim,dim)
Y = rn.multivariate_normal(mean,C,size=(samples,))

from sklearn.decomposition import PCA
# pca = PCA(n_components='mle',whiten=True)
pca = tesuract.preprocessing.PCATargetTransform(n_components='auto',cutoff=.05,whiten=False)
Yhat = pca.fit_transform(Y)
components = pca.components_
print(pca.n_components_)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
ysamples = [yhat for yhat in Yhat.T]
# for yi in ysamples: ax.hist(yi,bins=30)

# # try with repeated k fold cv
# from sklearn.model_selection import RepeatedKFold
# rkcv = RepeatedKFold(n_splits=5,n_repeats=10)
# start = T.time()
# pceCV2 = tesuract.RegressionWrapperCV(regressor='pce',reg_params=pce_param_grid,cv=rkcv,n_jobs=-1)
# pceCV2.fit(Xs,y)
# end = T.time()
# print("best pce error: {0:.5f}, {1:.5f} sec".format(-pceCV2.best_score_,end-start))

# rfCV = tesuract.RegressionWrapperCV(regressor='rf',reg_params=rf_param_grid,n_jobs=-1)
# rfCV.fit(Xs,y)
# print("best rf score:", rfCV.best_score_)

# svrCV = tesuract.RegressionWrapperCV(regressor='svr',reg_params=svr_param_grid,n_jobs=-1)
# svrCV.fit(Xs,y)
# print("best svr score:", svrCV.best_score_)

# multimodelCV = tesuract.RegressionWrapperCV(regressor=['pce','rf','svr'],reg_params=[pce_param_grid,rf_param_grid,svr_param_grid],n_jobs=-1)
# multimodelCV.fit(Xs,y)
# print("best pce/rf score:", multimodelCV.best_score_)

