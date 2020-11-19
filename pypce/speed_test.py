import numpy as np 

from pce import PCEReg, PCEBuilder
from sklearn.datasets import make_regression
from sklearn import preprocessing

from pceregression import *

# Single target, single PCE test
rn = np.random.RandomState(123)
X,y = make_regression(n_samples=2000,n_features=12,random_state=rn)

feature_scaler = preprocessing.MinMaxScaler(feature_range=(-1+1e-6,1-1e-6))
X = feature_scaler.fit_transform(X)

p = PCEReg(order=4,fit_type='LassoCV')
p.fit(X,y)

# def foo(repeat=10):
# 	for i in range(repeat):
# 		p.predict(X)

# import cProfile
# cProfile.run('foo()')


# Multi target with PCA 
X2,Y2 = make_regression(n_samples=2000,n_features=12,n_targets=100,random_state=rn)
X2 = feature_scaler.fit_transform(X2)

# pce grid search cv test
pce_param_grid = [{'order': [4],
	 'mindex_type': ['total_order'],
	 'fit_type': ['LassoCV']}]

pca_params={'n_components':8}

regmodel = MRegressionWrapperCV(
			regressor=['pce'],
			reg_params=[pce_param_grid],
			target_transform=PCATargetTransform,
			target_transform_params=pca_params,
			n_jobs=8)
regmodel.fit(X2,Y2)