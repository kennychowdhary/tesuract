import numpy as np 

from pce import PCEReg, PCEBuilder
from sklearn.datasets import make_regression
from sklearn import preprocessing


rn = np.random.RandomState(123)
X,y = make_regression(n_samples=2000,n_features=12,random_state=rn)

feature_scaler = preprocessing.MinMaxScaler(feature_range=(-1+1e-6,1-1e-6))
X = feature_scaler.fit_transform(X)

p = PCEReg(order=4,fit_type='LassoCV')
p.fit(X,y)

