import pypce
import unittest
import numpy as np
from sklearn.metrics import mean_squared_error

relpath = pypce.__file__[:-11] # ignore the __init__.py specification
print(relpath)

def mse(a,b):
	return mean_squared_error(a,b,squared=False)

class TestPCERegression(unittest.TestCase):
	def setUp(self):
		self.dim = 2
		self.order = 3
		self.nsamples = 30
		rn = np.random.RandomState(123)
		X = 2*rn.rand(self.nsamples,self.dim)-1
		y = 3 + 2*.5*(3*X[:,0]**2-1) + X[:,1] * .5*(3*X[:,0]**2-1)
		self.c_true = np.array([3.,0.,0.,2.,0.,0.,0.,1.,0.,0.,])
		self.X,self.y = X,y
		self.rn = rn
	def test_linear_fit(self):
		p = pypce.pcereg(self.order)
		p.fit(self.X,self.y)
		assert np.mean(np.abs(p.coef - self.c_true)) <= 1e-15, "Fit coefficient did not match truth."
	def test_multindex(self):
		p = pypce.pcereg(self.order)
		p.fit(self.X,self.y)
		M = p._M
		assert np.sum(M - p.mindex) == 0, "Multindex does not match."
	def test_custom_index(self):
		p0 = pypce.pcereg(self.order)
		p0.fit(self.X,self.y)
		p = pypce.pcereg(order=self.order,customM=p0._M)
		p.fit(self.X,self.y)
		assert np.mean(np.abs(p.coef - self.c_true)) <= 1e-15, "Fit coefficient did not match truth."
	def test_LassoCV_fit(self):
		p = pypce.pcereg(self.order,fit_type='LassoCV')
		y = self.y + .001*self.rn.rand(len(self.y)) # add noise
		p.fit(self.X,y)
		assert np.mean(np.abs(p.coef - self.c_true)) <= 2e-2, "LassoCV not accurate enough."
	def test_OmpCV_fit(self):
		p = pypce.pcereg(self.order,fit_type='OmpCV')
		y = self.y + .001*self.rn.rand(len(self.y)) # add noise
		p.fit(self.X,y)
		assert np.mean(np.abs(p.coef - self.c_true)) <= 2e-1, "OmpCV not accurate enough."
	def test_ElasticNetCV_fit(self):
		p = pypce.pcereg(self.order,fit_type='ElasticNetCV')
		p.fit(self.X,self.y)
		assert np.mean(np.abs(p.coef - self.c_true)) <= 2e-2, "ElasticNetCV error is not small enough."
	def test_predict(self):
		p = pypce.pcereg(self.order)
		p.fit(self.X,self.y)
		ypred1 = p.predict(self.X)
		ypred2 = p.polyeval(self.X)
		assert np.mean(np.abs(ypred1 - ypred2)) == 0, "Mismatch between eval and prediction."
	def test_single_predict(self):
		p = pypce.pcereg(self.order)
		p.fit(self.X,self.y)
		ypred1 = p.predict(self.X[0])
		ypred2 = p.polyeval(self.X[0])
		assert np.mean(np.abs(ypred1 - ypred2)) == 0, "Mismatch between eval and prediction."
	def test_set_custom_multiindex(self):
		M = np.array([[0,0],[2,0],[2,1]])
		p = pypce.pcereg(order=self.order,customM=M)
		p.fit(self.X,self.y)
		assert np.mean(p.coef - np.array([3.,2.,1.])) <= 1e-15, "coefficients did not converge for custom multindex."
		assert np.mean(np.abs(self.y - p.predict(self.X))) <= 1e-15, "Fit did not converge and/or predict did not match pce.eval."
	def test_grid_search_cv(self):
		X = 2*self.rn.rand(self.nsamples,self.dim)-1
		y = 3 + 2*.5*(3*X[:,0]**2-1) + X[:,1] * .5*(3*X[:,0]**2-1)
		param_grid = [
			{'order': [1,3,4], 
			'mindex_type':['total_order','hyperbolic',],
			'fit_type': ['LassoCV','linear']}
			]
		from sklearn.model_selection import GridSearchCV 
		pceCV = GridSearchCV(pypce.pcereg(), param_grid, scoring='neg_root_mean_squared_error')
		pceCV.fit(X,y)
		assert mse(self.c_true,pceCV.best_estimator_.coef) <= 5e-15


