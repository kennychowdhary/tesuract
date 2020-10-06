import pypce
import unittest
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error

relpath = pypce.__file__[:-11] # ignore the __init__.py specification
print(relpath)

def mse(a,b):
	return mean_squared_error(a,b,squared=False)

class TestPCEBuilder(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.dim=2
		self.rn = np.random.RandomState(23423)
	def test_pcebuilder_customM(self):
		p = pypce.PCEBuilder(order=4)
		p.compile(dim=self.dim)
		M = p.mindex
		p2 = pypce.PCEBuilder(customM=M)
		p2.compile(dim=self.dim)
		assert np.sum(M - p2.mindex) == 0
	def test_pcebuilder_customM_as_float(self):
		p = pypce.PCEBuilder(order=4)
		p.compile(dim=self.dim)
		M = p.mindex
		M = M.astype('float')
		p2 = pypce.PCEBuilder(customM=M)
		with warnings.catch_warnings(record=True) as w:
			p2.compile(dim=self.dim)
			assert p2.mindex.dtype == 'int', "Custom multiindex is not saved as an integer type."
	def test_fit_transform(self):
		p = pypce.PCEBuilder(order=4)
		X = 2*self.rn.rand(10,self.dim)-1
		Xhat = p.fit_transform(X)
		assert Xhat.shape[1] == p.mindex.shape[0], "Mistmatch between number of columns of Xhat (Vandermonde) and terms in multiindex."
	def test_fit_transform_w_normalized(self):
		p = pypce.PCEBuilder(order=4,normalized=True)
		X = 2*self.rn.rand(10,self.dim)-1
		Xhat = p.fit_transform(X)
		int_P0 = 2*np.mean(Xhat[:,0])
		assert int_P0 == 1, "First term should integrate to 1"
	def test_fit_transform_w_nonnormalized(self):
		p = pypce.PCEBuilder(order=4,normalized=False)
		X = 2*self.rn.rand(10,self.dim)-1
		Xhat = p.fit_transform(X)
		int_P0 = 2*np.mean(Xhat[:,0])
		assert int_P0 == 2, "First term is not normalized and should integrate to 2"
	def test_fit_transform_single_input(self):
		p = pypce.PCEBuilder(order=4)
		X = 2*self.rn.rand(1,self.dim)-1
		Xhat = p.fit_transform(X)
		Xhat = p.fit_transform(X[0])
	def test_polyeval_w_variable_coef_and_fixed_X(self):
		p = pypce.PCEBuilder(order=3)
		X = 2*self.rn.rand(30,self.dim)-1
		y = 3 + 2*.5*(3*X[:,0]**2-1) + X[:,1] * .5*(3*X[:,0]**2-1)
		c_true = np.array([3.,0.,0.,2.,0.,0.,0.,1.,0.,0.,])
		Xhat = p.fit_transform(X)
		assert Xhat.shape[1] == len(c_true), "coefficient array size is incorrect."
		ypred = p.polyeval(c=c_true)
		assert np.sum(np.abs(ypred - y)) <= 1e-15, "poly eval is not converging."
		ctest = np.zeros(len(c_true)); ctest[2] = 1.0
		ypred2 = p.polyeval(c=ctest)
		assert np.sum(np.abs(ypred2 - X[:,1])) <= 1e-15, "polyeval with fixed X is not working right."
	def test_polyeval_w_fixed_coef_and_variable_X(self):
		c_true = np.array([3.,0.,0.,2.,0.,0.,0.,1.,0.,0.,])
		X = 2*self.rn.rand(30,self.dim)-1
		y = 3 + 2*.5*(3*X[:,0]**2-1) + X[:,1] * .5*(3*X[:,0]**2-1)
		p = pypce.PCEBuilder(order=3,coef=c_true)
		ypred = p.polyeval(X)
		assert mse(ypred,y) <= 1e-15, "fixed coef and variable X does not work."
	def test_polyeval_custom_M_and_custom_coef(self):
		p0 = pypce.PCEBuilder(order=3)
		p0.compile(dim=2)
		M = p0.mindex[[0,3,7]]
		c_true = np.array([3.,0.,0.,2.,0.,0.,0.,1.,0.,0.,])
		p = pypce.PCEBuilder(customM=M,coef=c_true[[0,3,7]])
		X = 2*self.rn.rand(30,self.dim)-1
		y = 3 + 2*.5*(3*X[:,0]**2-1) + X[:,1] * .5*(3*X[:,0]**2-1)
		assert mse(p.polyeval(X),y) <= 1e-15, "polyeval not returning accurate prediction."
	def test_sobol_sensitivity_with_var_coef(self):
		p = pypce.PCEBuilder(order=3)
		p.compile(dim=2)
		coef = np.array([3.,0.,0.,2.,0.,0.,0.,1.,0.,0.,])
		s = p.computeSobol(c=coef)
		assert mse(s,[1.0, 0.07692307692307691]) == 0, "sobol indices are not right."
	def test_sobol_sensitivity_after_fit_transform(self):
		p = pypce.PCEBuilder(order=3)
		X = 2*self.rn.rand(30,self.dim)-1
		p.fit_transform(X)
		coef = np.array([3.,0.,0.,2.,0.,0.,0.,1.,0.,0.,])
		s = p.computeSobol(c=coef)
		assert mse(s,[1.0, 0.07692307692307691]) == 0, "sobol indices are not right."
	# def test_sobol_sensitivity_customM_and_coef_constructor(self):
	# 	p0 = pypce.PCEBuilder(order=3)
	# 	p0.compile(dim=2)
	# 	M = p0.mindex[[0,3,7]]
	# 	coef = np.array([3.,2.,1.])
	# 	p = pypce.PCEBuilder(customM=M,coef=coef)
	# 	# p.compile(dim=2)
	# 	s = p.computeSobol()
	# 	assert mse(s,[1.0, 0.07692307692307691]) == 0, "sobol indices are not right."
	def test_sobol_sensitivity_customM_and_coef_constructor(self):
		p0 = pypce.PCEBuilder(order=3)
		p0.compile(dim=2)
		M = p0.mindex[[0,3,7]]
		coef = np.array([3.,2.,1.])
		p = pypce.PCEBuilder(customM=M,coef=coef)
		# p.compile(dim=2)
		s = p.computeSobol()
		assert mse(s,[1.0, 0.07692307692307691]) == 0, "sobol indices are not right."
	def test_sobol_sensitivity_w_normalization(self):
		p0 = pypce.PCEBuilder(order=3)
		p0.compile(dim=2)
		M = p0.mindex[[0,3,7]]
		coef = np.array([3.,2.,1.])
		p = pypce.PCEBuilder(customM=M,coef=coef,normalized=True)
		# p.compile(dim=2)
		s = p.computeSobol()
		print(s,p.mindex)
		assert mse(s,[1.0, 0.2]) == 0, "sobol indices for normalized case is  not right."



