import tesuract
import unittest
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error
import os
from joblib import dump, load
import time

relpath = tesuract.__file__[:-11] # ignore the __init__.py specification
print("relpath:",relpath)

def mse(a,b):
	return mean_squared_error(a,b,squared=False)

class TestPCEBuilderOnArbDomain(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.n, self.d = 7,3
		self.rn = np.random.RandomState(23)
		self.X = 5*self.rn.rand(self.n,self.d)
		self.ibounds = [(0,5) for r in range(self.d)]
	def test_full_domain_scaling(self):
		n, d, X, ibounds = self.n, self.d, self.X, self.ibounds
		scaler = tesuract.preprocessing.DomainScaler(dim=d,input_range=ibounds,output_range=(-1,1))
		
		# print(X)
		X_scaled = scaler.fit_transform(X)
		pce = tesuract.PCEBuilder(order=4)
		Xhat_ref = pce.fit_transform(X_scaled)
		# X_scaled = scaler.transform(X)
		# print(X_scaled)
		scaler._range_check(X_scaled,[(-1,1) for r in range(d)])
		scaler._range_check(X,[(0,5) for r in range(d)])
		return Xhat_ref
	def test_pcebuild_raise_bound_assertion(self):
		n, d, X, ibounds = self.n, self.d, self.X, self.ibounds
		pce = tesuract.PCEBuilder(order=4)
		self.assertRaises(AssertionError, pce.fit_transform, X)
	def test_pcebuild_with_input_range(self):
		n, d, X, ibounds = self.n, self.d, self.X, self.ibounds
		pce = tesuract.PCEBuilder(order=4,input_range=ibounds)
		Xhat = pce.fit_transform(X)
		Xhat_ref = self.test_full_domain_scaling()
		assert mse(Xhat_ref,Xhat) <= 1e-16, "mismatch between expected poly transform and actual poly transform"
	def test_pcebuild_polyeval_with_input_range(self):
		n, d, X, ibounds = self.n, self.d, self.X, self.ibounds
		pce = tesuract.PCEBuilder(order=4,input_range=ibounds)
		pce.compile(X.shape[1])
		coef = np.zeros(35)
		coef[0] = 1.0
		ypred = pce.polyeval(X,coef)
		assert np.all(ypred == 1), "vector must be all ones. Something may have gone wrong with polyeval and input bounds."
	def test_pcebuild_polyeval_assert_error_wo_input_range(self):
		n, d, X, ibounds = self.n, self.d, self.X, self.ibounds
		pce = tesuract.PCEBuilder(order=4)
		pce.compile(X.shape[1])
		coef = np.zeros(35)
		coef[0] = 1.0
		self.assertRaises(AssertionError, pce.polyeval, X, coef)
		


class TestPCEBuilder(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.dim=2
		self.rn = np.random.RandomState(23423)
	def test_pcebuilder_customM(self):
		p = tesuract.PCEBuilder(order=4)
		p.compile(dim=self.dim)
		M = p.mindex
		p2 = tesuract.PCEBuilder(customM=M)
		p2.compile(dim=self.dim)
		assert np.sum(M - p2.mindex) == 0
	def test_pcebuilder_customM_as_float(self):
		p = tesuract.PCEBuilder(order=4)
		p.compile(dim=self.dim)
		M = p.mindex
		M = M.astype('float')
		p2 = tesuract.PCEBuilder(customM=M)
		with warnings.catch_warnings(record=True) as w:
			p2.compile(dim=self.dim)
			assert p2.mindex.dtype == 'int', "Custom multiindex is not saved as an integer type."
	def test_fit_transform(self):
		p = tesuract.PCEBuilder(order=4)
		X = 2*self.rn.rand(10,self.dim)-1
		Xhat = p.fit_transform(X)
		assert Xhat.shape[1] == p.mindex.shape[0], "Mistmatch between number of columns of Xhat (Vandermonde) and terms in multiindex."
	def test_fit_transform_w_normalized(self):
		p = tesuract.PCEBuilder(order=4,normalized=True)
		X = 2*self.rn.rand(10,self.dim)-1
		Xhat = p.fit_transform(X)
		int_P0 = 2*np.mean(Xhat[:,0])
		assert np.abs(int_P0-1.0) <= 1e-15, "First term should integrate to 1"
	def test_fit_transform_w_nonnormalized(self):
		p = tesuract.PCEBuilder(order=4,normalized=False)
		X = 2*self.rn.rand(10,self.dim)-1
		Xhat = p.fit_transform(X)
		int_P0 = 2*np.mean(Xhat[:,0])
		assert int_P0 == 2, "First term is not normalized and should integrate to 2"
	def test_fit_transform_single_input(self):
		p = tesuract.PCEBuilder(order=4)
		X = 2*self.rn.rand(1,self.dim)-1
		Xhat = p.fit_transform(X)
		Xhat = p.fit_transform(X[0])
	def test_pcebuilder_poleval_w_custom_coef(self):
		# define the mean solution as a PCE
	    coef = np.array([0.,-.25,1./3,np.sqrt(2)])#,-.05*np.sqrt(2)])
	    pce = tesuract.PCEBuilder(order=3)
	    pce.compile(dim=1) # must compile to compute the multiindex
	    polyfun = lambda x: pce.polyeval(x,c=coef)
	    x = np.linspace(-1,1,10)
	    y = polyfun(x)
	def test_polyeval_w_variable_coef_and_fixed_X(self):
		p = tesuract.PCEBuilder(order=3,store_phi=True)
		X = 2*self.rn.rand(30,self.dim)-1
		y = 3 + 2*.5*(3*X[:,0]**2-1) + X[:,1] * .5*(3*X[:,0]**2-1)
		c_true = np.array([3.,0.,0.,2.,0.,0.,0.,1.,0.,0.,])
		Xhat = p.fit_transform(X)
		assert Xhat.shape[1] == len(c_true), "coefficient array size is incorrect."
		ypred = p.polyeval(c=c_true)
		assert np.sum(np.abs(ypred - y)) <= 1e-14, "poly eval is not converging."
		ctest = np.zeros(len(c_true)); ctest[2] = 1.0
		ypred2 = p.polyeval(c=ctest)
		assert np.sum(np.abs(ypred2 - X[:,1])) <= 1e-15, "polyeval with fixed X is not working right."
	def test_polyeval_w_fixed_coef_and_variable_X(self):
		c_true = np.array([3.,0.,0.,2.,0.,0.,0.,1.,0.,0.,])
		X = 2*self.rn.rand(30,self.dim)-1
		y = 3 + 2*.5*(3*X[:,0]**2-1) + X[:,1] * .5*(3*X[:,0]**2-1)
		p = tesuract.PCEBuilder(order=3,coef=c_true)
		ypred = p.polyeval(X)
		assert mse(ypred,y) <= 1e-15, "fixed coef and variable X does not work."
	def test_polyeval_custom_M_and_custom_coef(self):
		p0 = tesuract.PCEBuilder(order=3)
		p0.compile(dim=2)
		M = p0.mindex[[0,3,7]]
		c_true = np.array([3.,0.,0.,2.,0.,0.,0.,1.,0.,0.,])
		p = tesuract.PCEBuilder(customM=M,coef=c_true[[0,3,7]])
		X = 2*self.rn.rand(30,self.dim)-1
		y = 3 + 2*.5*(3*X[:,0]**2-1) + X[:,1] * .5*(3*X[:,0]**2-1)
		assert mse(p.polyeval(X),y) <= 1e-15, "polyeval not returning accurate prediction."
	def test_sobol_sensitivity_with_var_coef(self):
		p = tesuract.PCEBuilder(order=3)
		p.compile(dim=2)
		coef = np.array([3.,0.,0.,2.,0.,0.,0.,1.,0.,0.,])
		s = p.computeSobol(c=coef)
		assert mse(s,[1.0, 0.07692307692307691]) == 0, "sobol indices are not right."
	def test_sobol_sensitivity_after_fit_transform(self):
		p = tesuract.PCEBuilder(order=3)
		X = 2*self.rn.rand(30,self.dim)-1
		p.fit_transform(X)
		coef = np.array([3.,0.,0.,2.,0.,0.,0.,1.,0.,0.,])
		s = p.computeSobol(c=coef)
		assert mse(s,[1.0, 0.07692307692307691]) == 0, "sobol indices are not right."
	# def test_sobol_sensitivity_customM_and_coef_constructor(self):
	# 	p0 = tesuract.PCEBuilder(order=3)
	# 	p0.compile(dim=2)
	# 	M = p0.mindex[[0,3,7]]
	# 	coef = np.array([3.,2.,1.])
	# 	p = tesuract.PCEBuilder(customM=M,coef=coef)
	# 	# p.compile(dim=2)
	# 	s = p.computeSobol()
	# 	assert mse(s,[1.0, 0.07692307692307691]) == 0, "sobol indices are not right."
	def test_sobol_sensitivity_customM_and_coef_constructor(self):
		p0 = tesuract.PCEBuilder(order=3)
		p0.compile(dim=2)
		M = p0.mindex[[0,3,7]]
		coef = np.array([3.,2.,1.])
		p = tesuract.PCEBuilder(customM=M,coef=coef)
		# p.compile(dim=2)
		s = p.computeSobol()
		assert mse(s,[1.0, 0.07692307692307691]) == 0, "sobol indices are not right."
	def test_sobol_sensitivity_w_normalization(self):
		p0 = tesuract.PCEBuilder(order=3)
		p0.compile(dim=2)
		M = p0.mindex[[0,3,7]]
		coef = np.array([3.,2.,1.])
		p = tesuract.PCEBuilder(customM=M,coef=coef,normalized=True)
		# p.compile(dim=2)
		s = p.computeSobol()
		print(s,p.mindex)
		assert mse(s,[1.0, 0.2]) == 0, "sobol indices for normalized case is  not right."
	def test_1d_poly_feature_transform(self):
		N = 32 # number of x grid points
		x = np.linspace(-1,1,N)[:,np.newaxis]
		rn = np.random.RandomState(23948)
		x_random = 2*rn.rand(N)[:,np.newaxis]-1
		x_chebychev = -np.cos(np.pi*x)

		# define test functions
		runge = lambda x: 1./(1 + 16.*x**2)
		gibbs = lambda x: np.sign(np.sin(np.pi*(x-np.pi/2)))

		# get polynomial features
		from sklearn.preprocessing import PolynomialFeatures
		P = 8
		polyfeatures = PolynomialFeatures(degree=P)
		X = polyfeatures.fit_transform(x)

		# Legendre features
		pce = tesuract.PCEBuilder(order=P)
		Xleg = pce.fit_transform(x)
		# print(Xleg.shape, Xleg)
	
	def test_memory_footprint_w_attr_assert(self):
		N = 2500 # number of x grid points
		rn = np.random.RandomState(9)
		X = 2*rn.rand(N,5)-1.0

		# # define test functions
		# runge = lambda x: 1./(1 + 16.*np.sum(x,axis=1)**2)
		# gibbs = lambda x: np.sign(np.sin(np.pi*(np.sum(x,axis=1)-np.pi/2)))

		pceb = tesuract.PCEBuilder(order=4) #store_phi=False default
		pceb.fit_transform(X)
		self.assertRaises(AttributeError, getattr, pceb, "Phi")

		pceb = tesuract.PCEBuilder(order=4,store_phi=True)
		pceb.fit_transform(X)
		assert pceb.Phi.shape[0] == N, "Phi is kept but is NOT the right shape. "

	def test_timing_for_repeated_transform(self):
		N = 2500 # number of x grid points
		rn = np.random.RandomState(9)
		X = 2*rn.rand(N,5)-1.0

		pceb = tesuract.PCEBuilder(order=4) #store_phi=False default
		start1 = time.time()
		pceb.fit_transform(X)
		end1 = time.time() - start1
		start2 = time.time()
		pceb.fit_transform(X)
		end2 = time.time() - start2
		assert end2 < end1, "second fit transform should be faster since mindex is already computed."
		assert pceb._mindex_compute_count_ == 1, "mindex is being computed more than once or none at all."


# code to test the joint Sobol function in PCEBuilder
class TestJointSobol(unittest.TestCase):
	def test_initialize_joint_sensitivity(self):
		p = tesuract.PCEBuilder(order=2)
		dim = 3
		p.compile(dim=dim)
		print(p.mindex)
		nterms, ndim = p.mindex.shape
		dummy_coef = np.ones(nterms)
		assert ndim == dim, "dimensions don't match"
		S2 = p.computeJointSobol(c=dummy_coef)
		print("\n",S2)
	def test_joint_sensitivity_2D_3rd(self):
		pce = tesuract.PCEReg(order=3)
		rn  = np.random.RandomState(123)
		X   = 2*rn.rand(30,2)-1
		y   = 3 + 2*.5*(3*X[:,0]**2-1) + X[:,1] * .5*(3*X[:,0]**2-1)
		pce.fit(X,y)
		coef_sum  = np.sum([pce.normsq[i]*pce.coef[i]**2 for i in range(len(pce.coef)) if ( (pce.mindex[i,0]>0) and (pce.mindex[i,1]>0) )])
		index     = pce.joint_effects()[1,0]
		var_temp  = coef_sum / index
		var_tot   = np.sum([pce.normsq[i]*pce.coef[i]**2 for i in range(len(pce.coef)) if (np.sum(pce.mindex[i,:])>0)])
		assert var_temp == var_tot, "dimensions don't match"
