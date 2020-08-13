import pypce
import unittest
import numpy as np
from pypce.preprocessing import DomainScaler, MinMaxTargetScaler

relpath = pypce.__file__[:-11] # ignore the __init__.py specification
print(relpath)

class TestDomainScaler(unittest.TestCase):
	def setUp(self):
		self.rn = np.random.RandomState(3123)
	def test_scale_features_uniform_ab(self):
		X = 3.5*self.rn.rand(30,3) + 2
		a = np.array([2,2,2.])
		b = 3.5+a
		scaler = DomainScaler(a=a,b=b)
		Xhat = scaler.fit_transform(X)
		assert np.amin(Xhat) >= -1.0 and np.amax(Xhat) <= 1.0, "Scaler is not working properly"
	def test_scale_features_nonuniform_ab(self):
		a = np.array([2,3,0.])
		b = 3.5+a
		X = (b-a)*self.rn.rand(30,3) + a
		scaler = DomainScaler(a=a,b=b)
		Xhat = scaler.fit_transform(X)
		assert np.amin(Xhat) >= -1.0 and np.amax(Xhat) <= 1.0, "Scaler is not working properly"
	def test_scale_features_inverse_nonuniform_ab(self):
		a = np.array([2,3,0.])
		b = np.array([2.5,1.,1.])+a
		X = (b-a)*self.rn.rand(30,3) + a
		scaler = DomainScaler(a=a,b=b)
		Xhat = scaler.fit_transform(X)
		assert np.sum(scaler.inverse_transform(Xhat) - X) == 0, "Inverse transform not working properly."
	def test_scale_features_inverse_uniform_ab(self):
		X = 3.5*self.rn.rand(30,3) + 2
		a = np.array([2,2,2.])
		b = 3.5+np.array([2,2,2.])
		scaler = DomainScaler(a=a,b=b)
		Xhat = scaler.fit_transform(X)
		assert np.sum(scaler.inverse_transform(Xhat) - X) == 0, "Inverse transform not working properly."
	def test_target_scaler(self):
		Y = 3.5*self.rn.rand(10,100) + 2
		scaler = MinMaxTargetScaler()
		scaler.fit(Y)
		Yhat = scaler.fit_transform(Y)
		assert np.amin(Yhat) == 0.0 and np.amax(Yhat) == 1.0, "Scaler is not working properly"
	def test_target_scaler_inverse(self):
		Y = 3.5*self.rn.rand(10,100) + 2
		scaler = MinMaxTargetScaler()
		scaler.fit(Y)
		Yhat = scaler.fit_transform(Y)
		assert np.sum(scaler.inverse_transform(Yhat) - Y) <= 1e-10, "Inverse transform not working properly."



class TestTotalOrderMultiIndex(unittest.TestCase):
	def setUp(self):
		self.dim = 3
		self.order = 5
		self.M = pypce.MultiIndex(dim=self.dim,order=self.order)
		self.mtest = np.loadtxt(relpath + '/tests/data/multindex_d3_o5.txt')
	def test_multindex(self):
		abserr = np.sum(self.M.index - self.mtest)
		assert abserr == 0, "multindex for d=3,o=5 is not right."
	def test_verify_max_order(self):
		self.assertTrue(np.all(np.sum(self.M.index,1) <= self.order))
	def test_multiindex_length_and_dim(self):
		factorial = np.math.factorial
		nBasis = factorial(self.order+self.dim)/factorial(self.order)/factorial(self.dim)
		self.assertEqual(self.M.index.shape[0],nBasis)
		self.assertEqual(self.M.index.shape[1],self.dim)
	def test_nPCTerms(self):
		self.assertEqual(self.M.index.shape[0],self.M.nPCTerms)
	def test_set_index(self):
		nPCTerms_old = self.M.nPCTerms
		test_index = self.M.index[:-1]
		self.M.setIndex(test_index)
		self.assertEqual(self.M.nPCTerms,nPCTerms_old-1)

class TestHyperbolicMultiIndex(unittest.TestCase):
	def setUp(self):
		self.dim = 3
		self.order = 4
		self.M = pypce.MultiIndex(dim=self.dim,order=self.order,mindex_type='hyperbolic')
		self.mtest = np.loadtxt(relpath + '/tests/data/hypmultindex_d3_o4.txt')
	def test_hyperbolic_multindex(self):
		abserr = np.sum(self.M.index - self.mtest)
		assert abserr == 0, "hyperbolic multindex for d=3,o=4 is not right."
	def test_verify_max_order(self):
		self.assertTrue(np.all(np.sum(self.M.index,1) <= self.order))
	def test_multiindex_length_and_dim(self):
		self.assertEqual(self.M.index.shape[0],16)
		self.assertEqual(self.M.index.shape[1],self.dim)
	def test_nPCTerms(self):
		self.assertEqual(self.M.index.shape[0],self.M.nPCTerms)