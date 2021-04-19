import tesuract
import unittest
import numpy as np
from tesuract.preprocessing import DomainScaler, MinMaxTargetScaler

relpath = tesuract.__file__[:-11] # ignore the __init__.py specification
print(relpath)


class TestTotalOrderMultiIndex(unittest.TestCase):
	def setUp(self):
		self.dim = 3
		self.order = 5
		self.M = tesuract.MultiIndex(dim=self.dim,order=self.order)
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
		self.M = tesuract.MultiIndex(dim=self.dim,order=self.order,mindex_type='hyperbolic')
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