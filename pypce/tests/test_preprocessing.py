import pypce
import unittest
import numpy as np
from pypce.preprocessing import DomainScaler_old, MinMaxTargetScaler

relpath = pypce.__file__[:-11] # ignore the __init__.py specification
print(relpath)

class TestDomainScaler_old(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.rn = np.random.RandomState(3123)
	def test_scale_features_uniform_ab(self):
		X = 3.5*self.rn.rand(30,3) + 2
		a = np.array([2,2,2.])
		b = 3.5+a
		scaler = DomainScaler_old(a=a,b=b)
		Xhat = scaler.fit_transform(X)
		assert np.amin(Xhat) >= -1.0 and np.amax(Xhat) <= 1.0, "Scaler is not working properly"
	def test_scale_features_nonuniform_ab(self):
		a = np.array([2,3,0.])
		b = 3.5+a
		X = (b-a)*self.rn.rand(30,3) + a
		scaler = DomainScaler_old(a=a,b=b)
		Xhat = scaler.fit_transform(X)
		assert np.amin(Xhat) >= -1.0 and np.amax(Xhat) <= 1.0, "Scaler is not working properly"
	def test_scale_features_inverse_nonuniform_ab(self):
		a = np.array([2,3,0.])
		b = np.array([2.5,1.,1.])+a
		X = (b-a)*self.rn.rand(30,3) + a
		scaler = DomainScaler_old(a=a,b=b)
		Xhat = scaler.fit_transform(X)
		assert np.sum(scaler.inverse_transform(Xhat) - X) == 0, "Inverse transform not working properly."
	def test_scale_features_inverse_uniform_ab(self):
		X = 3.5*self.rn.rand(30,3) + 2
		a = np.array([2,2,2.])
		b = 3.5+np.array([2,2,2.])
		scaler = DomainScaler_old(a=a,b=b)
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
