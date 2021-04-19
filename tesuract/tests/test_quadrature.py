import tesuract
import unittest
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error

relpath = tesuract.__file__[:-11] # ignore the __init__.py specification
print(relpath)

def mse(a,b):
	return mean_squared_error(a,b,squared=False)

# Genz function tests
def genz_exp(x):
    if x.ndim == 2:
        output = np.exp(np.sum(x - 1,axis=1))
    elif x.ndim == 1:
        output = np.exp(np.sum(x - 1))
    return output
genz_exp_soln = lambda d : (np.exp(0) - np.exp(-1.0))**d

class TestQuad(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.Q_d5_o5_LG_sp = tesuract.QuadGen(dim=5,order=5,rule='legendre_gauss',sparse=True)
        self.xw_d5_o5_LG_sp = self.Q_d5_o5_LG_sp.run()
        self.Q_d5_o5_LG_tp = tesuract.QuadGen(dim=5,order=5,rule='legendre_gauss',sparse=False)
        self.xw_d5_o5_LG_tp = self.Q_d5_o5_LG_tp.run()

        self.Q_d10_o2_LG_sp = tesuract.QuadGen(dim=10,order=2,rule='legendre_gauss',sparse=True)
        self.xw_d10_o2_LG_sp = self.Q_d10_o2_LG_sp.run()
        self.Q_d10_o2_LG_tp = tesuract.QuadGen(dim=10,order=2,rule='legendre_gauss',sparse=False)
        self.xw_d10_o2_LG_tp = self.Q_d10_o2_LG_tp.run()

        self.Q_d5_o5_CC_sp = tesuract.QuadGen(dim=5,order=5,rule='clenshaw_curtis',sparse=True)
        self.xw_d5_o5_CC_sp = self.Q_d5_o5_CC_sp.run()
        self.Q_d5_o5_CC_tp = tesuract.QuadGen(dim=5,order=5,rule='clenshaw_curtis',sparse=False)
        self.xw_d5_o5_CC_tp = self.Q_d5_o5_CC_sp.run()

        self.Q_d10_o2_CC_sp = tesuract.QuadGen(dim=10,order=2,rule='clenshaw_curtis',sparse=True)
        self.xw_d10_o2_CC_sp = self.Q_d10_o2_CC_sp.run()
        self.Q_d10_o2_CC_tp = tesuract.QuadGen(dim=10,order=2,rule='clenshaw_curtis',sparse=False)
        self.xw_d10_o2_CC_tp = self.Q_d10_o2_CC_tp.run()

    def test_sparse_quadrature_dim5_order5_LegGauss(self): 
        # test...
        dim = 5
        # Q = tesuract.QuadGen(dim=dim,order=5,rule='legendre_gauss',sparse=True)
        x,w = self.xw_d5_o5_LG_sp
        approx = np.sum(genz_exp(x)*w)
        soln = genz_exp_soln(dim)
        error = np.mean((soln - approx)**2)/np.mean(soln**2)
        assert error <= 1e-15, "Quadrature error is too high. "
    def test_tensor_prod_quadrature_dim5_order5_LegGauss(self):
        # test...
        dim = 5
        # Q = tesuract.QuadGen(dim=dim,order=5,rule='legendre_gauss',sparse=False)
        x,w = self.xw_d5_o5_LG_tp
        approx = np.sum(genz_exp(x)*w)
        soln = genz_exp_soln(dim)
        error = np.mean((soln - approx)**2)/np.mean(soln**2)
        # print(soln, 'vs', approx,' , error = ', error)
        assert error <= 1e-30, "Quadrature error is too high. "
    def test_sparse_quadrature_dim5_order5_clenshaw_curtis(self):
        # test...
        dim = 5
        # Q = tesuract.QuadGen(dim=dim,order=5,rule='clenshaw_curtis',sparse=True)
        x,w = self.xw_d5_o5_CC_sp
        approx = np.sum(genz_exp(x)*w)
        soln = genz_exp_soln(dim)
        error = np.mean((soln - approx)**2)/np.mean(soln**2)
        # print(soln, 'vs', approx,' , error = ', error)
        assert error <= 5e-15, "Quadrature error is too high. "
    def test_sparse_vs_tensor_product_size_d10(self):
        # test...
        dim = 10
        # Q_sp = tesuract.QuadGen(dim=dim,order=2,rule='legendre_gauss',sparse=True)
        x_sp,w_sp = self.xw_d10_o2_LG_sp

        # Q_tp = tesuract.QuadGen(dim=dim,order=2,rule='legendre_gauss',sparse=False)
        x_tp,w_tp = self.xw_d10_o2_LG_tp

        assert x_tp.shape[0]/x_sp.shape[0] >= 250.0, \
            "Sparse grid size should be much smaller than the full tensor product grid"
    def test_sparse_and_tensor_grid_dim_sizes(self):
        # test...
        dim = 10
        # Q_sp = tesuract.QuadGen(dim=dim,order=2,rule='legendre_gauss',sparse=True)
        x_sp,w_sp = self.xw_d10_o2_LG_sp

        # Q_tp = tesuract.QuadGen(dim=dim,order=2,rule='legendre_gauss',sparse=False)
        x_tp,w_tp = self.xw_d10_o2_LG_tp

        assert x_tp.shape[1] == 10 and x_sp.shape[1] == 10, \
            "Dimensions of grids do not match input dimensions. "
    

    def test_sparse_and_tensor_grid_x_and_w_size_match_LG(self):
        # test...
        dim = 10
        # Q_sp = tesuract.QuadGen(dim=dim,order=2,rule='legendre_gauss',sparse=True)
        x_sp,w_sp = self.xw_d10_o2_LG_sp

        # Q_tp = tesuract.QuadGen(dim=dim,order=2,rule='legendre_gauss',sparse=False)
        x_tp,w_tp = self.xw_d10_o2_LG_tp

        assert x_tp.shape[0] == len(w_tp) and x_sp.shape[0] == len(w_sp), \
            "Grid and weight size mismatch."
    def test_sparse_and_tensor_grid_x_and_w_size_match_CC(self):
        # test...
        dim = 5
        # Q_sp = tesuract.QuadGen(dim=dim,order=5,rule='clenshaw_curtis',sparse=True)
        x_sp,w_sp = self.xw_d5_o5_CC_sp

        # Q_tp = tesuract.QuadGen(dim=dim,order=5,rule='clenshaw_curtis',sparse=False)
        x_tp,w_tp = self.xw_d5_o5_CC_tp

        assert x_tp.shape[0] == len(w_tp) and x_sp.shape[0] == len(w_sp), \
            "Grid and weight size mismatch."
    def test_sparse_and_tensor_weight_norm_check_to_1_d10_LG(self):
        # test...
        dim = 10
        # Q_sp = tesuract.QuadGen(dim=dim,order=2,rule='legendre_gauss',sparse=True)
        x_sp,w_sp = self.xw_d10_o2_LG_sp

        # Q_tp = tesuract.QuadGen(dim=dim,order=2,rule='legendre_gauss',sparse=False)
        x_tp,w_tp = self.xw_d10_o2_LG_tp

        assert np.abs(np.sum(w_tp)-1) <= 5e-14 and np.abs(np.sum(w_sp)-1) <= 5e-14, \
            "Weights must sum to 1."
    def test_sparse_and_tensor_weight_norm_check_to_1_d5_LG(self):
        # test...
        dim = 5
        # Q_sp = tesuract.QuadGen(dim=dim,order=5,rule='legendre_gauss',sparse=True)
        x_sp,w_sp = self.xw_d5_o5_LG_sp

        # Q_tp = tesuract.QuadGen(dim=dim,order=5,rule='legendre_gauss',sparse=False)
        x_tp,w_tp = self.xw_d5_o5_LG_tp

        assert np.abs(np.sum(w_tp)-1) <= 5e-14 and np.abs(np.sum(w_sp)-1) <= 5e-14, \
            "Weights must sum to 1."
    def test_sparse_and_tensor_weight_norm_check_to_1_d5_CC(self):
        # test...
        dim = 5
        # Q_sp = tesuract.QuadGen(dim=dim,order=5,rule='clenshaw_curtis',sparse=True)
        x_sp,w_sp = self.xw_d5_o5_CC_sp

        # Q_tp = tesuract.QuadGen(dim=dim,order=5,rule='clenshaw_curtis',sparse=False)
        x_tp,w_tp = self.xw_d5_o5_CC_tp

        assert np.abs(np.sum(w_tp)-1) <= 5e-14 and np.abs(np.sum(w_sp)-1) <= 5e-14, \
            "Weights must sum to 1."
    def test_sparse_check_grid_domain_0_to_1_LG(self):
        # test...
        dim = 5
        # Q_sp = tesuract.QuadGen(dim=dim,order=5,rule='legendre_gauss',sparse=True)
        x_sp,w_sp = self.xw_d5_o5_LG_sp

        # Q_tp = tesuract.QuadGen(dim=dim,order=2,rule='legendre_gauss',sparse=False)
        x_tp,w_tp = self.xw_d5_o5_LG_tp

        assert np.all(x_sp >= 0.0) and np.all(x_sp <= 1.0), \
            "Sparse Grid points must be between 0 and 1. "

        assert np.all(x_tp >= 0.0) and np.all(x_tp <= 1.0), \
            "Tensor Product Grid points must be between 0 and 1. "
    def test_sparse_check_grid_domain_0_to_1_CC(self):
        # test...
        dim = 5
        # Q_sp = tesuract.QuadGen(dim=dim,order=5,rule='clenshaw_curtis',sparse=True)
        x_sp,w_sp = self.xw_d5_o5_CC_sp

        assert np.all(x_sp >= 0.0) and np.all(x_sp <= 1.0), \
            "Sparse Grid points must be between 0 and 1. "

