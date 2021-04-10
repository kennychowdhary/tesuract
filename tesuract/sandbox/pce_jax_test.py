import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import scipy

from multiindex import MultiIndex

# This needs to run at startup
# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#Double-(64bit)-precision
jax.config.update('jax_enable_x64', True)

# Polynomials classes
class PolyBase:
    def __init__(self):
        pass
    def Eval1dBasis(self,x):
        # virtual fun to compute basis values at x up to order K
        pass
    def normsq(self,K):
        # compute L2 norm of the basis up to order K
        pass

def Legendre(order,x):
    if order == 0:
        if x.ndim == 0:
            y = 1.0
        else:
            y = jnp.ones(len(x))
    elif order == 1:
        y = x
    elif order == 2:
        y = .5*(3.0*x**2-1)
    elif order == 3:
        y = .5*(5.0*x**3-3*x)
    elif order == 4:
        y = (1./8)*(35.0*x**4 - 30.0*x**2 + 3.0)
    return y
    
class LegPoly(PolyBase):
    '''
    Usage for 1d polynomial basis construction
    L = LegPoly()
    x = np.linspace(-1,1,100)
    L.plotBasis(x,K,normed=True)
    '''
    def __init__(self,domain=[-1.0,1.0]):
        self.domain = domain
        self.a = domain[0]
        self.b = domain[1]
        super().__init__()
    def Eval1dBasis(self,x,K):
        # returns matrix where each column is Li(x)
        self.K = K
        # assert jnp.all(x>=self.a) and jnp.all(x<=self.b), "x is not in the domain."
        # transform x to [-1,1]
        x0 = 2.*(x - 1.*self.a)/(1.*self.b-self.a) - 1.0
        # assert jnp.all(x0>=-1) and jnp.all(x0<=1), "x is not in the domain."
        self.Lis = [Legendre(k,x0) for k in range(self.K+1)]
        return jnp.array(self.Lis) # dim x nx
    def normsq(self,K):
        # compute the squared norm of each basis up to K
        bma = self.domain[1] - self.domain[0]
        normsq = np.array([bma/(2*k+1) for k in range(K+1)])
        self.normsq = normsq
        return normsq

class LegPolyNorm(PolyBase):
    '''
    Usage for 1d polynomial basis construction
    L = LegPoly()
    x = np.linspace(-1,1,100)
    L.plotBasis(x,K,normed=True)
    '''
    def __init__(self,domain=[-1.,1.]):
        self.domain = domain
        self.a = domain[0]
        self.b = domain[1]
        super().__init__()
    def Eval1dBasis(self,x,K):
        # returns matrix where each column is Li(x)
        self.K = K
        # compute norms
        normsq = np.array([(self.b-self.a)/(2*k+1) for k in range(K+1)])
        # check bounds
        # assert jnp.all(x>=self.a) and jnp.all(x<=self.b), "x is not in the domain."
        # transform x to [-1,1]
        x0 = 2.*(x - 1.*self.a)/(1.*self.b-self.a) - 1.0
        # assert jnp.all(x0>=-1) and jnp.all(x0<=1), "x is not in the domain."
        self.Lis = [Legendre(k,x0)/jnp.sqrt(normsq[i]) for i,k in enumerate(range(self.K+1))]
        return jnp.array(self.Lis) # dim x nx
    def normsq(self,K):
        # compute the squared norm of each basis up to K
        bma = self.domain[1] - self.domain[0]
        normsq = np.array([bma/(2*k+1) for k in range(K+1)])
        self.normsq = 1 + 0*normsq
        return normsq

from sklearn.base import BaseEstimator
class PCEBuilder(BaseEstimator):
    def __init__(self,order=1,customM=None,mindex_type='total_order',coef=None,polytype='Legendre',normalized=False):
        # self.dim = dim # no need to initialize with dim
        self.order = order
        self.dim = None
        self.customM = customM
        self.mindex_type = mindex_type
        self.coef = coef
        self.polytype = polytype
        self.normsq = np.array([])
        self.compile_flag = False
        self.mindex = None
        self.normalized = normalized
    def compile(self,dim):
        """Placeholder
        """
        self.dim = dim
        #constructor for different multindices (None, object, and array)
        if self.customM is None:
            self.M = MultiIndex(dim,self.order,self.mindex_type)
        elif isinstance(self.customM,np.ndarray):
            self.dim = self.customM.shape[1] # set dim to match multiindex
            self.M = MultiIndex(self.dim,order=1,mindex_type='total_order') # use default
            self.M.setIndex(self.customM)
            self.order = None # leave blank to indicate custom order
        self.mindex = self.M.index
        self.multiindex = self.mindex
    def fit_transform(self,X):
        """Placeholder
        """
        self.compile(dim=X.shape[1])
        Kmax = jnp.amax(mindex,axis=0)
            # construct and evaluate each basis using mindex
        L = []
        NormSq = []
        self.output_type = None
        if X.ndim == 1: 
            X = jnp.atleast_2d(X)
            self.output_type = 'scalar'
        for i in range(dim):
            # Compute Legendre objects and eval 1d basis
            Leg = LegPoly()
            Li_max = Leg.Eval1dBasis(x=X[:,i],K=Kmax[i])
            L.append(Li_max) # list of size dim of Legendre polys
            NormSq.append(Leg.normsq(Kmax[i])) # norms up to K order

        # start computing products
        Phi = 1.0
        normsq = 1.0
        for di in range(dim):
            Phi = L[di][self.mindex[:,di]] * Phi
            normsq = NormSq[di][self.mindex[:,di]] * normsq
        self.Phi = Phi.T 
        return self.Phi
    def eval(self,c,X=None):
        """Placeholder
        """
        if X is not None:
            self.fit_transform(X)
        
        peval = jnp.dot(self.Phi,c)
        if self.output_type == 'scalar':
            return peval[0]
        else:
            return peval

#########################
######## TEST ###########
#########################

# generate multivariate polynomial
dim = 2
M = MultiIndex(dim=dim,order=1)
mindex = M.index
coef = jnp.array([1.,2,3])
assert len(coef) == len(mindex)
# X = jrandom.uniform(rng,shape=(100,dim))
x = jnp.linspace(-1,1,25)
X = jnp.array([x.flatten() for x in jnp.meshgrid(x,x)]).T

p = PCEBuilder(order=1)
p.fit_transform(X)
def mvp2(c): 
    return p.eval(c=c)
Z = mvp2(coef)

from scipy.optimize import minimize

def fun(c):
    return jnp.mean((Z - mvp2(c=c))**2)

gfun = jax.grad(fun)
gfun_np = lambda c: np.array(jax.jit(gfun)(c))
# gfun_used = gfun_np

fun_and_gfun = jax.value_and_grad(fun)

c0 = np.zeros(3)
optopt = {'gtol': 1e-8, 'disp': False}
res = scipy.optimize.minimize(fun, c0, jac=gfun_np, method='L-BFGS-B', options=optopt)
print(res)

# from mpl_toolkits.mplot3d import Axes3D  
# from matplotlib.pyplot import *
# fig = figure()
# ax = fig.gca(projection='3d')

# surf = ax.scatter(X[:,0],X[:,1],Z)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# show()
