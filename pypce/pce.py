import numpy as np
import pdb, warnings, pickle
# from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from .multiindex import RecursiveHypMultiIndex
from .multiindex import MultiIndex

from sklearn.utils.estimator_checks import check_estimator 
from sklearn.utils.validation import check_X_y, check_array
from sklearn import linear_model

# Polynomials classes
class PolyBase:
    def __init__(self):
        pass
    def Eval1dBasis(self,x):
        # virtual fun to compute basis values at x up to order K
        pass
    def Eval1dBasisNorm(self,x):
        # virtual fun to compute norm basis values at x up to order K
        pass

def Leg1dPoly(order,x):
    if order == 0:
        if x.ndim == 0:
            y = 1.0
        else:
            y = np.ones(len(x))
    elif order == 1:
        y = x
    elif order == 2:
        y = .5*(3.0*x**2-1)
    elif order == 3:
        y = .5*(5.0*x**3-3*x)
    elif order == 4:
        y = (1./8)*(35.0*x**4 - 30.0*x**2 + 3.0)
    elif order == 5:
        y = (1./8)*(63.0*x**5 - 70.0*x**3 + 15.0*x)
    elif order == 6:
        y = (1./16)*(231.0*x**6 - 31.0*x**4 + 105.0*x**2  - 5.0)
    return y
    
class LegPoly2(PolyBase):
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
        self.Lis = [Leg1dPoly(k,x0) for k in range(self.K+1)]
        return np.array(self.Lis) # dim x nx
    def normsq(self,K):
        # compute the squared norm of each basis up to K
        bma = self.domain[1] - self.domain[0]
        normsq = np.array([bma/(2*k+1) for k in range(K+1)])
        self.normsq = normsq
        return normsq

from numpy.polynomial.legendre import Legendre
class LegPoly(PolyBase):
    '''
    Usage for 1d polynomial basis construction
    L = LegPoly()
    x = np.linspace(-1,1,100)
    L.plotBasis(x,K,normed=True)
    '''
    def __init__(self,domain=np.array([-1,1])):
        self.domain = domain
        super().__init__()
    def Eval1dBasis(self,x,K):
        # returns matrix where each column is Li(x)
        self.K = K
        self.Lis = [Legendre.basis(k,self.domain) for k in range(self.K+1)]
        return np.array([Li(x) for Li in self.Lis])
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
    def __init__(self,domain=np.array([-1,1])):
        self.domain = domain
        self.a = domain[0]
        self.b = domain[1]
        super().__init__()
    def Eval1dBasis(self,x,K):
        # returns matrix where each column is Li(x)
        self.K = K
        # compute norms
        normsq = np.array([(self.b-self.a)/(2*k+1) for k in range(K+1)])
        self.Lis = [Legendre.basis(k,self.domain) for k in range(self.K+1)]
        return np.array([Li(x)/np.sqrt(normsq[i]) for i,Li in enumerate(self.Lis)])
    def normsq(self,K):
        # compute the squared norm of each basis up to K
        bma = self.domain[1] - self.domain[0]
        normsq = np.array([bma/(2*k+1) for k in range(K+1)])
        self.normsq = 1.0 + 0*normsq
        return normsq

class PolyFactory:
    # generates PolyBase class object
    @staticmethod
    def newPoly(polytype='Leg'):
        L = LegPoly()
        return L
    def newPoly(polytype='LegNorm'):
        L = LegPolyNorm()
        return L


class PCEBuilder(BaseEstimator):
    ''' Base class for building a multivariate polynomial basis.  

    This class creates a multi-variate polynomial object, aka as a polynomial
    chaos model. The expansion looks like 

    .. math::

        \sum_{i=1}^N c_i \Phi_i(\mathbf{x})

    where :math:`N` is the number of polynomial terms, and
    :math:`\Phi_i:\mathbf{x} \in \mathbb{R}^d \mapsto \mathbb{R}` is the
    multivariate basis function which takes the form 

    .. math::

        \Phi_i = \prod_{j=1}^d L_{\\alpha_j^{(i)}}(x_i),

    where :math:`\\alpha_j` is an integer tuple of size :math:`d` which
    represents the multiindex, and :math:`L:\mathbb{R} \mapsto \mathbb{R}` are
    the one-dimensional Legendre polynomials. So for example, in two dimensions,
    we can have :math:`\Phi_i = xy`. The construction of the multiindex is done
    behind the scenes in the ``multindex`` module.

    Below we list the parameters to construct the object followed by the class
    attributes and returns. 

    Parameters
    ----------
    order : int, default=1 

        Description of the order of the polynomials in the
        expansion. For total order, the order is the maximum polynomial order
        for each basis function per dimension.

    customM : numpy.ndarray, default=None 

        An integer numpy aray of size (# of basis functions, # of dimensions). Each
        row represents the basis function and each column represents the order of
        the 1d polynomial, :math:`L_i`. 

    mindex_type : {'total_order', 'hyperbolic'}, default='total_order' 

        Different types of multiindex generators. Total order produces basis vectors
        that have a maximum order as defined by **order**. Hyperbolic order is
        similar but generates fewer cross terms. 

    coef : numpy.ndarray, default=None

        1d numpy coefficient array if defining polynomial. Must be the same length
        as the number of basis elements, i.e. length of multiindex array. 

    polytype : {'Legendre'}, default='Legendre'

        A string representing the type of polynomial. So far we only include
        Legendre polynomial construction defined on [-1,1]

    normalized : bool, default=False

        Whether or not to use normalized polynomials such that
        :math:`\int_{-1}^{1}L_i(x)dx = 1` or :math:`\\frac{2}{2n+1}` if True or False, respecively, where
        :math:`n` is the order of the polynomial (only for Legendre polynomials).  

    Attributes
    ----------
    dim : int

        Dimension of the polynomial, defined after construction. 
        
    mindex : ndarray of shape (nbasis, dim)

        Integer array of the multiindex which describes the structure of the
        polynomial basis. 

    Notes
    -----
    As of now, the base class requires the input range to be on the default
    range :math:`[-1,1]` for all dimensions. We have included a useful
    preprocessing utility (``DomainScaler``) to transform the domain easily to
    the canonical target range from any input range. In the future this will be
    an option in the this class. 

    Todo
    -----
    * Add option for non-standard domain. 
    * Add attribute for feature_importances() is polynomial is defined. 

    Examples
    --------
    >>> from pypce import PCEBuilder p = PCEBuilder(order=3,normalized=True)
    >>> print(p.mindex)
    '''
    def __init__(self,order=1,customM=None,mindex_type='total_order',coef=None,a=None,b=None,polytype='Legendre',normalized=False):
        # self.dim = dim # no need to initialize with dim
        self.order = order
        self.dim = None
        self.customM = customM
        self.mindex_type = mindex_type
        self.coef = coef
        # self.coef_ = self.coef
        self.polytype = polytype
        # self.a = a # lower bounds on domain of x
        # self.b = b # upper bound on domain of x
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
        elif isinstance(self.customM,MultiIndex):
            # print("Using custom Multiindex object.")
            assert(self.customM.dim == dim)
            self.M = customM
        elif isinstance(self.customM,np.ndarray):
            self.dim = self.customM.shape[1] # set dim to match multiindex
            self.M = MultiIndex(self.dim,order=1,mindex_type='total_order') # use default
            self.M.setIndex(self.customM)
            self.order = None # leave blank to indicate custom order
        self.mindex = self.M.index
        self.multiindex = self.mindex
        if self.mindex.dtype != 'int':
            warnings.warn("Converting multindex array to integer array.")
            self.mindex = self.mindex.astype('int')
        self.nPCTerms = self.mindex.shape[0]
        if self.coef is not None: 
            assert len(self.coef) == self.mindex.shape[0], "coefficient array is not the same size as the multindex array."
        self.compile_flag = True
    def computeNormSq(self):
        ''' separate routine to compute norms, in order to bypass construct basis

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        x : int
            Parameter names mapped to their values.

        '''
        mindex = self.mindex # use internal mindex array
        # L = []
        Kmax = np.amax(self.mindex,axis=0)
        NormSq = []
        for i in range(self.dim):
            # Compute Legendre objects and eval 1d basis
            if self.normalized == False:
                Leg = LegPoly()
            elif self.normalized == True:
                Leg = LegPolyNorm()
            NormSq.append(Leg.normsq(Kmax[i])) # norms up to K order

        # start computing products
        # Phi = 1.0
        normsq = 1.0
        for di in range(self.dim):
            # Phi = L[di][self.mindex[:,di]] * Phi
            normsq = NormSq[di][mindex[:,di]] * normsq   
        self.normsq = normsq # norm squared * integration factor of 1/2 (since domain is [-1,1])
        self.norm = np.sqrt(normsq)
        return self.normsq
    def fit_transform(self,X):
        """Placeholder
        """
        # only works for [-1,1] for far
        # compute multindex
        assert np.amin(X) >= -1 and np.amax(X) <= 1, "range for X must be between -1 and 1 for now. scale inputs accordingly. "
        X = np.atleast_2d(X)
        if self.mindex is None:
            self.compile(dim=X.shape[1]) # only compiles once
        Max = np.amax(self.mindex,axis=0)
        # construct and evaluate each basis using mindex
        L = []
        NormSq = []
        self.output_type = None
        if X.ndim == 1: 
            X = np.atleast_2d(X)
            self.output_type = 'scalar'
        for i in range(self.dim):
            # Compute Legendre objects and eval 1d basis
            if self.normalized == False:
                Leg = LegPoly()
            elif self.normalized == True:
                Leg = LegPolyNorm()
            Li_max = Leg.Eval1dBasis(x=X[:,i],K=Max[i])
            L.append(Li_max) # list of size dim of Legendre polys
            NormSq.append(Leg.normsq(Max[i])) # norms up to K order

        # start computing products
        Phi = 1.0
        normsq = 1.0
        for di in range(self.dim):
            Phi = L[di][self.mindex[:,di]] * Phi
            normsq = NormSq[di][self.mindex[:,di]] * normsq
        self.Phi = Phi.T 
        self.normsq = normsq # norm squared
        # if self.normalized:
        #     return self.Phi/np.sqrt(normsq)
        # else:
        #     return self.Phi
        return self.Phi
    def polyeval(self,X=None,c=None):
        """Placeholder
        """
        if c is not None: 
            coef_ = c # use c in polyeval
            assert self.compile_flag == True, "Must compile to get multindex."
            assert len(coef_) == self.mindex.shape[0], "c is not the right size."
        elif c is None: 
            assert self.coef is not None, "Must define coefficient in constructor or polyeval or run the fit method"
            coef_ = self.coef # use coefficient from constructor

        if X is not None:
            Phi = self.fit_transform(X)
            yeval = np.dot(Phi,coef_)
        elif X is None:
            Phi = self.Phi # using stored Vandermonde matrix
            yeval = np.dot(Phi,coef_)

        return yeval
    def eval(self,X=None,c=None):
        """Placeholder
        """
        return self.polyeval(X=None,c=None)
    def computeSobol(self,c=None):
        """Placeholder
        """
        if self.mindex is None:
            self.compile(dim=self.dim)
        # assert self.compile_flag == True, "Must compile to set mindex. Avoids having to recompute multiindex everytime we run SA."
        if len(self.normsq) == 0: 
            normsq = self.computeNormSq() # if c and M specified in constructor
        else:
            normsq = self.normsq # from fit transform
        # compute Sobol indices using coefficient vector
        if c is None:
            assert len(self.coef) != 0, "must specify coefficient array or feed it in the constructor."
            coef_ = self.coef
        else:
            coef_ = c

        assert len(coef_) == self.mindex.shape[0], "Coefficient vector must match the no of rows of the multiindex."

        if self.normalized:
            totvar_vec = coef_[1:]**2
            self.coefsq = coef_**2
        else:
            totvar_vec = normsq[1:]*coef_[1:]**2
            self.coefsq = normsq*coef_**2
        totvar = np.sum(totvar_vec)
        # assert totvar > 0, "Coefficients are all zero!"
        S = []
        # in case elements have nan in them
        if np.all(coef_[1:] == 0): # ignore the mean
            print("Returning equal weights!")
            S = np.ones(self.dim)/self.dim # return equal weights
        else:
            for i in range(self.dim):
                si = self.mindex[:,i] > 0 # boolean
                s = np.sum(totvar_vec[si[1:]])/totvar
                S.append(s)
        return S
    def computeMoments(self,c=None):
        """Placeholder
        """
        if c is None:
            assert self.coef is not None, "coef is not defined. Try running fit or input c"
            coef_ = self.coef
        else:
            coef_ = c
            assert len(c) == len(self.mindex), "coef is wrong size"
        # normsq = self.computeNormSq()
        normsq = self.normsq
        prob_weight = .5**(self.dim)
        if self.normalized is False:
            self.mu = prob_weight * normsq[0]*coef_[0]
            self.var = prob_weight * np.sum(normsq[1:]*coef_[1:]**2)
        elif self.normalized is True:
            self.mu = prob_weight * np.sqrt(normsq[0])*coef_[0]
            self.var = prob_weight * np.sum(coef_[1:]**2)
        return self.mu, self.var

class PCEReg_old(PCEBuilder,RegressorMixin):
    def __init__(self,order=2,customM=None,mindex_type='total_order',coef=None,a=None,b=None,polytype='Legendre',fit_type='linear',alphas=np.logspace(-12,1,20),l1_ratio=[.001,.5,.75,.95,.999,1],lasso_tol=1e-2,normalized=False,w=None):
        self.order = order
        self.mindex_type = mindex_type
        self.customM = customM
        self.a = a
        self.b = b
        self.coef = coef
        # self.c = self.coef # need to remove eventually
        self.polytype = 'Legendre'
        self.fit_type = fit_type
        self.alphas = alphas # LassoCV default
        self.l1_ratio = l1_ratio # ElasticNet default
        self.lasso_tol = lasso_tol
        self.w = w
        self.normalized = normalized
        super().__init__(order=self.order,customM=self.customM,mindex_type=self.mindex_type,coef=self.coef,a=self.a,b=self.b,polytype=self.polytype,normalized=self.normalized)
    def _quad_fit(self,X,y,w):
        # let us assume the weights are for [-1,1]
        X,w = check_X_y(X,w) # make sure # quadrature points matces X
        self._compile(X) # compute normalized or non-normalized Xhat
        normsq = self.computeNormSq()
        assert np.abs(np.sum(w) - 1) <= 1e-15, "quadrature weights must be scaled to integrate over [-1,1] with unit weight"
        int_fact = 2**self._dim
        if self.normalized == True:
            self.coef = int_fact*np.dot(w*y,self.Xhat)/np.sqrt(normsq)
        else:   
            self.coef = int_fact*np.dot(w*y,self.Xhat)/normsq
        return self
    def _compile(self,X):
        # build multindex and get Xhat
        self._dim = X.shape[1]
        super().compile(dim=self._dim) # use parent compile to produce the multiindex
        self._M = self.mindex
        self.Xhat = self.fit_transform(X)
        return self
    def fit(self,X,y,w=None):
        X,y = check_X_y(X,y)
        # get data attributes
        self._n,self._dim = X.shape
        self._compile(X) # build multindex and construct basis
        # pypce.PCEBuilder(dim=self.dim,self.order)
        # run quadrature fit if weights are specified:
        if self.coef is not None:
            assert len(self.coef) == self._M.shape[0],"length of coefficient vector is not the same shape as the multindex!"
        elif w is not None:
            self._quad_fit(X,y,w)
        elif self.w is not None:
            self._quad_fit(X,y,self.w)
        else:
            # LINEAR fit
            Xhat,y = check_X_y(self.Xhat,y)
            if self.fit_type == 'linear':
                regmodel = linear_model.LinearRegression(fit_intercept=False)
                regmodel.fit(Xhat,y)
            elif self.fit_type == 'LassoCV':
                regmodel = linear_model.LassoCV(alphas=self.alphas,fit_intercept=False,max_iter=1000,tol=self.lasso_tol)
                regmodel.fit(Xhat,y)
                self.alpha_ = regmodel.alpha_
            elif self.fit_type == 'OmpCV':
                regmodel = linear_model.OrthogonalMatchingPursuitCV(fit_intercept=False)
                regmodel.fit(Xhat,y)
            elif self.fit_type == 'ElasticNetCV':
                regmodel = linear_model.ElasticNetCV(l1_ratio=self.l1_ratio,fit_intercept=False,n_alphas=25,tol=1e-2)
                regmodel.fit(Xhat,y)
            self.coef = regmodel.coef_
            # self.pce.coef_ = self.coef_ # set internal pce coef
        return self
    def sensitivity_indices(self):
        assert self.coef is not None, "Must run fit or feed in coef array."
        return self.computeSobol()
    def multiindex(self):
        assert self._M is not None, "Must run fit or feed in mindex array."
        return self._M
    def predict(self,X):
        # check_is_fitted(self)
        X = np.atleast_2d(X)
        X = check_array(X)
        # self._fit_transform(X)
        # Phi = check_array(self._Phi)
        # ypred = np.dot(Phi,self.coef_)
        ypred = self.polyeval(X)
        return ypred

class PCEReg(PCEBuilder,RegressorMixin):
    def __init__(self,order=2,customM=None,mindex_type='total_order',coef=None,a=None,b=None,polytype='Legendre',fit_type='linear',fit_params={},normalized=False):
        # alphas=np.logspace(-12,1,20),l1_ratio=[.001,.5,.75,.95,.999,1],lasso_tol=1e-2
        self.order = order
        self.mindex_type = mindex_type
        self.customM = customM
        self.a = a
        self.b = b
        self.coef = coef
        # self.c = self.coef # need to remove eventually
        self.polytype = 'Legendre'
        self.fit_type = fit_type
        self.fit_params = fit_params
        # self.alphas = alphas # LassoCV default
        # self.l1_ratio = l1_ratio # ElasticNet default
        # self.lasso_tol = lasso_tol
        # self.w = w
        self.normalized = normalized
        super().__init__(order=self.order,customM=self.customM,mindex_type=self.mindex_type,coef=self.coef,a=self.a,b=self.b,polytype=self.polytype,normalized=self.normalized)
    def _compile(self,X):
        # build multindex and get Xhat
        self._dim = X.shape[1]
        super().compile(dim=self._dim) # use parent compile to produce the multiindex
        self._M = self.multiindex
        self.Xhat = self.fit_transform(X)
        return self
    def _quad_fit(self,X,y):
        # let us assume the weights are for [-1,1]
        self._compile(X) # compute normalized or non-normalized Xhat
        normsq = self.computeNormSq()
        assert 'w' in self.fit_params.keys(), "quadrature weights must be given in dictionary fit_params."
        w = self.fit_params['w']
        X,w = check_X_y(X,w) # make sure # quadrature points matces X
        assert np.abs(np.sum(w) - 1) <= 1e-15, "quadrature weights must be scaled to integrate over [-1,1] with unit weight"
        int_fact = 2**self._dim
        if self.normalized == True:
            self.coef = int_fact*np.dot(w*y,self.Xhat)/np.sqrt(normsq)
        else:   
            self.coef = int_fact*np.dot(w*y,self.Xhat)/normsq
        return self
    def fit(self,X,y):
        X,y = check_X_y(X,y)
        # get data attributes
        self._n,self._dim = X.shape
        self._compile(X) # build multindex and construct basis
        Xhat,y = check_X_y(self.Xhat,y)
        # pypce.PCEBuilder(dim=self.dim,self.order)
        # run quadrature fit if weights are specified:
        if self.coef is not None:
            assert len(self.coef) == self.multiindex.shape[0],"length of coefficient vector is not the same shape as the multindex!"
        if self.fit_type == "quadrature":
            self._quad_fit(X,y)
        if self.fit_type == 'linear':
            regmodel = linear_model.LinearRegression(fit_intercept=False)
            regmodel.fit(Xhat,y)
        if self.fit_type == 'LassoCV':
            if not self.fit_params: # if empty dictionary
                self.fit_params={'alphas':np.logspace(-12,1,20),'max_iter':1000,'tol':1e-2}
            regmodel = linear_model.LassoCV(fit_intercept=False,**self.fit_params)
            regmodel.fit(Xhat,y)
            self.alpha_ = regmodel.alpha_
        if self.fit_type == 'ElasticNetCV':
            if not self.fit_params: # if empty dictionary
                self.fit_params={'l1_ratio':[.001,.5,.75,.95,.999,1],'n_alphas':25,'tol':1e-2}
            regmodel = linear_model.ElasticNetCV(fit_intercept=False,**self.fit_params)
            regmodel.fit(Xhat,y)
        elif self.fit_type == 'OmpCV':
            if not self.fit_params: # if empty dictionary
                pass
            regmodel = linear_model.OrthogonalMatchingPursuitCV(fit_intercept=False,**self.fit_params)
            regmodel.fit(Xhat,y)
        if self.fit_type != "quadrature":
            self.coef = regmodel.coef_
        self.coef_ = self.coef # backwards comaptible with sklearn API
        self.feature_importances()
        return self
    def sensitivity_indices(self):
        assert self.coef is not None, "Must run fit or feed in coef array."
        return self.computeSobol()
    def feature_importances(self):
        S = np.array(self.computeSobol())
        S = S/S.sum()
        self.feature_importances_ = S
        return self.feature_importances_
    def multiindex(self):
        assert self._M is not None, "Must run fit or feed in mindex array."
        return self._M
    def predict(self,X):
        # check_is_fitted(self)
        X = np.atleast_2d(X)
        X = check_array(X)
        # self._fit_transform(X)
        # Phi = check_array(self._Phi)
        # ypred = np.dot(Phi,self.coef)
        ypred = self.polyeval(X)
        return ypred

# In development
class PCESeries():
    def __init__(self,dim,C,MIs,time=[],a=None,b=None):
        # construct series of pce's
        assert len(C) == len(MIs), 'coef and multindex mismatch!'
        # add assert for single array
        if len(time) == 0:
            # assign simple index as time
            time = np.arange(len(C))
        self.time = time
        self.C = C
        self.MIs = MIs
        self.N = len(C)
        self.dim = dim
        self.a = a
        self.b = b
        # construct PCEs
        self.__constructPCEs()
    def __constructPCEs(self):
        self.PCEs = []
        for i in range(self.N):
            pcetemp = PCEBuilder(dim=self.dim,
                                 customM=self.MIs[i],
                                 c=self.C[i],a=self.a,b=self.b)
            self.PCEs.append(pcetemp)
    def eval(self,X):
        assert np.atleast_2d(X).shape[1] == self.dim, "dimension mismatch!"
        y = []
        for i in range(self.N):
            ytemp = self.PCEs[i].eval(X)
            y.append(ytemp)
        y = np.array(y).T
        return y
    def ieval(self,T,X):
        # each element of T can be a vector
        # first evaluate all PCEs at X points
        Y = self.eval(X)
        # interpolate for each time point
        T1d = np.atleast_1d(T)
        assert len(Y) == len(T1d), "time and X mismatch!"
        Yi = []
        for i in range(len(T1d)):
            yitemp = np.interp(T1d[i],self.time,Y[i])
            Yi.append(yitemp)
        return np.array(Yi)

# class pcasurrogate:
#   def __init__(self,X,Y,norm=True,xbounds=None,xlabels=None,npca=50,pca_datafile=None):
#       self.xnsamples,self.d = X.shape
#       self.ynsamples,self.n = Y.shape
#       self.xlabels = xlabels
#       self.pca_datafile = pca_datafile
#       assert self.xnsamples == self.ynsamples, "Sample mismatch between X and Y!"
#       # normalize X data
#       if norm == True:
#           assert xbounds is not None, "Must provide bounds on X. Otherwise, set norm=False"
#           LB,UB = xbounds
#           self.Xscaled = 2*( (X - LB) * 1./(UB-LB) ) - 1
#           # normalize Y -> [0,1]
#           self.Ymax = np.amax(Y)
#           self.Ymin = np.amin(Y)
#           self.Yscaled = (Y - self.Ymin)/(self.Ymax - self.Ymin)
#       # perform PCA with npca components
#       self.npca = npca
#       assert self.npca >= 2, "Number of PCA should be > 1!"
#       self.project_flag = False
#   def run(self,full=True,load=False,save=False,mre_cutoff=1e-3,K=None,error=False):
#       # if load == True, just load data
#       if error == False: assert K is not None, "Must choose K PCA modes."
#       if K is not None: self.npca = K # set npca to K
#       if save == True: assert self.pca_datafile is not None, "datafile field must be specified."
#       if full == True and load == False:
#           self.performPCA(self.npca)
#           if error == True: # only perform recon error if user defined
#               if self.pca_datafile is None or save == False:
#                   "running PCA without saving data..." 
#                   self.performPCAerroranalysis(save=False)
#               else:
#                   self.performPCAerroranalysis(save=True,filename=self.pca_datafile)
#       elif load == True:
#           assert self.pca_datafile is not None, "Must specify pca_datafile."
#           if save == True: warnings.warn("save option is ignored.")
#           self.loaddata(self.pca_datafile)
#       self.project(K=K,mre_cutoff=mre_cutoff)
#   def performPCA(self,npca):
#       print("performing PCA analysis...")
#       self.pca = PCA(n_components=self.npca,svd_solver='full')
#       self.pca.fit(self.Yscaled)
#       # cumulative sum of explained variance
#       self.explained_variance = np.cumsum(self.pca.explained_variance_ratio_)
#       # save data
#       self.muY = self.Yscaled.mean(0)
#       self.Y0 = self.Yscaled - self.muY
#       self.V = self.pca.components_.T # projection basis
#       self.P = np.dot(self.Y0,self.V) # projection coefficients
#       assert self.V.shape[0] == self.n, "Basis dimension should be same dimension as each Y sample."
#       assert self.P.shape == (self.ynsamples,self.npca), "Shape of projection coefficients is not right!"
#   def performPCAerroranalysis(self,save=True,filename=None):
#       # calculate relative error of projections for increasing npca up to self.npca
#       self.__mre = [] # mean relative reconstruction error
#       for k in tqdm(range(2,self.npca)):
#           Y0approx = np.dot(self.V[:,:k],self.P[:,:k].T).T
#           RE = [] # relative error for each sample
#           for i,y0 in enumerate(self.Y0):
#               re = np.linalg.norm(y0 - Y0approx[i])/np.linalg.norm(self.muY + y0)
#               RE.append(re)
#           muRE = np.mean(np.array(RE))
#           # print("Mean relative errors", np.mean(RE))
#           self.__mre.append(muRE) # average over all samples
#           self.mre = np.array(self.__mre)
#       if save == True:
#           assert filename is not None, "Must specify filename to save pca dictionary data."
#           # save data as dictionary
#           datadict = {}
#           datadict['V'] = self.V # projection basis
#           datadict['P'] = self.P # projection coefficients
#           datadict['Y0'] = self.Y0 # centered data
#           datadict['muY'] = self.muY # mean of original data
#           datadict['mre'] = self.mre # reconstruction relative error
#           datadict['expvar'] = self.explained_variance # cumulative explained variance
#           datadict['Xscaled'] = self.Xscaled # normalized data to [-1,1]
#           datadict['xlabels'] = self.xlabels # labels of x data
#           print("saving PCA data...")
#           with open(filename + '.pkl', 'wb') as pfile:
#               pickle.dump(datadict, pfile)
#   def loaddata(self,filename):
#       # import data dictionary with PCA modes and projections
#       try:
#           with open(filename, 'rb') as pfile:
#               datadict = pickle.load(pfile)
#       except:
#           with open(filename + '.pkl', 'rb') as pfile:
#               datadict = pickle.load(pfile)
#       print("loading PCA data...")
#       self.V = datadict['V'] # projection basis
#       self.P = datadict['P'] # projection coefficients
#       self.Y0 = datadict['Y0'] # centered data
#       self.muY = datadict['muY'] # mean of original data
#       self.mre = datadict['mre'] # reconstruction relative error
#       self.explained_variance = datadict['expvar'] # cumulative explained variance
#       self.Xscaled = datadict['Xscaled'] # normalized data to [-1,1]
#       self.xlabels = datadict['xlabels'] # labels of x data
#   def project(self,K=None,mre_cutoff=1e-3):
#       if K is not None:
#           self.K = K # use K if user defined
#       else:
#           # find PCA mode where recon error is greater than cutoff
#           cutoff_indices = np.where(self.mre > mre_cutoff)[0]
#           if len(cutoff_indices) == 0:
#               warnings.warn("cutoff is too large. try decreasing. setting K = 1")
#               self.K = 1 # only use 1 PCA mode
#           else:
#               self.K = cutoff_indices[-1] + 1 # add one since we are starting with 1 after the mean

#       # get projections and basis columns
#       self.Vk = self.V[:,:self.K]
#       self.Pk = self.P[:,:self.K] # each row is the proj coef
#       self.project_flag = True
#   def invert(self,P):
#       P = np.atleast_2d(P.copy())
#       assert P.shape[1] == S.K, "dimension mismatch in P and V."
#       R = np.dot(self.Vk,P.T)
#       return R.T
#   def pcefit(self):
#       assert self.project_flag == True, "Must project data onto K PCA modes first. "
#       npce = self.Vk.shape[1]


# # create function to plot sobol indices in a stacked graph
# # to be added to the pypce PCESeries class
# C = []
# M = []
# Sobol = []
# t = []
# for pce in PCEs:
#   C.append(pce['c'])
#   M.append(pce['M'])
#   t.append(pce['pca_mode'])
#   Sobol.append(pce['S'])

# xlabels = np.array(['dens_sc', 'vel_sc', 'temp_sc', 
#                   'sigk1', 'sigk2', 'sigw1', 'sigw2',
#                   'beta_s', 'kap', 'a1', 'beta1r', 'beta2r'])
# pceS = pypce.PCESeries(dim=xdim,C=C,MIs=M)

# Sobol = np.array(Sobol)
# # S = np.random.rand(4,12)
# normS = np.array([s/np.sum(s) for s in Sobol]).T

# import matplotlib.pyplot as plt
# import matplotlib._color_data as mcd

# xkcd_colors = []
# xkcd = {name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS}
# for j, n in enumerate(xkcd):
#   xkcd = mcd.XKCD_COLORS["xkcd:" + n].upper()
#   xkcd_colors.append(xkcd)

# Ps = []
# bottom = np.zeros(len(t))
# import seaborn as sns
# sns.set_palette(sns.color_palette("Paired", 12))
# plt.figure(figsize=(20,9))
# for ii in range(xdim):
#   ptemp = plt.bar(t,normS[ii],bottom=bottom,width=.25)
#   bottom = bottom + normS[ii] # reset bottom to new height
#   Ps.append(ptemp)
# plt.ylabel('Sobol Scores')
# plt.ylim([0,1.1])
# # plt.title('Sobol indices by pca mode')
# plt.xticks(t, ('PCA1','PCA2','PCA3','PCA4','PCA5','PCA6'))
# plt.legend(((p[0] for p in Ps)), (l for l in xlabels),
#           fancybox=True, shadow=True,
#           loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=xdim)

# # plot explained variance
# plt.plot(t,S.explained_variance,'--ok')
# plt.savefig('figs/SA_'+y_label+'.png')