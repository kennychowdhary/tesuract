import numpy as np
import matplotlib.pyplot as mpl
import pdb, warnings, pickle
from tqdm import tqdm
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
	def __init__(self,K):
		pass
	def Eval1dBasis(self,x):
		# virtual fun to compute basis values at x up to order K
		pass
	def Eval1dBasisNorm(self,x):
		# virtual fun to compute norm basis values at x up to order K
		pass

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
	def Eval1dBasis(self,x,K):
		# returns matrix where each column is Li(x)
		self.K = K
		self.Lis = [Legendre.basis(k,self.domain) for k in range(self.K+1)]
		return np.array([Li(x) for Li in self.Lis]).T
	def Eval1dBasisNorm(self,x,K):
		# returns matrix where each column is Li(x)
		self.K = K
		self.Lis = [Legendre.basis(k,self.domain) for k in range(self.K+1)]
		return np.array([Li(x)/self.normsq[i] for i,Li in enumerate(self.Lis)]).T
	def normsq(self,K):
		# compute the squared norm of each basis up to K
		bma = self.domain[1] - self.domain[0]
		normsq = np.array([bma/(2*k+1) for k in range(K+1)])
		self.normsq = normsq
		return normsq
	def plotBasis(self,x,K,normed=False):
		if normed == False: 
			output = self.Eval1dBasis(x,K)
		elif normed == True:
			output = self.Eval1dBasisNorm(x,K)
		[mpl.plot(Li) for Li in output]
		mpl.show()

class PolyFactory:
	# generates PolyBase class object
	@staticmethod
	def newPoly(polytype='Legendre'):
		L = LegPoly()
		return L

class scale:
	def __init__(self,X,a=None,b=None):
		self.X = np.atleast_2d(X)
		self.dim = self.X.shape[1] # each column is a dimension or feature
		if a is None and b is None:
			# default bounds
			self.a = -1*np.ones(self.dim)
			self.b =  1*np.ones(self.dim)
			self.__scaled = self.X # already scaled
		elif len(a)==0 and len(b)==0:
			self.a = -1*np.ones(self.dim)
			self.b =  1*np.ones(self.dim)
			self.__scaled = self.X # already scaled
		else:
			self.a = a
			self.b = b
			assert self.dim == len(a), "dim mismatch in a!"
			assert self.dim == len(b), "dim mismatch in b!"
			
			self.w = self.b - self.a
			assert self.X.shape[1] == len(self.w),"X and a,b mismatch!"
			# scale to -1 -> 1
			self.__scaled = 2*(self.X - self.a)/self.w - 1.0
		# integration factor ([a,b] -> [-1,1])
		# self.intf = np.prod(.5*(self.b - self.a))
		self.intf = np.prod(.5*(self.b - self.a)/(self.b - self.a)) # (b-a) canceled by prod prob weight
	def __getitem__(self,key):
		return self.__scaled[key]

class DomainScaler:
	def __init__(self,dim=None,a=None,b=None,domain_range=(-1,1)):
		if a is None:
			assert dim is not None, "Must defined dimension if a or b is None."
			self.dim = dim
		else:
			self.dim = len(a)
			assert len(a) == len(b), "Mismatch between a and b"
		self.a = a
		self.b = b
		self.dr_a,self.dr_b = domain_range
	def _compile(self):
		self.dr_w = self.dr_b - self.dr_a
		if self.a is None and self.b is None:
			# default bounds
			self.a = self.dr_a*np.ones(self.dim)
			self.b = self.dr_b*np.ones(self.dim)
		else:
			self.w = self.b - self.a
		# integration factor ([a,b] -> [-1,1])
		# self.intf = np.prod(.5*(self.b - self.a))
		self.intf = np.prod((1./self.dr_w)*(self.b - self.a)/(self.b - self.a)) # (b-a) canceled by prod prob weight
	def fit(self,X):
		self._compile() # get domain width
		X = np.atleast_2d(X)
		assert self.dim == X.shape[1], "Size of data matrix features does not match dimensions." 
		# scale to -1 -> 1
		# self.X_scaled_ = 2*(self.X - self.a)/self.w - 1.0
		return self
	def transform(self,X):
		X = np.atleast_2d(X)
		assert self.dim == X.shape[1], "Size of data matrix features does not match dimensions." 
		# scale to -1 -> 1
		X_scaled_ = self.dr_w*(X - self.a)/self.w + self.dr_a
		return X_scaled_
	def fit_transform(self,X):
		self.fit(X)
		return self.transform(X)
	def inverse_transform(self,Xhat):
		Xhat = np.atleast_2d(Xhat)
		assert self.dim == Xhat.shape[1], "Size of data matrix features does not match dimensions." 
		X = (1./self.dr_w)*(Xhat - self.dr_a) # [-1,1] -> [0,1]
		X = self.a + self.w*X
		return X

class MinMaxTargetScaler:
	'''
	Scales the entire target matrix with single scalar min and max. This way it is easy to invert for new predicted values.
	'''
	def __init__(self,target_range=(0,1)):
		self.a,self.b = target_range
	def fit(self,Y):
		self.min_ = np.amin(Y)
		self.max_ = np.amax(Y)
		self.w_ = self.max_ - self.min_
		self.ab_w_ = self.b - self.a
	def transform(self,Y):
		assert self.min_ is not None and self.max_ is not None, "Need to fit first."
		# scale to 0 -> 1 first
		Yhat = (Y - self.min_)/self.w_
		# scale to target_range
		Yhat = (Yhat - self.a)/self.ab_w_ 
		return Yhat
	def fit_transform(self,Y):
		self.fit(Y)
		return self.transform(Y)
	def inverse_transform(self,Yhat):
		# first transform back to [0,1]
		Y = Yhat*self.ab_w_ + self.a
		# back to true range
		Y = self.w_*Y  + self.min_
		return Y

class PCEBuilder(BaseEstimator):
	def __init__(self,order=1,customM=None,mindex_type='total_order',coef=None,a=None,b=None,polytype='Legendre'):
		# self.dim = dim # no need to initialize with dim
		self.order = order
		self.dim = None
		self.customM = customM
		self.mindex_type = mindex_type
		self.coef = coef
		# self.coef_ = self.coef
		self.polytype = polytype
		self.a = a # lower bounds on domain of x
		self.b = b # upper bound on domain of x
		self.normsq = np.array([])
		self.compile_flag = False
		self.mindex = None
	def compile(self,dim=None):
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
		if self.mindex.dtype != 'int':
			warnings.warn("Converting multindex array to integer array.")
			self.mindex = self.mindex.astype('int')
		self.nPCTerms = self.mindex.shape[0]
		if self.coef is not None: 
			assert len(self.coef) == self.mindex.shape[0], "coefficient array is not the same size as the multindex array."
		self.compile_flag = True
	def computeNormSq(self):
		mindex = self.mindex # use internal mindex array
		# compute multindex
		Kmax = np.amax(mindex,0)
		NormSq = []
		for d in range(self.dim):
			# Compute Legendre objects and eval 1d basis
			# Future work: replace this with Hermite
			# 	and polybase/poly factory class in order to 
			# 	generate mixed representations
			# 	E.g.: L = PolyFactory(PolyObj) for each dim
			#   will give a different poly object 
			assert self.polytype == 'Legendre', "Only works with Legendre for now!"
			L = PolyFactory.newPoly(self.polytype)
			NormSq.append(L.normsq(Kmax[d])) # norms up to K order
		# Multiply basis to eval nd polynomial
		normsq = NormSq[0][mindex[:,0]]
		# print(norms)
		for n in range(1,self.dim):
			norms_new = NormSq[n][mindex[:,n]]
			normsq = np.multiply(normsq,norms_new) # get norm squared of basis functions
		self.normsq = .5*normsq # norm squared * integration factor of 1/2 (since domain is [-1,1])
		return self.normsq
	def fit_transform(self,X):
		# compute multindex
		X = np.atleast_2d(X)
		if self.mindex is None:
			self.compile(dim=X.shape[1]) # only compiles once
		# evaluate basis at X which is n x dim
		# first compute max order for each dimension
		# basis is constructed based on multiindex
		if self.a is not None or self.b is not None: 
			assert len(self.a)==self.dim and len(self.b)==self.dim, "For user-specified bounds, they must be the same size as dimension."
		Xs = scale(X,a=self.a,b=self.b)  # scale to [-1,1]
		intf = Xs.intf
		self.intf = intf # save integration factor for ref
		X = np.atleast_2d(Xs[:]) # in case len(X) = 1
		Kmax = np.amax(self.mindex,0)
		P = []
		Px = []
		NormSq = []
		for d in range(self.dim):
			# Compute Legendre objects and eval 1d basis
			# Future work: replace this with Hermite
			# 	and polybase/poly factory class in order to 
			# 	generate mixed representations
			# 	E.g.: L = PolyFactory(PolyObj) for each dim
			#   will give a different poly object 
			assert self.polytype == 'Legendre', "Only works with Legendre for now!"
			L = PolyFactory.newPoly(self.polytype)
			P.append(L)
			Lix = L.Eval1dBasis(X[:,d],Kmax[d]) # each dimension may use different points so we need to loop
			# print(Lix.shape)
			Px.append(Lix) # list of size dim of Legendre polys
			NormSq.append(L.normsq(Kmax[d])) # norms up to K order
		# Multiply basis to eval nd polynomial
		Index = self.mindex
		Phi = Px[0] # get first group of Legendre basis polynomials
		Phi = Phi[:,Index[:,0]] # reorder according to multi-index for first dimension
		normsq = NormSq[0][Index[:,0]]
		# print(norms)
		for n in range(1,self.dim):
			Phi_temp = Px[n] # get next group of Leg basis polynomaisl
			Phi_new = Phi_temp[:,Index[:,n]] # order according to next multindex
			Phi = np.multiply(Phi,Phi_new) # multiple previous matrix
			norms_new = NormSq[n][Index[:,n]]
			normsq = np.multiply(normsq,norms_new) # get norm squared of basis functions
			# print(norms_new)

		# internally set Phi and norms for repeated use in eval
		self.Phi = Phi
		self.normsq = .5*normsq # norm squared
		return self.Phi
	def polyeval(self,X=None,c=None):
		if c is not None: 
			coef_ = c # use c in polyeval
			assert self.compile_flag == True, "Must compile to get multindex."
			assert len(coef_) == self.mindex.shape[0], "c is not the right size."
		elif c is None: 
			assert self.coef is not None, "Must define coefficient in constructor or polyeval"
			coef_ = self.coef # use coefficient from constructor

		if X is not None:
			Phi = self.fit_transform(X)
			yeval = np.dot(Phi,coef_)
		elif X is None:
			Phi = self.Phi # using stored Vandermonde matrix
			yeval = np.dot(Phi,coef_)

		return yeval
	def eval(self,X=None,c=None):
		return self.polyeval(X=None,c=None)
	def computeSobol(self,c=None):
		if self.mindex is None:
			self.compile(dim=self.dim)
		# assert self.compile_flag == True, "Must compile to set mindex. Avoids having to recompute multiindex everytime we run SA."
		if len(self.normsq) == 0: 
			normsq = self.computeNormSq()
		else:
			normsq = self.normsq # from fit transform
		# compute Sobol indices using coefficient vector
		if c is None:
			assert len(self.coef) != 0, "must specify coefficient array or feed it in the constructor."
			coef_ = self.coef
		else:
			coef_ = c

		assert len(coef_) == self.mindex.shape[0], "Coefficient vector must match the no of rows of the multiindex."

		totvar_vec = normsq[1:]*coef_[1:]**2
		totvar = np.sum(totvar_vec)
		# assert totvar > 0, "Coefficients are all zero!"
		S = []
		for i in range(self.dim):
			si = self.mindex[:,i] > 0 # boolean
			s = np.sum(totvar_vec[si[1:]])/totvar
			S.append(s)
		# in case elements have nan in them
		if np.all(coef_[1:] == 0): # ignore the mean
			print("Returning equal weights!")
			S = np.ones(self.dim)/self.dim # return equal weights
		return S

class pcereg(PCEBuilder,RegressorMixin):
	def __init__(self,order=2,customM=None,mindex_type='total_order',coef=None,a=None,b=None,polytype='Legendre',fit_type='linear',alphas=np.logspace(-12,1,20),l1_ratio=[.001,.5,.75,.95,.999,1],lasso_tol=1e-2):
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
		super().__init__(order=order,customM=customM,mindex_type=mindex_type,coef=coef,a=a,b=b,polytype=polytype)
	def compile(self,X):
		# build multindex
		self._dim = X.shape[1]
		super().compile(dim=self._dim) # use parent compile to produce the multiindex
		self._M = self.mindex
		self.Xhat = self.fit_transform(X)
		return self
	def fit(self,X,y):
		X,y = check_X_y(X,y)
		# get data attributes
		self._n,self._dim = X.shape
		self.compile(X) # build multindex and construct basis
		# pypce.PCEBuilder(dim=self.dim,self.order)
		if self.coef is not None:
			assert len(self.coef) == self._M.shape[0],"length of coefficient vector is not the same shape as the multindex!"
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