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





# In development
class pceseries():
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
# 	def __init__(self,X,Y,norm=True,xbounds=None,xlabels=None,npca=50,pca_datafile=None):
# 		self.xnsamples,self.d = X.shape
# 		self.ynsamples,self.n = Y.shape
# 		self.xlabels = xlabels
# 		self.pca_datafile = pca_datafile
# 		assert self.xnsamples == self.ynsamples, "Sample mismatch between X and Y!"
# 		# normalize X data
# 		if norm == True:
# 			assert xbounds is not None, "Must provide bounds on X. Otherwise, set norm=False"
# 			LB,UB = xbounds
# 			self.Xscaled = 2*( (X - LB) * 1./(UB-LB) ) - 1
# 			# normalize Y -> [0,1]
# 			self.Ymax = np.amax(Y)
# 			self.Ymin = np.amin(Y)
# 			self.Yscaled = (Y - self.Ymin)/(self.Ymax - self.Ymin)
# 		# perform PCA with npca components
# 		self.npca = npca
# 		assert self.npca >= 2, "Number of PCA should be > 1!"
# 		self.project_flag = False
# 	def run(self,full=True,load=False,save=False,mre_cutoff=1e-3,K=None,error=False):
# 		# if load == True, just load data
# 		if error == False: assert K is not None, "Must choose K PCA modes."
# 		if K is not None: self.npca = K # set npca to K
# 		if save == True: assert self.pca_datafile is not None, "datafile field must be specified."
# 		if full == True and load == False:
# 			self.performPCA(self.npca)
# 			if error == True: # only perform recon error if user defined
# 				if self.pca_datafile is None or save == False:
# 					"running PCA without saving data..." 
# 					self.performPCAerroranalysis(save=False)
# 				else:
# 					self.performPCAerroranalysis(save=True,filename=self.pca_datafile)
# 		elif load == True:
# 			assert self.pca_datafile is not None, "Must specify pca_datafile."
# 			if save == True: warnings.warn("save option is ignored.")
# 			self.loaddata(self.pca_datafile)
# 		self.project(K=K,mre_cutoff=mre_cutoff)
# 	def performPCA(self,npca):
# 		print("performing PCA analysis...")
# 		self.pca = PCA(n_components=self.npca,svd_solver='full')
# 		self.pca.fit(self.Yscaled)
# 		# cumulative sum of explained variance
# 		self.explained_variance = np.cumsum(self.pca.explained_variance_ratio_)
# 		# save data
# 		self.muY = self.Yscaled.mean(0)
# 		self.Y0 = self.Yscaled - self.muY
# 		self.V = self.pca.components_.T # projection basis
# 		self.P = np.dot(self.Y0,self.V) # projection coefficients
# 		assert self.V.shape[0] == self.n, "Basis dimension should be same dimension as each Y sample."
# 		assert self.P.shape == (self.ynsamples,self.npca), "Shape of projection coefficients is not right!"
# 	def performPCAerroranalysis(self,save=True,filename=None):
# 		# calculate relative error of projections for increasing npca up to self.npca
# 		self.__mre = [] # mean relative reconstruction error
# 		for k in tqdm(range(2,self.npca)):
# 			Y0approx = np.dot(self.V[:,:k],self.P[:,:k].T).T
# 			RE = [] # relative error for each sample
# 			for i,y0 in enumerate(self.Y0):
# 				re = np.linalg.norm(y0 - Y0approx[i])/np.linalg.norm(self.muY + y0)
# 				RE.append(re)
# 			muRE = np.mean(np.array(RE))
# 			# print("Mean relative errors", np.mean(RE))
# 			self.__mre.append(muRE) # average over all samples
# 			self.mre = np.array(self.__mre)
# 		if save == True:
# 			assert filename is not None, "Must specify filename to save pca dictionary data."
# 			# save data as dictionary
# 			datadict = {}
# 			datadict['V'] = self.V # projection basis
# 			datadict['P'] = self.P # projection coefficients
# 			datadict['Y0'] = self.Y0 # centered data
# 			datadict['muY'] = self.muY # mean of original data
# 			datadict['mre'] = self.mre # reconstruction relative error
# 			datadict['expvar'] = self.explained_variance # cumulative explained variance
# 			datadict['Xscaled'] = self.Xscaled # normalized data to [-1,1]
# 			datadict['xlabels'] = self.xlabels # labels of x data
# 			print("saving PCA data...")
# 			with open(filename + '.pkl', 'wb') as pfile:
# 			    pickle.dump(datadict, pfile)
# 	def loaddata(self,filename):
# 		# import data dictionary with PCA modes and projections
# 		try:
# 			with open(filename, 'rb') as pfile:
# 				datadict = pickle.load(pfile)
# 		except:
# 			with open(filename + '.pkl', 'rb') as pfile:
# 				datadict = pickle.load(pfile)
# 		print("loading PCA data...")
# 		self.V = datadict['V'] # projection basis
# 		self.P = datadict['P'] # projection coefficients
# 		self.Y0 = datadict['Y0'] # centered data
# 		self.muY = datadict['muY'] # mean of original data
# 		self.mre = datadict['mre'] # reconstruction relative error
# 		self.explained_variance = datadict['expvar'] # cumulative explained variance
# 		self.Xscaled = datadict['Xscaled'] # normalized data to [-1,1]
# 		self.xlabels = datadict['xlabels'] # labels of x data
# 	def project(self,K=None,mre_cutoff=1e-3):
# 		if K is not None:
# 			self.K = K # use K if user defined
# 		else:
# 			# find PCA mode where recon error is greater than cutoff
# 			cutoff_indices = np.where(self.mre > mre_cutoff)[0]
# 			if len(cutoff_indices) == 0:
# 				warnings.warn("cutoff is too large. try decreasing. setting K = 1")
# 				self.K = 1 # only use 1 PCA mode
# 			else:
# 				self.K = cutoff_indices[-1] + 1 # add one since we are starting with 1 after the mean

# 		# get projections and basis columns
# 		self.Vk = self.V[:,:self.K]
# 		self.Pk = self.P[:,:self.K] # each row is the proj coef
# 		self.project_flag = True
# 	def invert(self,P):
# 		P = np.atleast_2d(P.copy())
# 		assert P.shape[1] == S.K, "dimension mismatch in P and V."
# 		R = np.dot(self.Vk,P.T)
# 		return R.T
# 	def pcefit(self):
# 		assert self.project_flag == True, "Must project data onto K PCA modes first. "
# 		npce = self.Vk.shape[1]


# # create function to plot sobol indices in a stacked graph
# # to be added to the pypce pceseries class
# C = []
# M = []
# Sobol = []
# t = []
# for pce in PCEs:
# 	C.append(pce['c'])
# 	M.append(pce['M'])
# 	t.append(pce['pca_mode'])
# 	Sobol.append(pce['S'])

# xlabels = np.array(['dens_sc', 'vel_sc', 'temp_sc', 
# 					'sigk1', 'sigk2', 'sigw1', 'sigw2',
# 					'beta_s', 'kap', 'a1', 'beta1r', 'beta2r'])
# pceS = pypce.pceseries(dim=xdim,C=C,MIs=M)

# Sobol = np.array(Sobol)
# # S = np.random.rand(4,12)
# normS = np.array([s/np.sum(s) for s in Sobol]).T

# import matplotlib.pyplot as plt
# import matplotlib._color_data as mcd

# xkcd_colors = []
# xkcd = {name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS}
# for j, n in enumerate(xkcd):
# 	xkcd = mcd.XKCD_COLORS["xkcd:" + n].upper()
# 	xkcd_colors.append(xkcd)

# Ps = []
# bottom = np.zeros(len(t))
# import seaborn as sns
# sns.set_palette(sns.color_palette("Paired", 12))
# plt.figure(figsize=(20,9))
# for ii in range(xdim):
# 	ptemp = plt.bar(t,normS[ii],bottom=bottom,width=.25)
# 	bottom = bottom + normS[ii] # reset bottom to new height
# 	Ps.append(ptemp)
# plt.ylabel('Sobol Scores')
# plt.ylim([0,1.1])
# # plt.title('Sobol indices by pca mode')
# plt.xticks(t, ('PCA1','PCA2','PCA3','PCA4','PCA5','PCA6'))
# plt.legend(((p[0] for p in Ps)), (l for l in xlabels),
# 			fancybox=True, shadow=True,
# 			loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=xdim)

# # plot explained variance
# plt.plot(t,S.explained_variance,'--ok')
# plt.savefig('figs/SA_'+y_label+'.png')