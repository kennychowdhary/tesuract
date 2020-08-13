import numpy as np
import warnings, pdb 

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
		# self.intf = np.prod(.5*(self.b - self.a)/(self.b - self.a)) # (b-a) canceled by prod prob weight
	def __getitem__(self,key):
		return self.__scaled[key]

class DomainScaler:
	# need to add checks and add functionality
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
		# self.intf = np.prod((1./self.dr_w)*(self.b - self.a)/(self.b - self.a)) # (b-a) canceled by prod prob weight
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
		X = (1./self.dr_w)*(Xhat - self.dr_a) 
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

