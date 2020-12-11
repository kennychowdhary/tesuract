"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
	Examples can be given using either the ``Example`` or ``Examples``
	sections. Sections support any reStructuredText formatting, including
	literal blocks::

		$ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
	module_level_variable1 (int): Module level variables may be documented in
		either the ``Attributes`` section of the module docstring, or in an
		inline docstring immediately following the variable.

		Either form is acceptable, but the two should not be mixed. Choose
		one convention to document module level variables and be consistent
		with it.

Todo:
	* For module TODOs
	* You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html

"""

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

class DomainScaler_old:
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

# domain test
class DomainScaler:
	def __init__(self,dim,input_range,output_range=(-1,1)):
		self.dim = dim
		self.input_range = input_range
		self.output_range = output_range
	def _get_bound_list(self,input_range):
		if isinstance(input_range,list):
			assert len(input_range) == self.dim, "input range must be a list of tuples"
			input_bounds = input_range
		if isinstance(input_range,tuple):
			input_bounds = [(input_range[0],input_range[1]) for i in range(self.dim)]
		a = np.array([ab[0] for ab in input_bounds]) # lower bounds
		b = np.array([ab[1] for ab in input_bounds]) # upper bounds
		return input_bounds,a,b
	def _range_check(self,X,B):
		if X.ndim == 1:
			X = np.atleast_2d(X)
		n,d = X.shape
		assert d == self.dim, "columns of X must be the same as dimensions"
		assert len(B) == self.dim, "length of bounds list must be same as dimensions"
		dim_check = [(X[:,i] >= B[i][0]).all() and (X[:,i] <= B[i][1]).all() for i in range(d)]
		assert all(dim_check), "X is not in the range of the input range."
		return X
	def fit_transform(self,X):
		self.input_bounds,a,b = self._get_bound_list(self.input_range)
		self.output_bounds,c,d = self._get_bound_list(self.output_range)
		X = self._range_check(X,self.input_bounds)
		# transform to [0,1] first for ease
		X_unit_scaled = (X - a)/(b-a)
		# transform to output bounds
		X_scaled = (d-c)*X_unit_scaled + c
		X_scaled = self._range_check(X_scaled,self.output_bounds)
		return X_scaled
	def inverse_transform(self,Xhat):
		self.input_bounds,a,b = self._get_bound_list(self.input_range)
		self.output_bounds,c,d = self._get_bound_list(self.output_range)
		Xhat = self._range_check(Xhat,self.output_bounds)
		Xhat_unit_scaled = (Xhat - c)/(d-c)
		X_inv = (b-a)*Xhat_unit_scaled + a
		X_inv = self._range_check(X_inv,self.input_bounds)
		return X_inv 

class MinMaxTargetScaler:
	"""Custom target scaler which uses one min and max for all elements
 
		Description:

	A Transformer class that scales and shifts by the min/max

	This transformer transforms the target Y by the absolute max and minimum to
	the unit hypercube, i.e., [0,1]^dim.  It does not scale each target column
	differently as in sklearn's preprocessing toolbox. The transform is simply 

	Y = (max(Y)-min(Y))*Yhat + min(Y)

	Todo:
		To do list
 
	Note:
		This is just a test
  
	Parameters:
		min_ (float):	minimum of Y (over all columns and rows)
		max_ (float):	maximum of Y (over all columns and rows)
		w_ (float):		diff btwn min and max of Y (over all columns and rows)
		ab_w_ (float):		width of target range 
	
	Attributes:
		min_ (float):	minimum of Y (over all columns and rows)
		max_ (float):	maximum of Y (over all columns and rows)
		w_ (float):		diff btwn min and max of Y (over all columns and rows)
		ab_w_ (float):		width of target range 

	Returns:
		The return value. 
  
	Raises:
		Attribute Errors
  
	Examples:
		Examples should be written in doctest format

		>>> print("Hello World!")
	"""
	def __init__(self,target_range=(0,1)):
		"""This is a description of the constructor"""
		self.a,self.b = target_range
	def fit(self,Y,target=None):
		"""
		Fit method

		This function compute the min, max and width, which are used in the above
		transform and inverse transform formulas.  

		Parameters
		----------
		Y : numpy.ndarray, required	
			ndarray of shape (n_samples ,n_features)
			This is the (possibly multi) target output. 
			
		"""
		self.min_ = np.amin(Y)
		self.max_ = np.amax(Y)
		self.w_ = self.max_ - self.min_
		self.ab_w_ = self.b - self.a
	def transform(self,Y):
		assert self.min_ is not None and self.max_ is not None, "Need to fit first."
		# scale to 0 -> 1 first
		Yhat = (Y - self.min_)/self.w_
		# scale to target_range
		Yhat = Yhat*self.ab_w_ + self.a 
		return Yhat
	def fit_transform(self,Y,target=None):
		self.fit(Y)
		return self.transform(Y)
	def inverse_transform(self,Yhat):
		# first transform back to [0,1]
		# Y = (Yhat - self.a)/self.ab_w_
		# back to true range
		Y = (self.w_/self.ab_w_)*(Yhat - self.a)  + self.min_
		return Y

