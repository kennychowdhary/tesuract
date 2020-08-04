import numpy as np
import matplotlib.pyplot as mpl
import pdb, ipdb, warnings, pickle
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.hermite_e import hermegauss
import itertools
from scipy.special import comb

class QuadBase:
	def __init__(self,nquad):
		self.nquad = nquad
	def get1dQuad(self):
		pass

class LegendreQuad(QuadBase):
	def __init__(self,nquad=2):
		super().__init__(nquad)
	def get1dQuad(self,nquad=None):
		if nquad is not None: self.nquad = nquad
		x,w = leggauss(self.nquad)
		return x,w

class HermiteQuad(QuadBase):
	''' normalized'''
	def __init__(self,nquad=2):
		super().__init__(nquad)
	def get1dQuad(self,nquad=None):
		if nquad is not None: self.nquad = nquad
		x,w = hermegauss(self.nquad)
		return x,w # add a factor of (2*np.pi)**-.5 to normalize each dimension

class ClenshawCurtis(QuadBase):
	def __init__(self,nquad=2):
		super().__init__(nquad)
	def _get1dQuad(self,nquad=None):
		''' old '''
		if nquad is not None: self.nquad = nquad
		if self.nquad == 1:
			return np.array([0.0]), np.array([2.0])
		else:
			n = self.nquad
			x = np.cos(np.pi*(n-1-np.arange(n))/(n-1))
			w = np.ones(len(x))
			for i in range(n):
				theta = i*np.pi/(n-1)
				for j in range(1,int(.5*(n-1)+1)):
					if 2*j == n-1:
						f = 1.0
					else:
						f = 2.0
					# print(i,j,f)
					w[i] -= f*np.cos(2.0*j*theta)/(4.0*j**2-1)
			w[0] /= n-1
			w[1:-1] = 2*w[1:-1]/(n-1)
			w[-1] *= 1.0/(n-1)
			return x,w
	def get1dQuad(self,nquad):
		'''from chaospy'''
		degree = self.nquad
		n = self.nquad
		points = -np.cos((np.pi * np.arange(n)) / (n - 1))
		if n == 2:
			weights = np.array([1.0, 1.0])
		else:
			n -= 1
			N = np.arange(1, n, 2)
			length = len(N)
			m = n - length
			v0 = np.concatenate(
				[2.0 / N / (N - 2), np.array([1.0 / N[-1]]), np.zeros(m)]
			)
			v2 = -v0[:-1] - v0[:0:-1]
			g0 = -np.ones(n)
			g0[length] += n
			g0[m] += n
			g = g0 / (n ** 2 - 1 + (n % 2))

			w = np.fft.ihfft(v2 + g)
			assert max(w.imag) < 1.0e-15
			w = w.real

			if n % 2 == 1:
				weights = np.concatenate([w, w[::-1]])
			else:
				weights = np.concatenate([w, w[len(w) - 2 :: -1]])
		return points, weights

class QuadFactory:
	# generates QuadBase class object
	@staticmethod
	def newQuad(quadtype='Legendre'):
		if quadtype == 'Legendre':
			Q = LegendreQuad()
		if quadtype == 'ClenshawCurtis':
			Q = ClenshawCurtis()
		if quadtype == 'Hermite':
			Q = HermiteQuad()
		return Q

class QuadRule:
	def __init__(self,x,w):
		self.x = x
		self.w = w
		self.n = len(w)
		if x.ndim == 1:
			self.dim = 1
			self.x = np.atleast_2d(x).T # col vector
		else:
			self.dim = x.shape[1]
		assert len(x) == len(w), "x and w dont habe the same # of points"
	def __add__(self,other):
		assert self.dim == other.dim, "Dimensions do not match!"
		xnew = np.vstack([self.x,other.x])
		wnew = np.hstack([self.w,other.w])
		Qnew = QuadRule(xnew,wnew)
		return Qnew
	def __sub__(self,other):
		assert self.dim == other.dim, "Dimensions do not match!"
		xnew = np.vstack([self.x,other.x])
		wnew = np.hstack([self.w,-1*other.w])
		Qnew = QuadRule(xnew,wnew)
		return Qnew
	def __mul__(self,other):
		# tensor product
		index_comb = list(itertools.product(range(self.n),range(other.n)))
		xnew = [np.concatenate([self.x[i[0]],other.x[i[1]]]) for i in index_comb]
		wnew = [self.w[i[0]]*other.w[i[1]] for i in index_comb]
		Qnew = QuadRule(np.array(xnew),np.array(wnew))
		return Qnew
	def copy(self):
		return QuadRule(self.x,self.w)


class QuadOps:
	@staticmethod
	def getMultiIndexLevel(level,ndim):
		''' returns the multindices of order = level'''
		iup = 0
		nup_level = int(comb(ndim+level-1,level))
		M = np.zeros((nup_level,ndim))
		if ndim == 1:
			M[0,0] = level
		else:
			for first in range(level,-1,-1):
				theRest = QuadOps.getMultiIndexLevel(level-first,ndim-1)

				for j in range(len(theRest)):
					# print(iup,j)
					M[iup,0] = first
					M[iup,1:ndim] = theRest[j,0:ndim-1]
					iup += 1

		return M
	@staticmethod
	def compressRule(Q):
		# assert self.rule_ is not None, "Must set rule first."
		# convert numpy array to list of tuples
		xtuple = [tuple(xi) for xi in Q.x]
		# create a dictionary 
		from collections import defaultdict
		dd = defaultdict(list)
		for ii,xi in enumerate(xtuple):
			dd[xi].append(Q.w[ii])
		# sum weights over keys
		for key in dd:
			dd[key] = np.sum(dd[key])
		x = np.array(list(dd.keys()))
		w = np.array([dd[key] for key in dd])
		x = x[np.abs(w) > 1e-12]
		w = w[np.abs(w) > 1e-12]
		return QuadRule(x,w)


class QuadBuilder():
	def __init__(self,grid_type='sparse',order=2,quad_type='Legendre'):
		self.grid_type = grid_type
		self.quad_type = quad_type
		self.order = order
		self.ndim = None
		self.growth_rule = None
	def SetRule(self,ndim):
		self.ndim = ndim
		if self.grid_type == 'full':
			self._full()
		if self.grid_type == 'sparse':
			if self.quad_type == 'Legendre': 
				self.growth_rule = 0
			self._sparse()
		return self
	def _full(self):
		# cannot do mixed quad yet. Easy if quad type takes in array
		quad_gen = QuadFactory.newQuad(self.quad_type)
		x,w = quad_gen.get1dQuad(nquad=self.order+1) # 0th order means 1 point
		q1d = QuadRule(x,w)
		for i in range(1,self.ndim):
			q1d *= q1d
		q = q1d
		self.rule_ = q
	def _sparse(self):
		for nlevel in range(-1,self.order):
			self._SetNextLevel2(nlevel)
	def _SetNextLevel(self,nlevel):
		nlevel += 1
		M = QuadOps.getMultiIndexLevel(nlevel,self.ndim)
		nM = M.shape[0]
		M_npts = np.zeros((nM,self.ndim))
		quad_gen = QuadFactory.newQuad(self.quad_type)
		for j in range(nM):
			Mj = M[j] # jth row of the multiindexlevel
			# 1 if Mj == 0, 3 if 1, else (Mj_i+1)^2
			if self.growth_rule == 0:
				npts = 1*(Mj == 0) + 3*(Mj == 1) + ((Mj)**2+1)*(Mj > 1)
				npts_1 = 0*(Mj == 0) + 1*(Mj == 1) + ((Mj-1)**2+1)*(Mj > 1)
			elif self.growth_rule == 1:
				npts = ((Mj+1)**2-1)*(Mj > 1)
				npts_1 = (Mj**2-1)*(Mj > 1)
			npts = npts.astype('int')
			npts_1 = npts_1.astype('int')
			print(npts,npts_1)
			xw = [quad_gen.get1dQuad(nquad=int(n)) for n in list(npts)]
			rules = [QuadRule(xwi[0],xwi[1]) for xwi in xw]
			xw_1 = [quad_gen.get1dQuad(nquad=int(n_1)) for n_1 in list(npts_1)]
			rules_1 = [QuadRule(xwi_1[0],xwi_1[1]) for xwi_1 in xw_1]
			srules = []
			for ii in range(len(npts)):
				if npts_1[ii] > 0:
					srules.append(rules[ii] - rules_1[ii])
				else:
					srules.append(rules[ii])

			# multiply rules in srules
			r = srules[0].copy()
			for ri in srules[1:]:
				r = r*ri

			if j == 0:
				rule_level = r.copy()
			else:
				rule_level = r + rule_cur
			rule_cur = rule_level.copy()
		# pdb.set_trace()
		if nlevel == 0:
			rule_total = rule_level.copy()
		else:
			rule_total = self.rule_ + rule_level
		self.rule_ = rule_total.copy()
		return self
	def _SetNextLevel2(self,nlevel):
		nlevel += 1
		M = QuadOps.getMultiIndexLevel(nlevel,self.ndim)
		self.M = M
		# nM = M.shape[0]
		M_npts = np.zeros(M.shape)
		quad_gen = QuadFactory.newQuad(self.quad_type)
		for j in range(len(M)):
			rules = []
			rules_1 = []
			srules = []
			Mj = M[j]
			for id in range(self.ndim):
				if M[j,id] == 0:
					npts = 1
					npts_1 = 0
				elif M[j,id] == 1:
					npts = 3
					npts_1 = 1
				else:
					npts = int((M[j,id])**2) + 1
					npts_1 = int((M[j,id]-1)**2)+1

				x,w = quad_gen.get1dQuad(nquad=npts)
				rule = QuadRule(x,w)
				if npts_1 > 0:
					x_1,w_1 = quad_gen.get1dQuad(nquad=npts_1)
					rule_1 = QuadRule(x_1,w_1)
					srule = rule - rule_1
				else:
					srule = rule.copy()
				srules.append(srule)
				# end of id iterator
			# multiple rule
			rule_temp = srules[0].copy()
			for s in srules[1:]:
				rule_temp = rule_temp * s
			# rule_temp = srules[0]*srules[1]
			if j == 0:
				rule_level = rule_temp.copy()
			else:
				rule_level = rule_cur + rule_temp
			# pdb.set_trace()
			rule_cur = rule_level.copy()

			# end of j iterator

		if nlevel == 0:
			rule_total = rule_level.copy()
		else:
			rule_total = self.rule_ + rule_level
		self.rule_ = rule_total.copy()
		# test = np.unique(self.rule_.x,axis=0)
		# ind = []
		# for i in range(len(test)):
		# 	ind = np.array_equal(test[0],self.rule_.x[i])
		# 	if ind == True:
		# 		print(self.rule_.w[i])
		# ipdb.set_trace()
		self.rule_ = QuadOps.compressRule(self.rule_)
		return self


quad = QuadBuilder(order=5,grid_type='sparse')
# quad.SetRule(ndim=2)
# rule = quad.rule_
# # rule = QuadOps.compressRule(quad.rule_)

# L = LegendreQuad()
# x,w = L.get1dQuad(8)
# rn = np.random.RandomState(123)
# X = 2*rn.rand(len(w),2)-1
# W = 4./X.shape[0] + 0*w

# q1 = QuadRule(x,w)
# q2 = QuadRule(x,w)
# Q1 = QuadRule(X,W)
# Q2 = QuadRule(X,W)
# q = q1+q2
# Q = Q1+Q2

# M = QuadOps.getMultiIndexLevel(level=3,ndim=2)

# test chaospy
def construct_lookup(
		orders,
		dists,
		rules='gaussian',
		accuracy=100,
		growth=None,
		recurrence_algorithm="",
):
	"""
	Create abscissas and weights look-up table so values do not need to be
	re-calculatated on the fly.
	"""
	# from .frontend import generate_quadrature
	if isinstance(rules, str):
		rules = (rules,)*len(dists)
	if isinstance(orders,int):
		orders = orders*np.ones(len(dists), dtype=int)
	x_lookup = []
	w_lookup = []
	if rules[0] == "gaussian":
		Q = QuadFactory.newQuad("Legendre")
	if rules[0] == "clenshaw_curtis":
		Q = QuadFactory.newQuad("ClenshawCurtis")
	# if rules[0] == 'hermite':
		# Q = QuadFactory.newQuad("Hermite")
	for max_order, dist, rule in zip(orders, dists, rules):
		x_lookup.append([])
		w_lookup.append([])
		for order in range(max_order+1):
			# abscissas, weights = Q.get1dQuad(order+1)
			(abscissas,), weights = generate_quadrature(
				order,
				dist,
				accuracy=100,
				rule=rule,
				growth=None
			)
			# err1 = np.linalg.norm(abscissas - abscissas0)
			# err2 = np.linalg.norm(weights - weights0)
			# print(err1,err2)
			# print(weights,weights0)
			x_lookup[-1].append(abscissas)
			w_lookup[-1].append(weights)
	return x_lookup, w_lookup

def construct_collection(
		orders,
		dists,
		x_lookup,
		w_lookup,
):
	"""Create a collection of {abscissa: weight} key-value pairs."""
	if isinstance(orders,int):
		orders = orders*np.ones(len(dists), dtype=int)
	order = np.min(orders)
	skew = orders-order

	# Indices and coefficients used in the calculations
	# indices = chaospy.numpoly.glexindex(
	# 	order-len(dists)+1, order+1, dimensions=len(dists))
	mi = []
	for ilevel in range(order-len(dists)+1,order+1):
		mi.append(QuadOps.getMultiIndexLevel(ilevel,len(dists)))
	indices = np.vstack(mi).astype('int')
	coeffs = np.sum(indices, -1)
	coeffs = (2*((order-coeffs+1) % 2)-1)*comb(len(dists)-1, order-coeffs)

	collection = defaultdict(float)
	for bidx, coeff in zip(indices+skew, coeffs.tolist()):
		abscissas = [value[idx] for idx, value in zip(bidx, x_lookup)]
		weights = [value[idx] for idx, value in zip(bidx, w_lookup)]
		for abscissa, weight in zip(product(*abscissas), product(*weights)):
			collection[abscissa] += np.prod(weight)*coeff

	return collection

import chaospy
from chaospy import generate_quadrature
from collections import defaultdict
from itertools import product

# dists = [chaospy.Uniform(-1, 1),chaospy.Uniform(-1, 1),chaospy.Uniform(-1, 1),chaospy.Uniform(-1, 1),chaospy.Uniform(-1, 1),chaospy.Uniform(-1, 1),chaospy.Uniform(-1, 1),chaospy.Uniform(-1, 1),chaospy.Uniform(-1, 1),chaospy.Uniform(-1, 1)]
dists = [chaospy.Uniform(-1, 1),chaospy.Uniform(-1, 1)]
distribution = chaospy.J(*dists)
# distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Normal(0, 1))    
order = 5
# rule = "gaussian"
rule = "clenshaw_curtis"
# rule = "gaussian"
x_lookup, w_lookup = construct_lookup(orders=order,dists=distribution,rules=rule)
collection = construct_collection(orders=order,dists=distribution,x_lookup=x_lookup,w_lookup=w_lookup)

x = sorted(collection)
w = np.array([collection[key] for key in x])
x = np.array(x).T
print(np.sum(w))









