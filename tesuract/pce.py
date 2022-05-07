import numpy as np
import pdb, warnings, pickle, time

# from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from .multiindex import RecursiveHypMultiIndex
from .multiindex import MultiIndex
from .preprocessing import DomainScaler

from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn import linear_model

from itertools import combinations

"""
Acknowledgement:
The PCE functionality was inspired by my work on UQTk (https://github.com/sandialabs/UQTk). 
All Python code is original. 
"""

# from numba import jit

# Polynomials classes
class PolyBase:
    def __init__(self):
        pass

    def Eval1dBasis(self, x):
        # virtual fun to compute basis values at x up to order K
        pass

    def Eval1dBasisNorm(self, x):
        # virtual fun to compute norm basis values at x up to order K
        pass


def Leg1dPoly(order, x):
    if order == 0:
        if x.ndim == 0:
            y = 1.0
        else:
            y = np.ones(len(x))
    elif order == 1:
        y = x
    elif order == 2:
        y = 0.5 * (3.0 * x**2 - 1)
    elif order == 3:
        y = 0.5 * (5.0 * x**3 - 3 * x)
    elif order == 4:
        y = (1.0 / 8) * (35.0 * x**4 - 30.0 * x**2 + 3.0)
    elif order == 5:
        y = (1.0 / 8) * (63.0 * x**5 - 70.0 * x**3 + 15.0 * x)
    elif order == 6:
        y = (1.0 / 16) * (231.0 * x**6 - 31.0 * x**4 + 105.0 * x**2 - 5.0)
    return y


class LegPoly2(PolyBase):
    """
    Usage for 1d polynomial basis construction
    L = LegPoly()
    x = np.linspace(-1,1,100)
    L.plotBasis(x,K,normed=True)
    """

    def __init__(self, domain=[-1.0, 1.0]):
        self.domain = domain
        self.a = domain[0]
        self.b = domain[1]
        super().__init__()

    def Eval1dBasis(self, x, K):
        # returns matrix where each column is Li(x)
        self.K = K
        # assert jnp.all(x>=self.a) and jnp.all(x<=self.b), "x is not in the domain."
        # transform x to [-1,1]
        x0 = 2.0 * (x - 1.0 * self.a) / (1.0 * self.b - self.a) - 1.0
        # assert jnp.all(x0>=-1) and jnp.all(x0<=1), "x is not in the domain."
        self.Lis = [Leg1dPoly(k, x0) for k in range(self.K + 1)]
        return np.array(self.Lis)  # dim x nx

    def normsq(self, K):
        # compute the squared norm of each basis up to K
        bma = self.domain[1] - self.domain[0]
        normsq = np.array([bma / (2 * k + 1) for k in range(K + 1)])
        self.normsq = normsq
        return normsq


from numpy.polynomial.legendre import Legendre


class LegPoly(PolyBase):
    """
    Usage for 1d polynomial basis construction
    L = LegPoly()
    x = np.linspace(-1,1,100)
    L.plotBasis(x,K,normed=True)
    """

    def __init__(self, domain=np.array([-1, 1])):
        self.domain = domain
        super().__init__()

    def Eval1dBasis(self, x, K):
        # returns matrix where each column is Li(x)
        self.K = K
        self.Lis = [Legendre.basis(k, self.domain) for k in range(self.K + 1)]
        return np.array([Li(x) for Li in self.Lis])

    def normsq(self, K):
        # compute the squared norm of each basis up to K
        bma = self.domain[1] - self.domain[0]
        normsq = np.array([bma / (2 * k + 1) for k in range(K + 1)])
        self.normsq = normsq
        return normsq


class LegPolyNorm(PolyBase):
    """
    Usage for 1d polynomial basis construction
    L = LegPoly()
    x = np.linspace(-1,1,100)
    L.plotBasis(x,K,normed=True)
    """

    def __init__(self, domain=np.array([-1, 1])):
        self.domain = domain
        self.a = domain[0]
        self.b = domain[1]
        super().__init__()

    def Eval1dBasis(self, x, K):
        # returns matrix where each column is Li(x)
        self.K = K
        # compute norms
        normsq = np.array([(self.b - self.a) / (2 * k + 1) for k in range(K + 1)])
        self.Lis = [Legendre.basis(k, self.domain) for k in range(self.K + 1)]
        return np.array([Li(x) / np.sqrt(normsq[i]) for i, Li in enumerate(self.Lis)])

    def normsq(self, K):
        # compute the squared norm of each basis up to K
        bma = self.domain[1] - self.domain[0]
        normsq = np.array([bma / (2 * k + 1) for k in range(K + 1)])
        self.normsq = 1.0 + 0 * normsq
        return normsq


class PolyFactory:
    # generates PolyBase class object
    @staticmethod
    def newPoly(polytype="Leg"):
        L = LegPoly()
        return L

    def newPoly(polytype="LegNorm"):
        L = LegPolyNorm()
        return L


class PCEBuilder(TransformerMixin):
    """Base class for building a multivariate polynomial basis.

    This class creates a multi-variate polynomial object, aka as a polynomial
    chaos model. The expansion looks like

    .. math::

        \sum_{i=1}^N c_i \Phi_i(\mathbf{x})

    where :math:`N` is the number of polynomial terms, and :math:`\Phi_i:\mathbf{x} \in \mathbb{R}^d \mapsto \mathbb{R}` is the multivariate basis function which takes the form

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

        Description of the order of the polynomials in the expansion. For total order, the order is the maximum polynomial order for each basis function per dimension.

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

    nPCTerms : int

        The number of basis terms of the multi-variate polynomial.

    coef    : ndarray of shape (nbasis,)

        Coefficient array

    Methods
    -------
    compile:
        pre-processing for fit, e.g., multiindex generation
    computeMoments:
        compute means and variances using PCE coefficients


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
    >>> from tesuract import PCEBuilder
    >>> p = PCEBuilder(order=3,normalized=True)
    >>> print(p.mindex)
    """

    def __init__(
        self,
        order=1,
        customM=None,
        mindex_type="total_order",
        coef=None,
        a=None,
        b=None,
        polytype="Legendre",
        normalized=False,
        store_phi=False,
        input_range=None,
        use_sklearn_poly_features=False,
    ):
        # self.dim = dim # no need to initialize with dim
        self.order = order
        self.dim = None
        self.customM = customM
        self.mindex_type = mindex_type
        self.coef = coef
        # self.coef_ = self.coef
        self.polytype = polytype
        self.a = a  # lower bounds on domain of x
        self.b = b  # upper bound on domain of x
        self.normsq = np.array([])
        self.compile_flag = False
        self.mindex = None
        self.input_range = input_range
        self.normalized = normalized
        self._mindex_compute_count_ = 0
        self.store_phi = store_phi
        self.use_sklearn_poly_features = use_sklearn_poly_features

    def compile(self, dim):
        """Setup for instantiating the basis class

        Constructs the multi-dimensional multi-index which defines the
        polynomial basis elements. Note that this is only done once during the
        fit method of the class, unless the mindex variable is undefined.

        The multi-index array is of size :math:`N \\times dim` where :math:`N`
        is the number of basis elements. The multi-index determine the order or
        degree of the univariate polynomial in

        .. math::

            \Phi_i = \prod_{j=1}^d L_{\\alpha_j^{(i)}}(x_i).

        So for example, :math:`\\alpha_j=2` corresponds to a second order or
        quadratic Legendre polynomial.

        Lastly, this method also defines the number of basis elements.

        Parameters
        ----------

        dim : int

            dimension of the polynomials

        Returns
        -------

        self: object

            Returns the object itself.

        """
        self.dim = dim
        # constructor for different multindices (None, object, and array)
        ##############################################################
        # Note that sklearn base cv search clones each estimator so that the multi-index is created for each cv sub score
        # Here we recompute the multiindex only if mindex is None or if the X dimension does not match the already existing mindex
        if self.customM is None:
            # only place where the multindex is computed!
            if self.mindex is None:
                # compute when mindex is None
                self.M = MultiIndex(dim, self.order, self.mindex_type)
                self._mindex_compute_count_ += 1
                # print(self.M.index.shape)
            elif self.mindex.shape[1] == self.dim:
                # don't recompute if dimension stays the same
                pass
            else:
                # recompute when dim changes
                self.M = MultiIndex(dim, self.order, self.mindex_type)
                self._mindex_compute_count_ += 1
        ##############################################################
        elif isinstance(self.customM, MultiIndex):
            # print("Using custom Multiindex object.")
            assert self.customM.dim == dim
            self.M = customM
        elif isinstance(self.customM, np.ndarray):
            self.dim = self.customM.shape[1]  # set dim to match multiindex
            self.M = MultiIndex(
                self.dim, order=1, mindex_type="total_order"
            )  # setup default
            self.M.setIndex(self.customM)
            self.order = None  # leave blank to indicate custom order
        self.mindex = self.M.index
        self.multiindex = self.mindex
        if self.mindex.dtype != "int":
            warnings.warn("Converting multindex array to integer array.")
            self.mindex = self.mindex.astype("int")
        self.nPCTerms = self.mindex.shape[0]
        # if self.coef is not None:
        #     assert len(self.coef) == self.mindex.shape[0], "coefficient array is not the same size as the multindex array."
        self.compile_flag = True

    def _fit_transform_sklearn_poly_features(self, X):
        """
        Compile the polynomial feature matrix using the sklearn.PolynomialFeatures class.
        """
        if not hasattr(self, "polyT"):
            self.polyT = PolynomialFeatures(self.order, include_bias=True)
        return self.polyT.fit_transform(X)

    def computeNormSq(self):
        """Separate method to compute norms, in order to bypass construct basis. For use when computing the feature importance/ Sobol sensitivity indices."""
        mindex = self.mindex  # use internal mindex array
        # L = []
        Kmax = np.amax(self.mindex, axis=0)
        NormSq = []
        for i in range(self.dim):
            # Compute Legendre objects and eval 1d basis
            if self.normalized == False:
                Leg = LegPoly()
            elif self.normalized == True:
                Leg = LegPolyNorm()
            NormSq.append(Leg.normsq(Kmax[i]))  # norms up to K order

        # start computing products
        # Phi = 1.0
        normsq = 1.0
        for di in range(self.dim):
            # Phi = L[di][self.mindex[:,di]] * Phi
            normsq = NormSq[di][mindex[:, di]] * normsq
        self.normsq = (
            normsq  # norm squared * integration factor of 1/2 (since domain is [-1,1])
        )
        self.norm = np.sqrt(normsq)
        return self.normsq

    def setup(self, X):
        # cannot name it self.fit() otherwise PCEReg would overwrite it
        X = np.atleast_2d(X)
        if self.mindex is None:
            self.compile(dim=X.shape[1])  # only compiles once
        else:
            pass

        return self

    def transform(self, X):
        """Transform the given :math:`X` coordinates to the polynomial
        space.

        This method essentially performs a high dimensional kernel mapping onto
        a polynomial space spanned by the Legendre polynomials. This is similar
        to sklearn's PolynomialFeatures, except here the features are
        multi-variate Legendre; the big difference being that the features are
        uncorrelated.

        Alternatively, the resulting

        Parameters
        ----------

        X   :  numpy.ndarray of shape (nsamples, dim)

            feature matrix where each row is a sample of the feature space.

        Returns
        -------

        Phi :  numpy.ndarray of shape (nsamples, nPCTerms)

            returns the polynomial feature map that transforms each sample in :math:`X` of dimension :math:`d` to dimension :math:`nPCTerms`.

        """
        # Scale X if input range to (-1,1) if range is specified
        if self.input_range is not None:
            scaler = DomainScaler(
                dim=X.shape[1], input_range=self.input_range, output_range=(-1, 1)
            )
            X_scaled = scaler.fit_transform(X)
            X = X_scaled.copy()

        # only works for [-1,1] for far
        # compute multindex
        assert (
            np.amin(X) >= -1 and np.amax(X) <= 1
        ), "range for X must be between -1 and 1 for now. scale inputs accordingly. "

        # if sklearn poly selected then use sklearn
        if self.use_sklearn_poly_features is False:

            Max = np.amax(self.mindex, axis=0)
            # construct and evaluate each basis using mindex
            L = []
            NormSq = []
            self.output_type = None
            if X.ndim == 1:
                X = np.atleast_2d(X)
                self.output_type = "scalar"
            for i in range(self.dim):
                # Compute Legendre objects and eval 1d basis
                if self.normalized == False:
                    Leg = LegPoly()
                elif self.normalized == True:
                    Leg = LegPolyNorm()
                Li_max = Leg.Eval1dBasis(x=X[:, i], K=Max[i])
                L.append(Li_max)  # list of size dim of Legendre polys
                NormSq.append(Leg.normsq(Max[i]))  # norms up to K order

            # start computing products
            Phi = 1.0
            normsq = 1.0
            L_array = np.array(L)

            # # method 2 looping over number of samples
            # start = time.time()
            # I = self.mindex.T
            # I0 = np.ones_like(I)*np.arange(self.dim)[:,np.newaxis]
            # Lbig = L_array[I0.ravel(),I.ravel(),:]
            # # Phi = np.prod(Lbig,axis=0)
            # # self.Phi = Phi.T
            # print(time.time() - start)

            # method 2 - looping over dimensions
            # start = time.time()
            for di in range(self.dim):
                # if di%2 == 0: print("main prod loop...", di)
                Phi = L_array[di][self.mindex[:, di]] * Phi
                normsq = NormSq[di][self.mindex[:, di]] * normsq
            if self.store_phi is True:
                self.Phi = Phi.T
            else:
                pass
            self.normsq = normsq  # norm squared
            # print(time.time() - start)

            # if self.normalized:
            #     return self.Phi/np.sqrt(normsq)
            # else:
            #     return self.Phi
            # pdb.set_trace()
            return Phi.T
        elif self.use_sklearn_poly_features is True:
            Vander = self._fit_transform_sklearn_poly_features(X)
            return Vander

    def fit_transform(self, X):
        """Fit and transform the given :math:`X` coordinates to the polynomial
        space.

        This method essentially performs a high dimensional kernel mapping onto
        a polynomial space spanned by the Legendre polynomials. This is similar
        to sklearn's PolynomialFeatures, except here the features are
        multi-variate Legendre; the big difference being that the features are
        uncorrelated.

        Alternatively, the resulting

        Parameters
        ----------

        X   :  numpy.ndarray of shape (nsamples, dim)

            feature matrix where each row is a sample of the feature space.

        Returns
        -------

        Phi :  numpy.ndarray of shape (nsamples, nPCTerms)

            returns the polynomial feature map that transforms each sample in :math:`X` of dimension :math:`d` to dimension :math:`nPCTerms`.

        """
        return self.setup(X).transform(X)

    def polyeval(self, X=None, c=None):
        """Method to evaluate the polynomial.

        Evaluates the polynomial for a given set of coefficients and data
        points.

        Parameters
        ----------

        X   :  numpy.ndarray of shape (nsamples, dim)

            feature matrix where each row is a sample of the feature space.

        c   : numpy.ndarray of shape (nPCTerms,) (optional)

            coefficient array. The evaluation is simply np.dot(Phi,c). The coefficient array can also be internally set so that it does not need to be fed in each time we need to evaluate the polynomial.

        Returns
        -------

        yeval   : numpy.ndarray of shape (X.shape[0],)

            The scalar outputs of the multivariate polynomial evaluated at the feature matrix data points.

        """
        if c is not None:
            coef_ = c  # use c in polyeval
            assert self.compile_flag == True, "Must compile to get multindex."
            assert len(coef_) == self.mindex.shape[0], "c is not the right size."
        elif c is None:
            assert (
                self.coef is not None
            ), "Must define coefficient in constructor or polyeval or run the fit method"
            coef_ = self.coef  # use coefficient from constructor

        if X is not None:
            Phi = self.fit_transform(X)
            yeval = np.dot(Phi, coef_)
        elif X is None:
            Phi = self.Phi  # using stored Vandermonde matrix
            yeval = np.dot(Phi, coef_)

        return yeval

    def eval(self, X=None, c=None):
        """Duplicate of polyeval"""
        return self.polyeval(X=None, c=c)

    def computeSobol(self, c=None):
        """Compute Sobol total order variance based sensitivity indices

        Depending on the Legendre polynomial type (normalized vs non-normalized)
        the formula will be different. The total order sensitivity is given by

        .. math::

            S_i = \sum_{k} \gamma^{-2}_{\\beta^i_k} c^2_{\\beta^i_k}

        where :math:`\{\\beta^i_k\}_{k=\dots}` are the indices that contain at
        least the :math:`i^{th}` dimension, and :math:`\gamma` is the square root norm of that particular basis polynomial (which is 1 for the normalized case).

        For this method, we return the normalized Sobol indices, i.e.

        .. math::

            T_i \doteq S_i/S_{T}

        where :math:`S_{T}` is the total variance, i.e. :math:`\{\\beta^i_k\}_{k=\dots}` are the entire set of basis functions. Thus, the total order sensitivity indices are always less than 1, although their sum can be greater than :math:`S_{T}`.

        This is another advantage of using the Legendre polynomials, in that they are uncorrelated so that their feature importance calculations are very easy to compute.

        Parameters
        ----------

        c   : numpy.ndarray of shape (nPCTerms,)

            coefficient to determine the Sobol indices.

        Returns
        -------

        T   : numpy.ndarray of shape (dim,)

            The total order Sobol sensitivity indices for each dimension

        """
        if self.mindex is None:
            self.compile(dim=self.dim)
        # assert self.compile_flag == True, "Must compile to set mindex. Avoids having to recompute multiindex everytime we run SA."
        if len(self.normsq) == 0:
            normsq = self.computeNormSq()  # if c and M specified in constructor
        else:
            normsq = self.normsq  # from fit transform
        # compute Sobol indices using coefficient vector
        if c is None:
            assert (
                len(self.coef) != 0
            ), "must specify coefficient array or feed it in the constructor."
            coef_ = self.coef
        else:
            coef_ = c

        assert (
            len(coef_) == self.mindex.shape[0]
        ), "Coefficient vector must match the no of rows of the multiindex."

        new_index = (
            np.sum(self.mindex, 1) != 0
        )  # boolean array that doesn't include mean mindex
        if self.normalized:
            totvar_vec = coef_[new_index] ** 2
            self.coefsq = coef_**2
        else:
            totvar_vec = normsq[new_index] * coef_[new_index] ** 2
            self.coefsq = normsq * coef_**2
        totvar = np.sum(totvar_vec)
        # assert totvar > 0, "Coefficients are all zero!"
        S = []
        # in case elements have nan in them
        if np.all(coef_[new_index] == 0):  # ignore the mean
            # print("Returning equal weights!")
            S = np.ones(self.dim) / self.dim  # return equal weights
        else:
            self.sobol_variances = []
            for i in range(self.dim):
                si = self.mindex[:, i] > 0  # boolean
                s = np.sum(totvar_vec[si[new_index]]) / totvar
                self.sobol_variances.append(totvar * s)  # just the variance
                S.append(s)
            # note that s does not have to sum to 1
        return S

    def computeJointSobol(self, c=None):
        if self.mindex is None:
            self.compile(dim=self.dim)
        # assert self.compile_flag == True, "Must compile to set mindex. Avoids having to recompute multiindex everytime we run SA."
        if len(self.normsq) == 0:
            normsq = self.computeNormSq()  # if c and M specified in constructor
        else:
            normsq = self.normsq  # from fit transform
        # compute Sobol indices using coefficient vector
        if c is None:
            assert (
                len(self.coef) != 0
            ), "must specify coefficient array or feed it in the constructor."
            coef_ = self.coef
        else:
            coef_ = c

        assert (
            len(coef_) == self.mindex.shape[0]
        ), "Coefficient vector must match the no of rows of the multiindex."

        # new_index = (np.sum(self.mindex,1)!=0) # boolean array that doesn't include mean mindex
        if self.normalized:
            totvar_vec = coef_**2
            self.coefsq = coef_**2
        else:
            totvar_vec = normsq * coef_**2
            self.coefsq = normsq * coef_**2
        totvar_vec[0] = 0.0
        totvar = np.sum(totvar_vec)

        # fill in joint sensitivities
        S2 = np.zeros((self.dim, self.dim))  # joint sobol matrix
        dim_list = list(range(self.dim))  # list of dimensions
        pairs = list(combinations(dim_list, 2))  # all unique pairs of dimensions
        for pair in pairs:
            # For each pair of dimensions, compute the total variance of the iteraction terms
            i, j = pair[0], pair[1]
            s2i = self.mindex[:, i] > 0  # index where first dim in pair > 0
            s2j = self.mindex[:, j] > 0  # index where first dim in pair > 0
            pindex_temp = [
                a and b for a, b in zip(s2i, s2j)
            ]  # index where both dimensions in pair as nonzero
            js = (
                np.sum(totvar_vec[pindex_temp]) / totvar
            )  # compute tot var if elements in pair
            S2[
                pair[0], pair[1]
            ] = js  # fill in joint sensitivity matrix which will be upper diagonal, with zero for diagonal entries
        return S2.T  # return transpose of upper diagonal to get lower diagonal matrix

    def computeMoments(self, c=None):
        """Methods to compute the mean and variance of the resulting polynomial

        Assuming a uniform distribution over :math:`[-1,1]`, this method
        computes the mean and variance. The mean is the coefficient of the
        :math:`0^{th}` order term, while the variance is the weighted sum of
        squares of the other coefficients. The weighted sum is determined by
        using either the normalized or non-normalized Legendre polynomials.
        Finally, the weight is scaled by .5 correpsonding to the density
        function for the uniform distribution on the previously stated domain.

        .. math::

            \mu = \\frac{1}{2} c_0

        .. math::

            \sigma^2 = \\frac{1}{2} \sum_{i=1}^N \gamma^2_i c^2_i

        Parameters
        ----------

        c   : numpy.ndarray of shape (nPCTerms, )

            coefficient of the polynomial basis. Must have the same number of elements as the number of basis elements.

        Returns
        -------

        mu, var : float,float

            mean and variance floating point values


        """
        if c is None:
            assert (
                self.coef is not None
            ), "coef is not defined. Try running fit or input c"
            coef_ = self.coef
        else:
            coef_ = c
            assert len(c) == len(self.mindex), "coef is wrong size"
        # normsq = self.computeNormSq()
        normsq = self.normsq
        prob_weight = 0.5 ** (self.dim)
        if self.normalized is False:
            self.mu = prob_weight * normsq[0] * coef_[0]
            self.var = prob_weight * np.sum(normsq[1:] * coef_[1:] ** 2)
        elif self.normalized is True:
            self.mu = prob_weight * np.sqrt(normsq[0]) * coef_[0]
            self.var = prob_weight * np.sum(coef_[1:] ** 2)
        return self.mu, self.var


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

# regressor mixin is to add scoring, and baseestimator is to add fit, predict
class PCEReg(PCEBuilder, BaseEstimator, RegressorMixin):
    """Class for performing multivariate polynomial regression

    This class fits the coefficients of a a multivariate polynomial object, aka
    as a polynomial chaos model, using different linear regression algorithms
    from sklearn. Given labeled data pairs :math:`(x_j,y_j)` for
    :math:`j=1,\dots,n`, where :math:`x_j \in \mathbb{R}^d` and :math:`y_j \in \mathbb{R}`, we look for

    .. math::

        \\text{arg} \min_{c} \sum_{j}d\left(f(\mathbf{x}_j;c),y_j\\right) + \\text{Regularizer}(c)

    where :math:`f` is the polynomial model with unknown coefficient parameters, i.e.,

    .. math::

        f(\mathbf{x}_j;c) \doteq \sum_{i=1}^N c_i \Phi_i(\mathbf{x}_j)

    where :math:`d` is the error metric, which is typically a squared error, and the regularizer can be either :math:`\ell_1` (lasso), :math:`\ell_2` (ridge), or both (elastic net). Again, we note that the input must be on :math:`[-1,1]` for now since the polynomials are setup on that domain by default. One may use sklearn's preprocessing utilities to make that transformation or use the DomainScaler class that comes with this library.

    Below we list the parameters to construct the object followed by the class
    attributes and returns.

    Parameters
    ----------
    order : int, default=2

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

    fit_type : {'linear','LassoCV','ElasticNetCV','OmpCV','quadrature'}, default='linear'

        A string defining the algorithm to solve the linear regression problem. All but the quadrature option utilizes sklearn's linear regression algorithms. In order to use the quadrature routine, you must define the 'w' variable in the fit_params dictionary.

    fit_params : default={}

        Dictionary to be passed to the particular fit type algorithm chosen above. See sklearn's documentation for parameters. This dictionary will be passed as a **kwargs type input for the fit algorithm.


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

    nPCTerms : int

        The number of basis terms of the multi-variate polynomial.

    feature_importances_ : ndarray of shape (dim,)

        Sobol sensitivity indices for each dimension. This is computed after the fit function is called.

    coef    : ndarray of shape (nbasis,)

        coefficient array of polynomial function. It can be fed into the constructor, but for most cases it will be computed after self.fit is called.

    Methods
    -------
    feature_importances:
        compute Sobol total order indices, normalized to sum to 1

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
    >>> from tesuract import PCEReg
    >>> p = PCEReg(order=3)
    >>> p.fit(X,y)
    """

    def __init__(
        self,
        order=2,
        customM=None,
        mindex_type="total_order",
        coef=None,
        a=None,
        b=None,
        polytype="Legendre",
        fit_type="linear",
        fit_params={},
        normalized=False,
        store_phi=False,
        input_range=None,
        use_sklearn_poly_features=False,
    ):
        # alphas=np.logspace(-12,1,20),l1_ratio=[.001,.5,.75,.95,.999,1],lasso_tol=1e-2
        self.order = order
        self.mindex_type = mindex_type
        self.customM = customM
        self.a = a
        self.b = b
        self.coef = coef
        # self.c = self.coef # need to remove eventually
        self.polytype = polytype
        self.fit_type = fit_type
        self.fit_params = fit_params
        # self.alphas = alphas # LassoCV default
        # self.l1_ratio = l1_ratio # ElasticNet default
        # self.lasso_tol = lasso_tol
        # self.w = w
        self.store_phi = store_phi
        self.input_range = input_range
        self.normalized = normalized
        self.use_sklearn_poly_features = use_sklearn_poly_features
        super().__init__(
            order=self.order,
            customM=self.customM,
            mindex_type=self.mindex_type,
            coef=self.coef,
            a=self.a,
            b=self.b,
            polytype=self.polytype,
            normalized=self.normalized,
            store_phi=self.store_phi,
            input_range=self.input_range,
            use_sklearn_poly_features=self.use_sklearn_poly_features,
        )

    def _compile(self, X):
        """Builds the multiindex using the PCEBuilder class. Private.

        Parameters
        ----------

        X   : numpy array of size (nsamples, dim)

            data matrix in feature space.

        Returns
        -------

        self    : self

            returns object

        """
        # only
        self._dim = X.shape[1]
        super().compile(dim=self._dim)  # use parent compile to produce the multiindex
        self._M = self.multiindex
        self.Xhat = self.fit_transform(X)
        return self

    def _quad_fit(self, X, y):
        """
        Fit the coefficients of the polynomial model using quadrature points and weights

        It is expected that the data matrix X represent the quadrature point and the weights are given in the fit_params dictionary with the key name 'w'.

        Parameters
        ----------
        X       : numpy.ndarray of size (nsamples, dim)

                Data matrix where each row is the first part of the data pairs (x,y)_i. X values must be between (-1,1), for now.

        y       : numpy.ndarray of size (nsamples,)

                1d array of the data labels. Must be the same size as the number of rows of X.
        """
        # let us assume the weights are for [-1,1]
        self._compile(X)  # compute normalized or non-normalized Xhat
        normsq = self.computeNormSq()
        assert (
            "w" in self.fit_params.keys()
        ), "quadrature weights must be given in dictionary fit_params."
        w = self.fit_params["w"]
        X, w = check_X_y(X, w)  # make sure # quadrature points matces X
        assert (
            np.abs(np.sum(w) - 1) <= 1e-15
        ), "quadrature weights must be scaled to integrate over [-1,1] with unit weight"
        int_fact = 2**self._dim
        if self.normalized == True:
            self.coef = int_fact * np.dot(w * y, self.Xhat) / np.sqrt(normsq)
        else:
            self.coef = int_fact * np.dot(w * y, self.Xhat) / normsq
        return self

    def fit(self, X, y):
        """
        Fit the polynomial using linear regression or quadrature solvers

        The algorithm is determined by the fit_type option in the initialization and the options in fit_params.

        Parameters
        ----------

        X       : numpy.ndarray of shape (nsamples, dim)

                data matrix feature space samples. Must be in [-1,1]

        y       : numpy.ndarray of shape (nsamples,)

                data labels.

        Returns
        -------

        self    : self object

                sets the internal coefficient array self.coef\_

        """
        X, y = check_X_y(X, y)
        # get data attributes
        self._n, self._dim = X.shape
        # Check if dimension has changed and clear multiindices and coefficient if that's the case. If not, leave everythign the same.
        # fit overwrites the coefficient array if there is a mismatch like if the X dimension changes
        if self.coef is not None:
            if len(self.coef) != self.multiindex.shape[0]:
                print("coefficient array mismatch. ")
                self.coef = None  # set to none and fit again

        self._compile(X)  # build multindex and construct basis
        Xhat, y = check_X_y(self.Xhat, y)
        # assert len(self.coef) == self.multiindex.shape[0],"length of coefficient vector is not the same shape as the multindex!"
        # run quadrature fit if weights are specified:
        sample_weights = 1.0 / (np.abs(y) + 1e-8)
        sample_weights /= np.sum(sample_weights)
        if self.fit_type == "quadrature":
            self._quad_fit(X, y)
        if self.fit_type == "linear":
            regmodel = linear_model.LinearRegression(fit_intercept=False)
            regmodel.fit(Xhat, y)
        if self.fit_type == "LassoCV":
            if not self.fit_params:  # if empty dictionary
                self.fit_params = {
                    "alphas": np.logspace(-12, 2, 25),
                    "max_iter": 2500,
                    "tol": 1e-2,
                }
            regmodel = linear_model.LassoCV(fit_intercept=False, **self.fit_params)
            regmodel.fit(Xhat, y)
            self.alpha_ = regmodel.alpha_
        if self.fit_type == "LassoCV_weighted":
            if not self.fit_params:  # if empty dictionary
                self.fit_params = {
                    "alphas": np.logspace(-12, 2, 25),
                    "max_iter": 2500,
                    "tol": 1e-2,
                }
                # self.fit_params={}
            regmodel = linear_model.LassoCV(fit_intercept=False, **self.fit_params)
            regmodel.fit(Xhat, y, sample_weight=sample_weights)
            self.alpha_ = regmodel.alpha_
        if self.fit_type == "ElasticNetCV":
            if not self.fit_params:  # if empty dictionary
                self.fit_params = {
                    "l1_ratio": [0.001, 0.5, 0.75, 0.95, 0.999, 1],
                    "n_alphas": 25,
                    "tol": 1e-2,
                    "n_jobs": 1,
                }
            regmodel = linear_model.ElasticNetCV(fit_intercept=False, **self.fit_params)
            regmodel.fit(Xhat, y)
        if self.fit_type == "ElasticNetCV_weighted":
            if not self.fit_params:  # if empty dictionary
                self.fit_params = {
                    "l1_ratio": [0.001, 0.5, 0.75, 0.95, 0.999, 1],
                    "n_alphas": 25,
                    "tol": 1e-2,
                }
            regmodel = linear_model.ElasticNetCV(fit_intercept=False, **self.fit_params)
            regmodel.fit(Xhat, y, sample_weight=sample_weights)
        if self.fit_type == "RidgeCV":
            if not self.fit_params:  # if empty dictionary
                self.fit_params = {"alphas": np.logspace(-3, 3, 100)}
            regmodel = linear_model.RidgeCV(fit_intercept=False, **self.fit_params)
            regmodel.fit(Xhat, y)
        elif self.fit_type == "OmpCV":
            if not self.fit_params:  # if empty dictionary
                pass
            regmodel = linear_model.OrthogonalMatchingPursuitCV(
                fit_intercept=False, **self.fit_params
            )
            regmodel.fit(Xhat, y)
        if self.fit_type != "quadrature":
            self.coef = regmodel.coef_
        self.coef_ = self.coef  # backwards comaptible with sklearn API
        self.feature_importances_ = self.feature_importances()
        if self.store_phi == False:
            del self.Xhat
        return self

    def sensitivity_indices(self):
        assert self.coef is not None, "Must run fit or feed in coef array."
        return self.computeSobol()

    def feature_importances(self):
        """
        Compute feature importances which are equivalent to the normalized Sobol total order sensitivity indices.

        Parameters
        ----------

        Returns
        -------

        feature_importances_    : numpy.ndarray of shape (dim,)

            array representing normalized Sobol total order indices.

        """
        S = np.array(self.computeSobol())
        S = S / S.sum()
        self.feature_importances_ = S
        return self.feature_importances_

    def joint_effects(self):
        """
        Compute Sobol joint effect sensitivity indices.

        Parameters
        ----------

        Returns
        -------

        joint_effects_    : numpy.ndarray of shape (dim,dim)

            the lower triangular part of the array contains the joint effect sensitivity indices.

        """
        S2 = self.computeJointSobol()
        self.joint_effects_ = S2
        return self.joint_effects_

    def multiindex(self):
        assert self._M is not None, "Must run fit or feed in mindex array."
        return self._M

    def predict(self, X):
        """
        After fitting, evaluates the polynomial for a single feature space sample or array of samples.

        Parameters
        ----------

        X       : numpy.ndarray of shape (nsamples, dim)

                Samples to evaluate the fit polynomial. Must have self.coef\_ set or defined already. Can take 1d or 2d array of samples.

        Returns
        -------

        y       : numpy.ndarray of shape (nsamples,)

                Returns the scalar array of polynomial evaluations.

        """
        # check_is_fitted(self)
        X = np.atleast_2d(X)
        X = check_array(X)
        # self._fit_transform(X)
        # Phi = check_array(self._Phi)
        # ypred = np.dot(Phi,self.coef)
        ypred = self.polyeval(X)
        return ypred


# In development
class PCESeries:
    def __init__(self, dim, C, MIs, time=[], a=None, b=None):
        # construct series of pce's
        assert len(C) == len(MIs), "coef and multindex mismatch!"
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
            pcetemp = PCEBuilder(
                dim=self.dim, customM=self.MIs[i], c=self.C[i], a=self.a, b=self.b
            )
            self.PCEs.append(pcetemp)

    def eval(self, X):
        assert np.atleast_2d(X).shape[1] == self.dim, "dimension mismatch!"
        y = []
        for i in range(self.N):
            ytemp = self.PCEs[i].eval(X)
            y.append(ytemp)
        y = np.array(y).T
        return y

    def ieval(self, T, X):
        # each element of T can be a vector
        # first evaluate all PCEs at X points
        Y = self.eval(X)
        # interpolate for each time point
        T1d = np.atleast_1d(T)
        assert len(Y) == len(T1d), "time and X mismatch!"
        Yi = []
        for i in range(len(T1d)):
            yitemp = np.interp(T1d[i], self.time, Y[i])
            Yi.append(yitemp)
        return np.array(Yi)
