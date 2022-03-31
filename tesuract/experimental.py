import numpy as np
import scipy
import time as T
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    DotProduct,
    WhiteKernel,
)
from sklearn.gaussian_process import GaussianProcessRegressor
import tesuract
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp
from scipy.optimize import fmin_l_bfgs_b
import pickle, pdb
from tqdm import tqdm
from scipy.linalg import sqrtm


from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV


def npsolve(K, y):
    """Matrix solver using numpy's linalgebra solve

    This uses numpy's solve mechanism, which automatically checks for positive definiteness

    Parameters
    ----------
    K: np.ndarray
            square matrix to be inverted. Must be positive definite.

    y: np.ndarray
            Must have the same dimensions as the K matrix

    Returns
    -------
    x: np.ndarray
            Returns solution to Kx = y

    """
    # returns inv(K)*y, i.e. solves Kx = y
    return np.linalg.solve(K, y)


def csolve(K, y):
    """Solves the matrix inverse problem using Cholesky decomposition

    Solves Kx = y using the Cholesky decomposition method such that x = inv(K)*y

    Parameters
    ----------
    K: np.ndarray
            square matrix to be inverted. Must be positive definite.

    y: np.ndarray
            Must have the same dimensions as the K matrix

    Returns
    -------
    x: np.ndarray
            Returns solution to Kx = y

    """
    # returns inv(K)*y, i.e. solves Kx = y
    L = scipy.linalg.cholesky(K, lower=True)
    return scipy.linalg.cho_solve((L, True), y), L


def nplml(theta, X, y, kernel, nugget=1e-10):
    """Compute the log marginal likelihood using numpy default solve (slower than cholesky)"""
    kernel_clone = kernel.clone_with_theta(theta)
    K = kernel_clone(X)
    K[np.diag_indices_from(K)] += nugget
    alpha = npsolve(K, y)
    # Compute log-likelihood (compare line 7)
    log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y, alpha)
    log_likelihood_dims -= 0.5 * np.log(np.linalg.det(K))
    log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
    log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
    return log_likelihood


def clml(theta, X, y, kernel, nugget=1e-10, eval_gradient=False, return_K=False):
    """Compute the log marginal likelihood using Cholesky decomposition

    Modified and adapted from sklearn's GP regression class
    """
    kernel_clone = kernel.clone_with_theta(theta)
    if eval_gradient:
        K, K_gradient = kernel_clone(X, eval_gradient=True)
    else:
        K = kernel_clone(X)
    K[np.diag_indices_from(K)] += nugget
    L = scipy.linalg.cholesky(K, lower=True)
    alpha = scipy.linalg.cho_solve((L, True), y)

    # Compute log-likelihood (compare line 7)
    log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y, alpha)
    log_likelihood_dims -= np.log(np.diag(L)).sum()  # cholesky includes factor of 1/2
    log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
    log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

    if eval_gradient:  # compare Equation 5.9 from GPML
        tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
        tmp -= scipy.linalg.cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
        # Compute "0.5 * trace(tmp.dot(K_gradient))" without
        # constructing the full matrix tmp.dot(K_gradient) since only
        # its diagonal is required
        log_likelihood_gradient_dims = 0.5 * np.einsum("ijl,jik->kl", tmp, K_gradient)
        log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)
        return log_likelihood, log_likelihood_gradient
    elif return_K:
        return log_likelihood, K
    else:
        return log_likelihood


# minimization solver
def minimize_call(xstart, objfunc, bounds, factr, disp=5):
    """
    Call to minimize in order to use mp within class

    This is a helper function to get multiprocessing to work internally within class. This gets around the pickling error based on the global interpreter lock limitation in Python.
    """
    res = fmin_l_bfgs_b(
        objfunc,
        xstart,
        approx_grad=True,
        bounds=bounds,
        factr=factr,
        pgtol=1e-10,
        disp=disp,
    )
    return res


# utility function for choosing the right starting points and checking bounds
def random_sample_within_bounds(n, bounds, rn=np.random.RandomState(327)):
    """Generate uniform random samples within bounds

    For use in an ensemble Bayesian MCMC method.
    """
    dim = len(bounds)
    samples = []
    low = [b[0] for b in bounds]
    high = [b[1] for b in bounds]
    for d in range(dim):
        usample = rn.rand(n)
        sample_d = usample * (high[d] - low[d]) + low[d]
        samples.append(sample_d)
    samples = np.array(samples).T
    return samples


def check_array_within_bounds(x, bounds):
    """Check that array is within the list of bounds"""
    low, high = np.array(bounds).T
    bounds_check = ((x >= low) & (x <= high)).all()
    return bounds_check


# import emcee
class GaussianProcessPCERegression:
    """Gaussian process regression with polynomial mean function

    This class wraps sklearn's GP regressor class to include a polynomial mean function using the PCEReg from tesuract.

    Parameters
    ----------
    kernel: sklearn.gaussian_process.kernels
        sklearn's GP kernels, e.g., matern, rbf, etc.

    order: int, default=2
        order or degree of the polynomials. By default we start with a quadratic.

    poly_alg: {'monomials','Legendre'}, default='monomials'
        Type of polynomials to use. For higher dimensions, use 'Legendre' since it scales better than the tensor product monomials.

    input_range: list(bounds), default=None
        List of bounds for the input range. If not provided and using Legendre, the input must be scaled from -1 to 1.

    alpha: float
        Unused for now, but I think this was for the pure GP method.

    n_restarts_optimizer: int, default = 0
        How many times to solve the optimization of the log marginal likelihood. If using ElasticNetCV or LassoCV, might want to use the default otherwise it could be very slow. Also, since we are using multiprocessing, it is important to

    random_state: int, default=None
        Random state for choosing the initial guess in the random restarts for optimization

    fit_params: dict, default=None
        Dictionary to give the fit type. E.g., if using sklearn's RidgeCV, you can specify any of the algorithm options. See sklearn's documentation for all the lists of possible fit parameters

    fit_type: {"RidgeCV", "LassoCV", "ElasticNetCV"}, default="RidgeCV"
        fit algorithm used for solving the least squares solution to the polynomial coefficients. Unless the problem is zero noise, it is always better to use a regularized least squares solution. RidgeCV is the fastest of the three, but the others allow l-1 or sparsity constraints which may be useful.

    Examples
    --------
    Given X which is n x d and Y which is n x R and an initial guess of the GP kernel, an a polynomial of order 7 using ElasticNetCV regularization, we can construct the GP+PCE regressor as follows.

    >>> gppce = GPReg(kernel=kernel_guess,order=7,poly_alg='Legendre',input_range=None,n_restarts_optimizer=0,random_state=np.random.RandomState(23423),fit_type='ElasticNetCV',fit_params={'n_jobs':-1})

    We can then fit it by simply calling
    >>> gppce.fit(X,Y)

    and return the predicted polynomial mean and GP standard deviation and covariance

    >>> gppce_mu_pred, gppce_std_pred = gppce.predict(X,return_std=True)
    >>> _, gppce_cov_pred = gppce.predict(X,return_cov=True)

    Notes
    -----
    Due to problems with compatibility with OPENBLAS and the multiprocessing package, it might be necessary to set the environmental variable "export OPENBLAS_NUM_THREADS=1" before running the GPReg fit function.

    """

    def __init__(
        self,
        kernel,
        order=2,
        poly_alg="monomials",
        input_range=None,
        alpha=None,
        n_restarts_optimizer=0,
        random_state=None,
        fit_params=None,
        fit_type="RidgeCV",
    ):
        self.kernel = kernel
        self.order = order
        self.fit_type = fit_type
        self.fit_params = fit_params
        self.poly_alg = poly_alg
        self.input_range = input_range
        self.compile_flag = False
        self.alpha = alpha  # reg param
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    def _compile(self, X, y):
        # X must be 2d and y can be 1d
        self.y = y.copy()
        self.X = X.copy()
        if self.y.ndim == 1:
            self.y = self.y[:, np.newaxis]
        assert self.X.shape[0] == self.y.shape[0], "shape mistmatch."
        if self.order == 0:
            self.Xhat = self.X
        elif self.order > 0:
            if self.poly_alg == "Legendre":
                self.pce = tesuract.PCEBuilder(
                    order=self.order, input_range=self.input_range
                )  # store_phi=True to keep in the pce object
            if self.poly_alg == "monomials":
                self.pce = PolynomialFeatures(degree=self.order)
            self.Xhat = self.pce.fit_transform(self.X)
            self.coef_size = self.Xhat.shape[1]
        self.theta_size = len(self.kernel.theta)
        self.feature_labels = [
            "theta_" + str(t + 1) for t in range(self.theta_size)
        ] + ["coef_" + str(c) for c in range(self.coef_size)]
        self.theta_bounds = [tuple(bi) for bi in self.kernel.bounds]
        self.compile_flag = True
        # self.coef_bounds = [(-np.inf,np.inf) for c in range(self.coef_size)]
        self.coef_bounds = [(-1e1, 1e1) for c in range(self.coef_size)]
        self.all_bounds = self.theta_bounds + self.coef_bounds
        return self

    def log_marginal_likelihood(self, theta, nugget=1e-8, return_coef=False):
        """Compute the log marginal likelihood of the data using the kernel hyper parameters and polynomial

        We use the formulation as outlined in https://www.mathworks.com/help/stats/exact-gpr-method.html. Note there is some uncertainty and confusion about the sigma variable in matlab's documentation and the white noise kernel parameter in sklearn's formulation. In this code and in sklearn's, if we add a white noise kernel to the full kernel, the log marginal likelihood is formulated the same as MATLAB's. The difference is in the prediction. Here, since we are separating out the stochastic/ GP from the polynomial model, it may not matter, but if we were to use MATLAB's posterior prediction, we have to modify the predict function.

        The log marginal likelihood can be written purely as a function of the kernel hyper parameters since the polynomial coefficient sub-problem can be written as a least squares problem with an exact OLS solution. This means that we do not have to search over the space of polynomial coefficients, avoiding, in part, the curse of dimensionality.

        Parameters
        ----------
        theta : np.ndarray
            kernel hyper-paramaters as computed using sklearn's kernel structure. Note that this structure is very confusing, so we try to leave it untouched to avoid bugs.

        nugget : float, default=1e-8
            Component to add to diagonal to avoid convergence issues with condition numbers. Make this number larger if having problems with Cholesky solves.

        return_coef : bool, default=False
            returns the polynomial coefficients. For use in debugging, but internally it is needed for the alternating minimization approach outlined in the MATLAB link above where we then solve the polynomial least squares problem.

        Returns
        -------
        float
            returns the scalar number which represents the log marginal likelihood of the data, given the polynomial and kernel theta hyper-parameters

        """
        # compute pce coefficient first and then use the coef to evaulate lml
        kernel_clone = self.kernel.clone_with_theta(theta)
        K = kernel_clone(self.X)
        K[np.diag_indices_from(K)] += nugget
        L = scipy.linalg.cholesky(K, lower=True)
        H = self.Xhat
        # KiH = np.linalg.solve(K,H) #scipy.linalg.cho_solve((L, True), H)
        # HTKiH_palpha = np.dot(H.T,KiH) + self.alpha*np.eye(self.coef_size)
        # Kiy = np.linalg.solve(K,self.y) #scipy.linalg.cho_solve((L,True),self.y)
        # HTKiy = np.dot(H.T,Kiy)
        # beta = np.linalg.solve(HTKiH_palpha,HTKiy)
        # beta_bar = beta.mean(axis=1)
        # coef = beta_bar
        # print("coef1",coef[:5])

        ############ new method
        Xtilde = np.linalg.solve(L, H)
        if self.y.ndim == 1:
            ytilde = np.linalg.solve(L, self.y).flatten()
        else:
            ytilde = np.linalg.solve(L, self.y.mean(axis=1)).flatten()

        # Default linear solver is ridge regression cv in sklearn with 25 alphas
        ridgecv_fit_params = {"alphas": np.logspace(-3, 3, 50)}
        self.fit_params = {}  # default empty fit params
        if self.fit_type is None:
            self.fit_type = "RidgeCV"
            self.fit_params = ridgecv_fit_params

        if self.fit_type == "RidgeCV":
            regr = RidgeCV(fit_intercept=False, **self.fit_params)
        elif self.fit_type == "ElasticNetCV":
            regr = ElasticNetCV(fit_intercept=False, **self.fit_params)
        elif self.fit_type == "LassoCV":
            regr = LassoCV(fit_intercept=False, **self.fit_params)

        regr.fit(Xtilde, ytilde)
        coef = regr.coef_
        #######################

        self.y_minus_mu = self.y - np.dot(self.Xhat, coef)[:, np.newaxis]
        assert (
            self.compile_flag == True
        ), "Must compile first to precompute polynomial transform and save X,y data."
        # assert len(theta) == self.theta_size, "theta wrong length."
        # Compute log-likelihood (compare line 7)
        Kiymu = scipy.linalg.cho_solve((L, True), self.y_minus_mu)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", self.y_minus_mu, Kiymu)
        log_likelihood_dims -= np.log(np.diag(L)).sum()  # cholesky w factor of 1/2
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
        # loglike = clml(theta,self.X,self.y_minus_mu,self.kernel,nugget=nugget,eval_gradient=False)
        if return_coef:
            return log_likelihood, coef
        else:
            return log_likelihood

    def _objfunc(self, theta, minimize=True):
        """Wrapper for the log marginal likelihood to turn the problem into a minimization instead of a maximization.

        Parameters
        ----------
        theta : np.ndarray
            kernel hyper-parameters. Note that the polynomial coefficients are completely determined by the kernel hyper-parameters using an OLS analytic solution (see the MATLAB link above for the formulation).

        minimize : bool, default=True
            Turns the log marginal likelihood objective into a minimization problem by adding a factor of -1 if True, and 1 if False (for maximization). The latter may be used for a Bayesian formulation where the log likelihood is needed in its maximization form.

        Returns
        -------
        float
            Returns the log marginal likelihood
        """
        mfact = 1.0
        if minimize:
            mfact = -1.0
        lml = self.log_marginal_likelihood(theta)
        lml = mfact * lml
        return lml

    def fit(self, X, y, **fit_params):
        """Main fit routine for determining the GP kernel hyper-parameters and polynomial coefficients for the stochastic/ repeated data X, Y

        Fitting a Polynomial + GP to the data pair X,y where y can have repeated elements. In general, it is difficult to deal with repeated data in a standard sense since this means your data matrix would have to have repeated rows, thus reducing the conditioning of your system and making the GP approach much more difficult (it scales like O(n^3) where n is the total number of samples). This routine essentially fits one GP model using a log marginal likelihood that averages the error out over the repeated samples. If you were to just fit a deterministic regression model to the averaged data, you would obtain a good mean approximation but loose information about the stochastic component. This helps avoid that.

        Parameters
        ----------
        X : np.ndarray
            Feature space data matrix, n x feature_dimensions, where n is the number of spatial samples
        y : np.ndarray
            Can be a matrix of size n x R, where R is the number of repeated outputs. Since this is meant for stochastic code, each realization of the simulation can result in slightly different target values so y can be a matrix, where the columns are the repeated samples.

        Returns
        -------
        self
            returns the GPReg object itself

        """
        self._compile(X, y)
        opt_args = {"factr": 1e7, "disp": False}
        myfun = partial(
            minimize_call, objfunc=self._objfunc, bounds=self.theta_bounds, **opt_args
        )
        nsolves = self.n_restarts_optimizer + 1
        # xstarts = [np.zeros(len(self.all_bounds)) for i in range(nsolves)]
        if self.random_state is None:
            self.random_state = np.random.RandomState(997)
        ##########
        start_bounds = self.theta_bounds
        xstarts = random_sample_within_bounds(
            nsolves, start_bounds, rn=self.random_state
        )
        # add solver for pce coef only to improve problem of poor fit when data not scaled.
        ##########
        # xstarts = [2*self.random_state.rand(len(self.all_bounds))-1 for i in range(nsolves)]
        # print(xstarts)
        print("Starting parallel optimization ...")
        start = T.time()
        with mp.get_context("fork").Pool() as pool:
            res = pool.map(myfun, [x for x in xstarts])
        end = T.time()
        # print("Total time for optimization is %.5f seconds." %(end-start))

        xvalues = [r[0] for r in res]
        fvalues = [r[1] for r in res]
        min_index = np.argmin(fvalues)
        self.max_logp_ = -1 * fvalues[min_index]
        self.xmap_ = xvalues[min_index]
        self.theta_ = self.xmap_  # separate theta
        logl, self.coef_ = self.log_marginal_likelihood(
            self.theta_, return_coef=True
        )  # self.xmap_[self.theta_size:]   # separate pce coef
        return self

    def predict(
        self,
        X,
        return_pce_only=False,
        return_std=False,
        return_cov=False,
        return_samples=None,
    ):
        # plot the test prediction using the GP PCE model
        y_mu_gppce_pred = self.pce.polyeval(X, c=self.coef_)
        kernel_opt_ = self.kernel.clone_with_theta(self.theta_)
        rn = self.random_state
        gp = GaussianProcessRegressor(kernel=kernel_opt_, random_state=rn)
        gp_mu_resid, gp_std_resid = gp.predict(X, return_std=True)
        _, gp_cov_resid = gp.predict(X, return_cov=True)
        if return_std is True:
            return y_mu_gppce_pred + gp_mu_resid, gp_std_resid
        elif return_cov is True:
            return y_mu_gppce_pred + gp_mu_resid, gp_cov_resid
        elif return_pce_only is True:
            return y_mu_gppce_pred, self.pce
        if isinstance(return_samples, int):
            nsamples = return_samples
            gp_resid_samples = gp.sample_y(X, nsamples)
            return y_mu_gppce_pred[:, np.newaxis] + gp_resid_samples
        else:
            return y_mu_gppce_pred + gp_mu_resid


# metrics for comparing outputs
def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
            ONLY FOR DIAGONAL COV
    - accepts stacks of means, but only one S0 and S1

    From wikipedia
    KL( (m0, S0) || (m1, S1))
             = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                              (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term = np.trace(iS1 @ S0)
    # det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0))
    # det_term = np.sum(np.log(S1)) - np.sum(np.log(S0))
    det_term = np.sum(np.log(np.diag(S1) / np.diag(S0)))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff  # np.sum( (diff*diff) * iS1, axis=1)
    # print(tr_term,det_term,quad_term)
    return 0.5 * (tr_term + det_term + quad_term - N)


def Wasserstein2(mu1, cov1, mu2, cov2):
    """Wasserstein or earth mover distance

    For large matrices that have very small eigenvalues, this may not work well.
    """
    mu_sse = np.sum((mu1 - mu2) ** 2)
    trC1 = np.trace(cov1)
    trC2 = np.trace(cov2)
    sqrtS1 = sqrtm(cov1)
    B2 = trC1 + trC2 - 2 * np.trace(sqrtm(sqrtS1 @ cov2 @ sqrtS1))
    return mu_sse + B2
