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


# mtx_solve_mse = np.mean((npsolve(K,y) - csolve(K,y)[0])**2)


def nplml(theta, X, y, kernel, nugget=1e-10):
    """Compute the log marginal likelihood using numpy's mechanisms"""
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
    """Compute the log marginal likelihood using Cholesky decomposition"""
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
class GPReg:
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
        Random state 

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

    def log_marginal_likelihood(
        self,
        theta,
        coef=None,
        nugget=1e-6,
        eval_gradient=False,
        coef_bounds=(-2.0, 2.0),
    ):
        if coef is None:
            self.coef = np.zeros(self.coef_size)
            self.y_minus_mu = self.y
        else:
            assert (
                len(coef) == self.coef_size
            ), "coefficient array is not the corrent size."
            self.y_minus_mu = self.y - np.dot(self.Xhat, coef)[:, np.newaxis]
            # return list of bounds
            self.coef_bounds = [coef_bounds for c in coef]
        # evaluate log marginal likelihood
        assert (
            self.compile_flag == True
        ), "Must compile first to precompute polynomial transform and save X,y data."
        assert len(theta) == self.theta_size, "theta wrong length."
        loglike = clml(
            theta,
            self.X,
            self.y_minus_mu,
            self.kernel,
            nugget=nugget,
            eval_gradient=eval_gradient,
        )
        return loglike

    # test optimize
    def _objfunc(self, x0, fix_coef=False, coef=None, minimize=True):
        l2reg = self.alpha
        theta0 = x0[: self.theta_size]
        if fix_coef:
            coef0 = coef.copy()
        else:
            coef0 = x0[self.theta_size :]
        mfact = 1.0
        if minimize:
            mfact = -1.0
        lml = self.log_marginal_likelihood(theta0, coef0)
        if l2reg is not None:
            lml += -l2reg * np.sum(coef0 ** 2)
            # lml += - l2reg*np.sqrt(np.sum(coef0**2+1e-6)) # l1
        lml = mfact * lml
        return lml

    def log_marginal_likelihood2(self, theta, nugget=1e-8, return_coef=False):
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

        # reg = eval(name + "()") # way to convert string to function call
        if self.fit_type == "RidgeCV":
            regr = RidgeCV(fit_intercept=False, **self.fit_params)
        elif self.fit_type == "ElasticNetCV":
            regr = ElasticNetCV(fit_intercept=False, **self.fit_params)
        elif self.fit_type == "LassoCV":
            regr = LassoCV(fit_intercept=False, **self.fit_params)
        # regr = ElasticNetCV(cv=2,random_state=0,tol=1e-4,max_iter=100000)
        # regr = LassoCV(cv=2,random_state=0,tol=1e-3,max_iter=10000)
        regr.fit(Xtilde, ytilde)
        coef = regr.coef_
        # print("coef2", coef2[:5])
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

    def _objfunc2(self, theta, minimize=True):
        # theta
        # if fix_coef:
        #   coef0 = coef.copy()
        # else:
        #   coef0 = x0[self.theta_size:]
        mfact = 1.0
        if minimize:
            mfact = -1.0
        lml = self.log_marginal_likelihood2(theta)
        # if l2reg is not None:
        #   lml += -l2reg*np.sum(coef0**2)
        # lml += - l2reg*np.sqrt(np.sum(coef0**2+1e-6)) # l1
        lml = mfact * lml
        return lml

    def fit2(self, X, y):
        self._compile(X, y)
        opt_args = {"factr": 1e7, "disp": False}
        myfun = partial(
            minimize_call, objfunc=self._objfunc2, bounds=self.theta_bounds, **opt_args
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
        with Pool() as pool:
            res = pool.map(myfun, [x for x in xstarts])
        end = T.time()
        # print("Total time for optimization is %.5f seconds." %(end-start))

        xvalues = [r[0] for r in res]
        fvalues = [r[1] for r in res]
        min_index = np.argmin(fvalues)
        self.max_logp_ = -1 * fvalues[min_index]
        self.xmap_ = xvalues[min_index]
        self.theta_ = self.xmap_  # separate theta
        logl, self.coef_ = self.log_marginal_likelihood2(
            self.theta_, return_coef=True
        )  # self.xmap_[self.theta_size:]   # separate pce coef
        return self

    def fit(self, X, y):
        self._compile(X, y)
        opt_args = {"factr": 1e2, "disp": 25}
        myfun = partial(
            minimize_call, objfunc=self._objfunc, bounds=self.all_bounds, **opt_args
        )
        nsolves = self.n_restarts_optimizer + 1
        # xstarts = [np.zeros(len(self.all_bounds)) for i in range(nsolves)]
        if self.random_state is None:
            self.random_state = np.random.RandomState(997)
        ##########
        start_bounds = self.theta_bounds + [
            (-100, 100.0) for c in range(self.coef_size)
        ]
        xstarts = random_sample_within_bounds(nsolves, start_bounds)
        # add solver for pce coef only to improve problem of poor fit when data not scaled.
        ##########
        # xstarts = [2*self.random_state.rand(len(self.all_bounds))-1 for i in range(nsolves)]
        # print(xstarts)
        print("Starting parallel optimization ...")
        start = T.time()
        with Pool() as pool:
            res = pool.map(myfun, [x for x in xstarts])
        end = T.time()
        print("Total time for optimization is %.5f seconds." % (end - start))

        xvalues = [r[0] for r in res]
        fvalues = [r[1] for r in res]
        min_index = np.argmin(fvalues)
        self.max_logp_ = -1 * fvalues[min_index]
        self.xmap_ = xvalues[min_index]
        self.theta_ = self.xmap_[: self.theta_size]  # separate theta
        self.coef_ = self.xmap_[self.theta_size :]  # separate pce coef
        return self

    def fit_bayesian(self, X, y):
        self._compile(X, y)

        def log_like(x):
            return self._objfunc(x, minimize=False)

        def log_prob(x):
            bc = check_array_within_bounds(x, self.all_bounds)
            if np.all(bc):
                return log_like(x)
            else:
                return -1e12

        nwalkers = 100
        nburn = 1500
        nruns = 500
        ndim = len(self.all_bounds)
        start_bounds = self.theta_bounds + [(-1, 1) for c in range(self.coef_size)]
        p0 = random_sample_within_bounds(nwalkers, start_bounds)
        # print([log_prob(xi) for xi in p0])

        # start emcee sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        state = sampler.run_mcmc(p0, nburn, progress=True)  # burn in
        sampler.reset()
        sampler.run_mcmc(state, nruns, progress=True)  # main run
        try:
            acor_times = sampler.get_autocorr_time()
            print(acor_times)
            thin = round(acor_times.mean() / 2)
        except:
            print("Cannot get acor times. Chain too short. ")
            thin = 1

        import matplotlib.pyplot as plt

        samples = sampler.get_chain(flat=True, thin=thin)
        logprobs = sampler.get_log_prob(flat=True)
        argmap = np.argmax(logprobs)
        map_soln = samples[argmap]
        self.xmap_bayes_ = map_soln
        self.theta_ = self.xmap_bayes_[: self.theta_size]  # separate theta
        self.coef_ = self.xmap_bayes_[self.theta_size :]  # separate pce coef

        self.samples_ = samples
        self.logprobs_ = logprobs
        self.thetas_ = samples[:, : self.theta_size]
        self.coefs_ = samples[:, self.theta_size :]

        map_logprob = logprobs[argmap]
        for nd in range(ndim):
            fig, ax = mpl.subplots()
            ax.hist(samples[:, nd], 100, color="k", histtype="step")
            fig.savefig("mcmc_hist_plot_dim" + str(nd + 1).zfill(2) + ".png")

        fig = corner.corner(samples, truths=map_soln, labels=self.feature_labels)
        fig.savefig("corner_plot.png")

        return self

    def predict_bayes(
        self,
        X,
        return_pce_only=False,
        return_std=False,
        return_cov=False,
        return_samples=None,
    ):
        # plot the test prediction using the GP PCE model
        predictions = []
        pce_predictions = []
        nsamples = len(self.samples_)
        plot_thin = round(nsamples / 200)
        for i in tqdm(range(0, nsamples, plot_thin)):
            y_mu_gppce_pred = self.pce.polyeval(X, c=self.coefs_[i])
            kernel_opt_ = self.kernel.clone_with_theta(self.thetas_[i])
            rn = self.random_state
            gp = GaussianProcessRegressor(kernel=kernel_opt_, random_state=rn)
            gp_mu_resid, gp_std_resid = gp.predict(X, return_std=True)
            _, gp_cov_resid = gp.predict(X, return_cov=True)
            pred = y_mu_gppce_pred + gp_mu_resid
            predictions.append(pred)
            pce_predictions.append(y_mu_gppce_pred)
        # if return_std is True:
        #   return y_mu_gppce_pred + gp_mu_resid, gp_std_resid
        # elif return_cov is True:
        #   return y_mu_gppce_pred + gp_mu_resid, gp_cov_resid
        # elif return_pce_only is True:
        #   return y_mu_gppce_pred, self.pce
        # if isinstance(return_samples,int):
        #   nsamples = return_samples
        #   gp_resid_samples = gp.sample_y(X,nsamples)
        #   return y_mu_gppce_pred[:,np.newaxis] + gp_resid_samples
        # else:
        #   return y_mu_gppce_pred + gp_mu_resid
        return np.array(predictions), np.array(pce_predictions)

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
