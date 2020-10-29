import jax
import jax.numpy as jnp
import numpy as onp
import scipy

# This needs to run at startup
# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#Double-(64bit)-precision
jax.config.update('jax_enable_x64', True)

def run(np,optname):
    print(f"\nRunning {optname} on {np}")
    def rosen(x):
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    # double precision is needed for lbfgsb and probably other optimisers
    # https://github.com/google/jax/issues/936
    # https://github.com/scipy/scipy/issues/5832
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2], dtype='float64')

    optopt = {'disp': True}

    if np==onp:
      return scipy.optimize.minimize(rosen, x0, method=optname, options=optopt)
    else:
      g_rosen = jax.grad(rosen)

      g_rosen_used = g_rosen
      if optname=='L-BFGS-B':
        # Need to make sure the data is copy from jax for LBFGSB
        # https://github.com/google/jax/issues/1510
        # asarray is not sufficient
        # g_rosen_as_np = lambda x:onp.asarray(jax.jit(g_rosen)(x))
        g_rosen_np = lambda x:onp.array(jax.jit(g_rosen)(x))
        g_rosen_used = g_rosen_np

      return scipy.optimize.minimize(rosen, x0, jac=g_rosen_used, method=optname, options=optopt)
