## PROFILE LIKELIHOOD AND RELATED FUNCTIONS
import numpy as np
import scipy

# Setup normal log density (for log-likelihood), zero mean, sigma std
def normal_logpdf(x,sigma): 
    return -np.log(sigma*np.sqrt(2*np.pi)) - x**2 / 2 / sigma**2

# Maximise function (ensuring parameters are all positive)
def minimize_posparam(func,x0):
    func2 = lambda lx : func(np.exp(lx))
    res = scipy.optimize.minimize(func2,np.log(x0))
    res.x = np.exp(res.x)
    return res

# Code to produce a profile likelihood
def profile_parameter(negloglike,xv,i,n,z0):
    """Profile parameter `i` at points given by `xv` (assumes all parameters are positive)

    Inputs:
        - negloglike: negative log likelihood function
        - xv: array of values
        - i: index of parameter to profile
        - n: total number of parameters
        - z0: initial guess*

    * If len(z0) = n, guess is interpretated as such for the full parameter space, else for the nuisance parameters
    """
    # Indices of remaining variables
    zidx = np.setdiff1d(range(0,n),i)
    qidx = np.append(zidx,i)
    # Fix initial condition (if given as full length)
    if len(z0) == n:
        z0 = z0[zidx]
    # Negative log likelihood with parameter i fixed at x
    def negloglike_partial(x,z):
        q = np.append(z,x)
        p = q[np.argsort(qidx)]
        return negloglike(p)
    # The optimum (profile) at this value of x
    def optim_negloglike_partial(x):
        func = lambda z : negloglike_partial(x,z)
        return minimize_posparam(func,z0).fun
    # Calculate and return complete profile
    return np.array([optim_negloglike_partial(x) for x in xv])