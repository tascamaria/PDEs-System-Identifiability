## PROFILE LIKELIHOOD AND RELATED FUNCTIONS
import numpy as np
import scipy

# Setup normal log density (for log-likelihood), zero mean, sigma std
def normal_logpdf(x,sigma): 
    return -np.log(sigma*np.sqrt(2*np.pi)) - x**2 / 2 / sigma**2

# Maximise function (ensuring parameters are all positive)
def minimize_posparam(func,x0):
    func2 = lambda lx : func(np.exp(lx))
    res = scipy.optimize.minimize(func2,np.log(x0),method="COBYLA")
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

def profile_parameter_ci(negloglike,i,n,pmle,xinterval):
    """Use profile likelihood to construct a 95% CI for parameter `i`.

    Inputs:
        - negloglike: negative log likelihood function
        - i: index of parameter to profile
        - n: total number of parameters
        - pmle: maximum likelihood estimate
        - xinterval: [lb,ub], where lb < pmle[i] < ub, an interval containing both crossings of the profile likelihood
    """ 
    # Indices of remaining variables
    zidx = np.setdiff1d(range(0,n),i)
    qidx = np.append(zidx,i)
    # Lower and Upper intervals
    int_lower = np.array([xinterval[0],pmle[i]])
    int_upper = np.array([pmle[i],xinterval[1]])
    # Initial condition for profile optimisation
    z0 = pmle[zidx]
    # Profile value at the maximum
    negloglike_mle = negloglike(pmle)
    # Target (95% confidence interval)
    negloglike_target = negloglike_mle + 1.92
    # Negative log likelihood with parameter i fixed at x
    def negloglike_partial(x,z):
        q = np.append(z,x)
        p = q[np.argsort(qidx)]
        return negloglike(p)
    # Function to root find
    def func_to_zero(x):
        func = lambda z : negloglike_partial(x,z)
        return minimize_posparam(func,z0).fun - negloglike_target
    # Root find
    ci = np.zeros(2)
    ci[0] = scipy.optimize.brentq(func_to_zero,int_lower[0],int_lower[1],xtol=1e-6,rtol=1e-6)
    ci[1] = scipy.optimize.brentq(func_to_zero,int_upper[0],int_upper[1],xtol=1e-6,rtol=1e-6)
    return ci