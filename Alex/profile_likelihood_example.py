import numpy as np
import matplotlib.pyplot as plt
import scipy

# Import functionality from profile_likelihood.py
import profile_likelihood

## Logistic model

#  du/dt = r * u * (1 - u / K) subject to u(0) = u0

# True values
r = 1.0
K = 2.0
u0 = 0.1
sigma = 0.1

# Vector of true values
ptrue = np.array([r,K,u0,sigma])

# Times to look at
T = np.linspace(0,10,11)

## Setup logistic model
def solve_model(p):
    # Get the parameters
    [r,K,u0,sigma] = p
    # Setup the ODE
    def rhs(u,t):
        return r * u * (1 - u / K)
    # Solve the model
    sol = scipy.integrate.odeint(rhs,u0,T)
    # Format the solution to match the data
    y = np.hstack(sol)
    return y

## Generate some synthetic data
v = solve_model(ptrue)
ydata = v + sigma * np.random.randn(len(T))

## Setup (negative) log-likelihood function
def negloglike(p):
    # Get the parameters
    [r,K,u0,sigma] = p
    # Solve the model
    ymodel = solve_model(p)
    # Get the residuals
    resids = ydata - ymodel
    # Get the normal log pdf for each resid, sum them to get log likelihood
    return -np.sum(normal_logpdf(resids,sigma))

## Obtain the maximum likelihood estimate
res = minimize_posparam(negloglike,ptrue)
pmle = res.x
print("MLE is",pmle)

## Profile the parameter K
Kvals = np.linspace(1.5,2.5,20)
Kprof = profile_parameter(negloglike,Kvals,1,4,pmle)

## Plot the results

# Data against the model fit
fig = plt.subplot(1,2,1)
plt.scatter(T,ydata,color='k',label="Data")
plt.plot(T,solve_model(pmle),color='b',label="Model")

# Profile loglikelihood for K
plt.subplot(1,2,2)
plt.plot(Kvals,Kprof)
plt.axhline(y=negloglike(pmle)+1.92,color='k',linestyle='--')
plt.xlabel("K")
plt.show()