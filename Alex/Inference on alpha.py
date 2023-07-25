import pandas as pd
import numpy as np
import array
from matplotlib.pyplot import scatter
from numpy.linalg import norm
from scipy.integrate import odeint
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import profile_likelihood

#True values
Dr = 400
Dg = 400
kr = 0.035
kg = 0.038
K = 0.004
sigma = 2e-5
alpha = 0.3
ptrue = np.array([Dr,Dg, kr, kg, K, sigma,alpha])


#Times
t = np.linspace(0,96,7)

#Setup for the model

open('data.xlsx', 'r')
x0_left = pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'E', skiprows = [1, 2])
x0_left = np.array(x0_left).flatten()

x0_right = pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = 'F', skiprows = [1, 2])
x0_right = np.array(x0_right).flatten()


xdata = (x0_left + x0_right) / 2


n = 51
x = np.linspace(0,x0_left[-1],n)
dx = x[1]-x[0]

YR0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'H', skiprows = [1, 2])
YR0 = np.array(YR0).flatten()
YR0 = YR0/((x0_right-x0_left)*1745)

YG0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'I', skiprows = [1, 2])
YG0 = np.array(YG0).flatten()
YG0 = YG0/((x0_right-x0_left)*1745)

S0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = 'J', skiprows = [1, 2])
S0 = np.array(S0).flatten()
S0 = S0/(1745*(x0_right-x0_left))

cr0 = interpolate.interp1d(x0_left, YR0, bounds_error=False, fill_value="extrapolate")
xnew = np.linspace(0, 1254.5, 500)
yrnew0 = cr0(xnew)

cg0 = interpolate.interp1d(x0_left, YG0, bounds_error= False, fill_value = "extrapolate")
ygnew0 = cg0(xnew)

s0 = interpolate.interp1d(x0_left, S0, bounds_error=False, fill_value="extrapolate")
snew0 = s0(xnew)

y0_0 = np.hstack((cr0(x),cg0(x)))

y0_total = y0_0[0:n] + y0_0[n:]

def solve_model(p):
    # Get the parameters
    [Dr, Dg, kr, kg, K ,sigma,alpha]  = p
    # Solve the model
    def rhs(u,v,f, g, t):
        du = f(u,v)
        dv = g(u,v)
        for i in range(1,n-2):
                du[i] += Dr/ dx** 2 * (u[i-1]-2*u[i]+u[i+1])
                dv[i] += Dg/ dx** 2 * (v[i+1]-2*v[i]+ v[i-1])
        du[0] += 2 * Dr/ dx ** 2 * (u[1]-u[0])
        dv[0] += 2 * Dg/ dx ** 2 * (v[1]-v[0])
        du[n-1] += 2* Dr/ dx ** 2 * (u[n-2]-u[n-1])
        dv[n-1] += 2* Dg/ dx ** 2 * (v[n-2]-v[n-1])
        return (du,dv)
    def f1(vr,vg):
        return -kr*vr+2* kg * vg * (1 - (vg+vr)/K)
    def f2(vr,vg):
        return -kg*vg*(1-(vg+vr)/K)+kr*vr

    def ode(y,t):
        u = y[0:n]
        v = y[n:]
        (du,dv) = rhs(u,v,f1,f2,t)
        dy = np.hstack((du,dv))
        return dy
    u0 = alpha * y0_total
    v0 = (1 - alpha) * y0_total

    y_sol0 = odeint(ode,np.concatenate((u0,v0)),t)
    Y_0 = np.vstack(y_sol0)
    U_0 = Y_0[0:7,0:n]
    V_0 = Y_0[0:7, n:]

    y = U_0+V_0
    y_match_data = interpolate.interp1d(x, y, bounds_error= False, fill_value = "extrapolate")(xdata)
    return(y_match_data)

# def alpha_function(p):
#     # Get the parameters
#     [Dr, Dg, kr, kg, K ,sigma]  = p
#     # Solve the model
#     def rhs(u,v,f, g, t):
#         du = f(u,v)
#         dv = g(u,v)
#         for i in range(1,n-2):
#                 du[i] += Dr/ dx** 2 * (u[i-1]-2*u[i]+u[i+1])
#                 dv[i] += Dg/ dx** 2 * (v[i+1]-2*v[i]+ v[i-1])
#         du[0] += 2 * Dr/ dx ** 2 * (u[1]-u[0])
#         dv[0] += 2 * Dg/ dx ** 2 * (v[1]-v[0])
#         du[n-1] += 2* Dr/ dx ** 2 * (u[n-2]-u[n-1])
#         dv[n-1] += 2* Dg/ dx ** 2 * (v[n-2]-v[n-1])
#         return (du,dv)
#     def f1(vr,vg):
#         return -kr*vr+2* kg * vg * (1 - (vg+vr)/K)
#     def f2(vr,vg):
#         return -kg*vg*(1-(vg+vr)/K)+kr*vr

#     def ode(y,t):
#         u = y[0:n]
#         v = y[n:]
#         (du,dv) = rhs(u,v,f1,f2,t)
#         dy = np.hstack((du,dv))
#         return dy
#     u0 = y0_0[0:n]
#     v0 = y0_0[n:]

#     y_sol0 = odeint(ode,y0_0,t)
#     Y_0 = np.vstack(y_sol0)
#     U_0 = Y_0[0:7,0:n]
#     V_0 = Y_0[0:7, n:]
#     y = U_0+V_0
#     alpha = np.zeros((7,n))
#     for i in range(7):
#          for j in range(n):
#              alpha[i,j] = U_0[i,j] / y[i,j]
#     alpha_interp = interpolate.interp1d(x, alpha, bounds_error =  False, fill_value = "extrapolate")(x0_left)
#     #plt.plot(x0_left, np.transpose(alpha_interp[1, ]))
#     #plt.show()
#     return(alpha_interp)

# Generate some data
a = alpha_function(ptrue)
v = solve_model(ptrue)

ydata = v + sigma*(np.random.randn(np.shape(v)[0], np.shape(v)[1]))

def negloglike(p):
    # Get the parameters
    [Dr, Dg, kr, kg, K ,sigma,alpha]  = p
    # Solve the model
    ymodel= solve_model(p)
    # Get the residuals
    resids = ydata - ymodel
    # Get the normal log pdf for each resid, sum them to get log likelihood
    nloglike = -np.sum(normal_logpdf(resids,sigma))
    print(nloglike)
    return nloglike

## Obtain the maximum likelihood estimate
res = minimize_posparam(negloglike,ptrue)
pmle = res.x
print("MLE is",pmle)

adata = alpha_function(pmle)

plt.plot(x0_left, adata[1,], color = 'r')
plt.plot(x0_left, a[1,], color = 'b')
plt.show()
