# Doing inference on our model

import pandas as pd
import numpy as np
from matplotlib.pyplot import scatter
from numpy.linalg import norm
from scipy.integrate import odeint
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#Solve the model

open('data.xlsx', 'r')
x0_left = pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'E', skiprows = [1, 2])
x0_left = np.array(x0_left).flatten()

x0_right = pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = 'F', skiprows = [1, 2])
x0_right = np.array(x0_right).flatten()

xdata = (x0_left + x0_right) / 2

# True parameters
Dr = 400
Dg = 400
kr = 0.035
kg = 0.038
K = 0.004
ptrue_short = [400,400, 0.035, 0.038, 0.004]
sigma = 0.01


###########################################################################


n = 101
x = np.linspace(0,x0_right[-1],n)
dx = x[1]-x[0]

t = np.linspace(0,48,4)

YR0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'H', skiprows = [1, 2])
YR0 = np.array(YR0).flatten()
YR0 = YR0/((x0_right-x0_left)*K*1745)

YG0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'I', skiprows = [1, 2])
YG0 = np.array(YG0).flatten()
YG0 = YG0/((x0_right-x0_left)*1745*K)

S0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = 'J', skiprows = [1, 2])
S0 = np.array(S0).flatten()
S0 = S0/(1745*(x0_right-x0_left)*K)

cr0 = interpolate.interp1d(x0_left, YR0)
xnew = np.linspace(0, 1254.5, 500)
yrnew0 = cr0(xnew)

cg0 = interpolate.interp1d(x0_left, YG0)
ygnew0 = cg0(xnew)

s0 = interpolate.interp1d(x0_left, S0)
snew0 = s0(xnew)

y0_0 = np.hstack((cr0(x),cg0(x)))

## Perform inference
ptrue = np.array([Dr, Dg, kr, kg, K, sigma])


# Generate model predictions
def solve_model(p):

    # Get the parameters
    [Dr, Dg, kr, kg, K ,sigma]  = p
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
        return -kr*vr+2* kg * vg * (1 - (vg+vr))
    def f2(vr,vg):
        return -kg*vg*(1-(vg+vr))+kr*vr

    def ode(y,t):
        u = y[0:n]
        v = y[n:]
        (du,dv) = rhs(u,v,f1,f2,t)
        dy = np.hstack((du,dv))
        return dy
    u0 = y0_0[0:n]
    v0 = y0_0[n:]

    y_sol0 = odeint(ode, y0_0, t)
    Y_0 = np.vstack(y_sol0)
    U_0 = Y_0[0:4,0:n]
    V_0 = Y_0[0:4, n:]

    y = np.concatenate((U_0, V_0))
    y_match_data = interpolate.interp1d(x, y)(xdata)

    return(y_match_data)

# Generate some data
v = solve_model(ptrue)

y = v + sigma*(np.random.randn(np.shape(v)[0], np.shape(v)[1]))


###
# y = solve_model(ptrue)
# yend = y[7,]

# plt.plot(x,yend)
# plt.show()

# xdata = (x0_left + x0_right) / 2
# plt.scatter(xdata,YR0)
# plt.show()

# yend_match = interpolate.interp1d(x, y)(xdata)
# plt.plot(x,y[6,])
# plt.scatter(xdata,yend_match[6,])
# plt.show()

###


# How close is our model to our data?

def model_data_norm(p):
    return (norm(solve_model(p) - (y)))
# Generate 

# Now, let's optimise
from scipy.optimize import minimize
res = minimize(model_data_norm, [500, 500, 0.1, 0.1, 1, 0.1], )
poptim = res.x

# Let's compare the model prediction at the best fit to the data
plt.scatter(xdata,y[7,],label="data")
plt.scatter(xdata,solve_model(poptim)[7,],label="model")
plt.show()
