# Doing inference on our model

import pandas as pd
import numpy as np
import array
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

x0 = np.zeros(1)
x0= np.append(x0, x0_right)
xdata = (x0_left + x0_right) / 2

for i in range(4):
     y_data[i, ] = np.array(pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = [4*i+7], skiprows = [1,2])).flatten()
for i in range(4,8):
     y_data[i, ] = np.array(pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = [4*(i-2)], skiprows = [1,2])).flatten()


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
x = np.linspace(0,x0_left[-1],n)
# here it used to be x0_left, but I think it should be x0_right, or matbe even x0
dx = x[1]-x[0]

t = np.linspace(0,96,7)

YR0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'H', skiprows = [1, 2])
YR0 = np.array(YR0).flatten()
YR0 = YR0/((x0_right-x0_left)*1745)

YG0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'I', skiprows = [1, 2])
YG0 = np.array(YG0).flatten()
YG0 = YG0/((x0_right-x0_left)*1745)

S0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = 'J', skiprows = [1, 2])
S0 = np.array(S0).flatten()
S0 = S0/(1745*(x0_right-x0_left))

cr0 = interpolate.interp1d(x0_left, YR0, bounds_error=False, fill_value=(x0_left[0], x0_left[-1]))
# here maybe x0/x0_right? but because of the size of YR0 we need to make a choice here about which ends to ignore
xnew = np.linspace(0, 1254.5, 500)
yrnew0 = cr0(xnew)

cg0 = interpolate.interp1d(x0_left, YG0, bounds_error= False, fill_value = (x0_left[0], x0_left[-1]))
ygnew0 = cg0(xnew)

s0 = interpolate.interp1d(x0_left, S0, bounds_error=False, fill_value=(x0_left[0], x0_left[-1]))
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
        return -kr*vr+2* kg * vg * (1 - (vg+vr)/K)
    def f2(vr,vg):
        return -kg*vg*(1-(vg+vr)/K)+kr*vr

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
    U_0 = Y_0[0:7,0:n]
    V_0 = Y_0[0:7, n:]

    y = np.concatenate((U_0, V_0))
    y_match_data = interpolate.interp1d(x, y, bounds_error= False, fill_value = (x[0], x[-1]))(xdata)

    return(y_match_data)

# xdata is between x0_left and x0_right so x also needs to have similar bounds which is why we get an error here

# For the rectangle rule
# for i in range(8):
#       y_rectangle_rule [i, ] = (x0_right-x0_left)*y_match_data[i, ]

#For the trapezium rule
#y_trapezium_rule = np.zeros ((8, 2*n))
#for i in range(8):
#    for j in range(2*n):
#        y_trapezium_rule[i,j] = (y[i,j]+y[i,j+1])/2*(x0_right[j]-x0_left[j])

# Generate some data
v = solve_model(ptrue)

y = v + sigma*(np.random.randn(np.shape(v)[0], np.shape(v)[1]))


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
