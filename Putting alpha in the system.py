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
alpha = 0.57
ptrue = np.array([Dr,Dg, kr, kg, K, alpha, sigma])


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
xnew = np.linspace(0, 1254.5, 500)

N0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = 'J', skiprows = [1, 2])
N0 = np.array(N0).flatten()
N0 = N0/(1745*(x0_right-x0_left))

def alpha_IC(alpha):
    YR0 = alpha * N0
    YG0 = (1 - alpha) * N0

    cr0 = interpolate.interp1d(x0_left, YR0, bounds_error=False, fill_value="extrapolate")
    yrnew0 = cr0(xnew)

    cg0 = interpolate.interp1d(x0_left, YG0, bounds_error= False, fill_value = "extrapolate")
    ygnew0 = cg0(xnew)

    n0 = interpolate.interp1d(x0_left, N0, bounds_error=False, fill_value="extrapolate")
    nnew0 = n0(xnew)

    y0_0 = np.hstack((cr0(x),cg0(x)))

    def solve_model(p):
        # Get the parameters
        [Dr, Dg, kr, kg, K ,alpha,sigma]  = p
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

        y_sol0 = odeint(ode,y0_0,t)
        Y_0 = np.vstack(y_sol0)
        U_0 = Y_0[0:7,0:n]
        V_0 = Y_0[0:7, n:]

        y = U_0+V_0
        y_match_data = interpolate.interp1d(x, y, bounds_error= False, fill_value = "extrapolate")(xdata)
        return(y_match_data)
    # Generate some data
    ptrue[5] = alpha
    v = solve_model(ptrue)
    return(v)

v = alpha_IC(0.57)
ydata = v + sigma*(np.random.randn(np.shape(v)[0], np.shape(v)[1]))

def negloglike(p):
    #Get the parameters
       [Dr, Dg, kr, kg, K ,alpha, sigma]  = p
    #Solve the model
       ymodel= alpha_IC(p[5])
    # Get the residuals
       resids = ydata - ymodel
    # Get the normal log pdf for each resid, sum them to get log likelihood
       nloglike = -np.sum(normal_logpdf(resids,sigma))
       return nloglike

#Obtain the maximum likelihood estimate
res = minimize_posparam(negloglike,ptrue)
pmle = res.x

val = np.linspace(0.5,0.6,10)
prof = profile_parameter(negloglike,val,5,len(ptrue),pmle)



#v = np.zeros((4,24))
#v[0,] = N0
#v[1,] = (np.array(pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'N', skiprows = [1, 2])).flatten())/(1745*(x0_right-x0_left))
#v[2,] = (np.array(pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'R', skiprows = [1, 2])).flatten())/(1745*(x0_right-x0_left))
#v[3,] = (np.array(pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'V', skiprows = [1, 2])).flatten())/(1745*(x0_right-x0_left))



plt.plot(val,prof)

h = negloglike(pmle) + 1.92
plt.axhline(y=h, color='k',linestyle='--')
plt.xlabel("alpha")
profx = scipy.interpolate.interp1d(val, prof)
#profreduced = np.array(prof) - h
#profxreduced =  scipy.interpolate.interp1d(val, profreduced)
#from scipy.optimize import fsolve
#idx = fsolve(profxreduced, [0.000015, 0.00003])
#plt.plot(idx, profx(idx),'ro')

plt.show()

#res = scipy.optimize.minimize(ydata - alpha_IC, 0.57)
#res.x
#alphamle = res.x[5]
#plt.figure(figsize=(15,9))
#for i in range(4):
    #plt.subplot(2,3,i+1)
    #plt.plot(xdata, alpha_IC(alphamle)[i,], color = 'c')
    #plt.scatter(xdata, v[i,], color = 'b')

#a = [0.5, 0.57, 0.58]
#for alpha in a:
    #w = alpha_IC(alpha) 
    #for i in range(4):
        #plt.subplot(2,3,i+1)
        #plt.plot(xdata,w[i, ])
        #plt.scatter(xdata,v[i,], color = 'b')

#plt.legend( [" Recovered model", "Data"]) 
#plt.show()
