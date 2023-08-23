import pandas as pd
import numpy as np
import array
from matplotlib.pyplot import scatter
from numpy.linalg import norm
from scipy.integrate import odeint
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.legend import Legend

import profile_likelihood

#True values
Dr = 400
Dg = 400
kr = 0.035
kg = 0.038
K = 0.004
sigma = 2e-5
ptrue = np.array([Dr,Dg, kr, kg, K, sigma])


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

YR48 = pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = 'T', skiprows = [1, 2])
YR48 = np.array(YR48).flatten()
YR48 = YR48/(1745*(x0_right-x0_left))

YG0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu' , usecols = 'I', skiprows = [1, 2])
YG0 = np.array(YG0).flatten()
YG0 = YG0/((x0_right-x0_left)*1745)

YG48 = pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = 'U', skiprows = [1, 2])
YG48 = np.array(YG48).flatten()
YG48 = YG48/(1745*(x0_right-x0_left))

S0 = pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = 'J', skiprows = [1, 2])
S0 = np.array(S0).flatten()
S0 = S0/(1745*(x0_right-x0_left))

S48 = pd.read_excel('data.xlsx', sheet_name = '1205Lu', usecols = 'V', skiprows = [1, 2])
S48 = np.array(S48).flatten()
S48 = S48/(1745*(x0_right-x0_left))

cr0 = interpolate.interp1d(x0_left, YR0, bounds_error=False, fill_value="extrapolate")
xnew = np.linspace(0, 1254.5, 500)
yrnew0 = cr0(xnew)

cg0 = interpolate.interp1d(x0_left, YG0, bounds_error= False, fill_value = "extrapolate")
ygnew0 = cg0(xnew)

s0 = interpolate.interp1d(x0_left, S0, bounds_error=False, fill_value="extrapolate")
snew0 = s0(xnew)

y0_0 = np.hstack((cr0(x),cg0(x)))

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

    y_sol0 = odeint(ode,y0_0,t)
    Y_0 = np.vstack(y_sol0)
    U_0 = Y_0[0:7,0:n]
    u_left = interpolate.interp1d(x, U_0, bounds_error = False, fill_value = "extrapolate")(x0_left)
    V_0 = Y_0[0:7, n:]
    v_left = interpolate.interp1d(x, V_0, bounds_error = False, fill_value = "extrapolate")(x0_left)
    y = U_0+V_0
    y_left = interpolate.interp1d(x, y, bounds_error= False, fill_value = "extrapolate")(x0_left)

    ydata = y_left + sigma*(np.random.randn(np.shape(y_left)[0], np.shape(y_left)[1]))
    udata =  u_left + sigma*(np.random.randn(np.shape(u_left)[0], np.shape(u_left)[1]))
    vdata =  v_left + sigma*(np.random.randn(np.shape(v_left)[0], np.shape(v_left)[1]))

    plt.figure(figsize = (15,9))
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex = True, sharey = True)
    plt.subplot(1, 3, 1)
    plt.scatter(x0_left,ydata[0,], color = 'b')
    plt.plot(x0_left,y_left[0,],color = 'b')
    plt.scatter(x0_left,udata[0,], color = 'm')
    plt.plot(x0_left,u_left[0,],color = 'm')
    plt.scatter(x0_left,vdata[0,], color = 'g')
    plt.plot(x0_left,v_left[0, ],color = 'g')
    plt.legend(['Synthetic data for the total population', 'Model for the total population', 'Synthetic data for the red population', 'Model for the red population', 'Synthetic data for the green population', 'Model for the green population'], loc = 'upper center', borderpad = 3)
    plt.title("t=0h")
    plt.ylabel("Cell densities")

    plt.subplot(1,3,2)
    plt.scatter(x0_left,ydata[3,],color = 'b')
    plt.plot(x0_left,y_left[3,],color = 'b')
    plt.scatter(x0_left,udata[3,],color = 'm')
    plt.plot(x0_left,u_left[3,],color = 'm')
    plt.scatter(x0_left,vdata[3,],color = 'g')
    plt.plot(x0_left,v_left[3, ],color = 'g')
    plt.title("t=48h")
    plt.xlabel("Position")

    plt.subplot(1,3,3)
    plt.scatter(x0_left,ydata[6,],color = 'b')
    plt.plot(x0_left,y_left[6,], color = 'b')
    plt.scatter(x0_left,udata[6,],color = 'm')
    plt.plot(x0_left,u_left[6,], color = 'm')
    plt.scatter(x0_left,vdata[6, ],color = 'g')
    plt.plot(x0_left,v_left[6,], color = 'g')
    plt.title("t=96h")
    plt.show()
    return(ydata)