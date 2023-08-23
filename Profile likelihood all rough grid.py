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
#Dg = 400
kr = 0.035
kg = 0.038
K = 0.004
sigma = 2e-5
alpha = 0.3
ptrue1 = np.array([Dr, kr, kg, K, sigma])
ptrue2 = np.array([Dr, kr, kg, K, sigma,alpha])


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

def solve_model1(p):
    # Get the parameters
    [Dr, kr, kg, K ,sigma]  = p
    # Solve the model
    def rhs(u,v,f, g, t):
        du = f(u,v)
        dv = g(u,v)
        for i in range(1,n-2):
                du[i] += Dr/ dx** 2 * (u[i-1]-2*u[i]+u[i+1])
                dv[i] += Dr/ dx** 2 * (v[i+1]-2*v[i]+ v[i-1])
        du[0] += 2 * Dr/ dx ** 2 * (u[1]-u[0])
        dv[0] += 2 * Dr/ dx ** 2 * (v[1]-v[0])
        du[n-1] += 2* Dr/ dx ** 2 * (u[n-2]-u[n-1])
        dv[n-1] += 2* Dr/ dx ** 2 * (v[n-2]-v[n-1])
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

    y_sol0 = odeint(ode,np.concatenate((u0,v0)),t)
    Y_0 = np.vstack(y_sol0)
    U_0 = Y_0[0:7,0:n]
    V_0 = Y_0[0:7, n:]

    y = U_0+V_0
    y_match_data = interpolate.interp1d(x, y, bounds_error= False, fill_value = "extrapolate")(xdata)
    return(y_match_data)

def solve_model2(p):
    # Get the parameters
    [Dr, kr, kg, K ,sigma,alpha]  = p
    # Solve the model
    def rhs(u,v,f, g, t):
        du = f(u,v)
        dv = g(u,v)
        for i in range(1,n-2):
                du[i] += Dr/ dx** 2 * (u[i-1]-2*u[i]+u[i+1])
                dv[i] += Dr/ dx** 2 * (v[i+1]-2*v[i]+ v[i-1])
        du[0] += 2 * Dr/ dx ** 2 * (u[1]-u[0])
        dv[0] += 2 * Dr/ dx ** 2 * (v[1]-v[0])
        du[n-1] += 2* Dr/ dx ** 2 * (u[n-2]-u[n-1])
        dv[n-1] += 2* Dr/ dx ** 2 * (v[n-2]-v[n-1])
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

# Generate some data
v1 = solve_model1(ptrue1)
v2 = solve_model2(ptrue2)
ydata1 = v1 + sigma*(np.random.randn(np.shape(v1)[0], np.shape(v1)[1]))
ydata2 = v2 + sigma*(np.random.randn(np.shape(v2)[0], np.shape(v2)[1]))

def negloglike1(p):
    # Get the parameters
    [Dr, kr, kg, K ,sigma]  = p
    # Solve the model
    ymodel= solve_model1(p)
    # Get the residuals
    resids = ydata1 - ymodel
    # Get the normal log pdf for each resid, sum them to get log likelihood
    nloglike = -np.sum(normal_logpdf(resids,sigma))
    print(nloglike)
    return nloglike

def negloglike2(p):
    # Get the parameters
    [Dr, kr, kg, K ,sigma,alpha]  = p
    # Solve the model
    ymodel= solve_model2(p)
    # Get the residuals
    resids = ydata2 - ymodel
    # Get the normal log pdf for each resid, sum them to get log likelihood
    nloglike = -np.sum(normal_logpdf(resids,sigma))
    print(nloglike)
    return nloglike

## Obtain the maximum likelihood estimate
res1 = minimize_posparam(negloglike1,ptrue1)
pmle1 = res1.x
print("MLE is",pmle1)

res2 = minimize_posparam(negloglike2,ptrue2)
pmle2 = res2.x
print("MLE is",pmle2)

h1 = negloglike1(pmle1) + 1.92
h2 = negloglike2(pmle2) +1.92

fig, axis = plt.subplots(2,3, sharey = True)
plt.figure(figsize =(15,9))

val = np.linspace(300,600,10)
prof01 = profile_parameter(negloglike1, val, 0, len(ptrue1),pmle1)
prof02 = profile_parameter(negloglike2,val,0,len(ptrue2),pmle2)
#axis[0,0].axhline(y=h1, color='b',linestyle='--')
#axis[0,0].axhline(y = h2, color = 'r', linestyle = '--')
axis[0,0].plot(val,prof01-h1, color = 'b')
axis[0,0].plot(val,prof02-h2, color = 'r')
axis[0,0].plot(pmle1[0], profile_parameter(negloglike1, [pmle1[0]], 0, len(ptrue1), pmle1)-h1, 'bo')
axis[0,0].plot(pmle2[0], profile_parameter(negloglike2, [pmle2[0]], 0, len(ptrue2), pmle2)-h2, 'ro')
axis[0,0].set(xlabel ='D')
#profx = scipy.interpolate.interp1d(val, prof)
#profreduced = np.array(prof) - h
#profxreduced =  scipy.interpolate.interp1d(val, profreduced)
#from scipy.optimize import fsolve
#idx = fsolve(profxreduced, [350, 500])
#plt.plot(idx, profx(idx),'ro')

#plt.subplots(2,4,2)
#val = np.linspace(300,500,10)
#prof = profile_parameter(negloglike,val,1,len(ptrue),pmle)
#plt.plot(val,prof)
#h = negloglike(pmle) + 1.92
#plt.axhline(y=h, color='k',linestyle='--')
#plt.xlabel("Dg")
#profx = scipy.interpolate.interp1d(val, prof)
#profreduced = np.array(prof) - h
#profxreduced =  scipy.interpolate.interp1d(val, profreduced)
#from scipy.optimize import fsolve
#idx = fsolve(profxreduced, [350, 500])
#plt.plot(idx, profx(idx),'ro')

val1 = np.linspace(0.02,0.05,10)
prof11 = profile_parameter(negloglike1,val1,1,len(ptrue1),pmle1)
prof12 = profile_parameter(negloglike2,val1,1,len(ptrue2),pmle2)
#axis[0,1].axhline(y=h1, color='b',linestyle='--')
#axis[0,1].axhline(y=h2, color = 'r', linestyle = '--')
axis[0,1].plot(val1,prof11-h1, color = 'b')
axis[0,1].plot(val1, prof12-h2, color = 'r')
axis[0,1].plot(pmle1[1], profile_parameter(negloglike1, [pmle1[1]], 1, len(ptrue1), pmle1)-h1, 'bo')
axis[0,1].plot(pmle2[1], profile_parameter(negloglike2, [pmle2[1]], 1, len(ptrue2), pmle2)-h2, 'ro')
axis[0,1].set(xlabel='kr')
#profx = scipy.interpolate.interp1d(val, prof)
#profreduced = np.array(prof) - h
#profxreduced =  scipy.interpolate.interp1d(val, profreduced)
#from scipy.optimize import fsolve
#idx = fsolve(profxreduced, [0.03, 0.04])
#plt.plot(idx, profx(idx),'ro')


val2 = np.linspace(0.02,0.06,10)
prof21 =  profile_parameter(negloglike1, val2, 2, len(ptrue1), pmle1)
prof22 = profile_parameter(negloglike2,val2,2,len(ptrue2),pmle2)
#axis[0,2].axhline(y=h1, color='b',linestyle = '--')
#axis[0,2].axhline(y=h2, color = 'r', linestyle ='--')
axis[0,2].plot(val2,prof21-h1, color = 'b')
axis[0,2].plot(val2, prof22-h2, color = 'r')
axis[0,2].plot(pmle1[2], profile_parameter(negloglike1, [pmle1[2]], 2, len(ptrue1), pmle1)-h1, 'bo')
axis[0,2].plot(pmle2[2], profile_parameter(negloglike2, [pmle2[2]], 2, len(ptrue2), pmle2)-h2, 'ro')
axis[0,2].set(xlabel='kg')
#profx = scipy.interpolate.interp1d(val, prof)
#profreduced = np.array(prof) - h
#profxreduced =  scipy.interpolate.interp1d(val, profreduced)
#from scipy.optimize import fsolve
#idx = fsolve(profxreduced, [0.035, 0.045])
#plt.plot(idx, profx(idx),'ro')

val3 = np.linspace(0.001,0.006,10)
prof31 = profile_parameter(negloglike1,val3,3,len(ptrue1),pmle1)
prof32 = profile_parameter(negloglike2, val3,3,len(ptrue2),pmle2)
#axis[1,0].axhline(y = h1, color = 'b', linestyle = '--')
#axis[1,0].axhline(y=h2, color='r',linestyle='--')
axis[1,0].plot(val3,prof31-h1, color = 'b')
axis[1,0].plot(val3, prof32-h2, color = 'r')
axis[1,0].plot(pmle1[3], profile_parameter(negloglike1, [pmle1[3]], 3, len(ptrue1), pmle1)-h1, 'bo')
axis[1,0].plot(pmle2[3], profile_parameter(negloglike2, [pmle2[3]], 3, len(ptrue2), pmle2)-h2, 'ro')
axis[1,0].set(xlabel='K')
#profx = scipy.interpolate.interp1d(val, prof)
#profreduced = np.array(prof) - h
#profxreduced =  scipy.interpolate.interp1d(val, profreduced)
#from scipy.optimize import fsolve
#idx = fsolve(profxreduced, [0.0035, 0.0045])
#plt.plot(idx, profx(idx),'ro')


val4 = np.linspace(0.00001,0.00005,10)
prof41 = profile_parameter(negloglike1,val4,4,len(ptrue1),pmle1)

prof42 = profile_parameter(negloglike2, val4,4,len(ptrue2), pmle2)
#axis[1,1].axhline(y=h1, color='b',linestyle='--')
#axis[1,1].axhline(y = h2, color = 'r', linestyle = '--')
axis[1,1].plot(val4,prof41-h1, color = 'b')
axis[1,1].plot(val4, prof42-h2, color ='r')
axis[1,1].plot(pmle1[4], profile_parameter(negloglike1, [pmle1[4]], 4, len(ptrue1), pmle1)-h1, 'bo')
axis[1,1].plot(pmle2[4], profile_parameter(negloglike2, [pmle2[4]], 4, len(ptrue2), pmle2)-h2, 'ro')
axis[1,1].set(xlabel='sigma')
#profx = scipy.interpolate.interp1d(val, prof)
#profreduced = np.array(prof) - h
#profxreduced =  scipy.interpolate.interp1d(val, profreduced)
#from scipy.optimize import fsolve
#idx = fsolve(profxreduced, [0.00002, 0.000025])
#plt.plot(idx, profx(idx),'ro')

val5 = np.linspace(0,1,10)
prof5 = profile_parameter(negloglike2,val5,5,len(ptrue2),pmle2)
#axis[1,2].axhline(y=h2, color='k',linestyle='--')
axis[1,2].plot(val5,prof5-h2)
axis[1,2].set(xlabel='alpha')
axis[1,2].plot(pmle2[5], profile_parameter(negloglike2, [pmle2[5]], 5, len(ptrue2), pmle2)-h2, 'ko')


#profx = scipy.interpolate.interp1d(val, prof)
#profreduced = np.array(prof) - h
#profxreduced =  scipy.interpolate.interp1d(val, profreduced)
#from scipy.optimize import fsolve
#idx = fsolve(profxreduced, [0.55, 0.045])
#plt.plot(idx, profx(idx),'ro')

plt.show() 

import csv
with open("data.csv", 'w', newline = '') as file :
     writer = csv.writer(file)
     writer.writerow(['prof11', prof01])
     writer.writerow(['prof11', prof02])
     writer.writerow(['prof11', prof11])
     writer.writerow(['prof11', prof12])
     writer.writerow(['prof11', prof21])
     writer.writerow(['prof11', prof22])
     writer.writerow(['prof11', prof31])
     writer.writerow(['prof11', prof32])
     writer.writerow(['prof11', prof41])
     writer.writerow(['prof11', prof42])
     writer.writerow(['prof11', prof5])

