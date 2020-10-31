# -*- coding: utf-8 -*-
"""
Demonstration of the EnKF method using the Lorenz96 system 
"PyDA: A hands-on introduction to dynamical data assimilation with Python"
@authors: Shady E. Ahmed, Suraj Pawar, Omer San
"""


import numpy as np
import matplotlib.pyplot as plt
from examples import *
from time_integrators import *


def EnKF(ubi,w,ObsOp,JObsOp,R,B):
    
    # The analysis step for the (stochastic) ensemble Kalman filter 
    # with virtual observations

    n,N = ubi.shape # n is the state dimension and N is the size of ensemble
    m = w.shape[0] # m is the size of measurement vector

    # compute the mean of forecast ensemble
    ub = np.mean(ubi,1)    
    # compute Jacobian of observation operator at ub
    Dh = JObsOp(ub)
    # compute Kalman gain
    D = Dh@B@Dh.T + R
    K = B @ Dh.T @ np.linalg.inv(D)
        
    
    wi = np.zeros([m,N])
    uai = np.zeros([n,N])
    for i in range(N):
        # create virtual observations
        wi[:,i] = w + np.random.multivariate_normal(np.zeros(m), R)
        # compute analysis ensemble
        uai[:,i] = ubi[:,i] + K @ (wi[:,i]-ObsOp(ubi[:,i]))
        
    # compute the mean of analysis ensemble
    ua = np.mean(uai,1)    
    # compute analysis error covariance matrix
    P = (1/(N-1)) * (uai - ua.reshape(-1,1)) @ (uai - ua.reshape(-1,1)).T
    return uai, P

#%% Application: Lorenz 96

n = 36 #dimension of state
F = 8 #forcing term

dt = 0.01
tm = 20
nt = int(tm/dt)
t = np.linspace(0,tm,nt+1)

############################ Twin experiment ##################################
np.random.seed(seed=1)
# Observation operator
def h(u):
    n= u.shape[0]
    m= 9
    H = np.zeros((m,n))
    di = int(n/m) #distance between measurements
    for i in range(m):
        H[i,(i+1)*di-1] = 1
    z = H @ u
    return z

# Jacobian of observational map
def Dh(u):
    n= u.shape[0]
    m= 9
    H = np.zeros((m,n))    
    di = int(n/m) # distance between measurements 
    for i in range(m):
        H[i,(i+1)*di-1] = 1

    return H

# compute a physical IC by running the model from t=-20 to t = 0
u0 = F * np.ones(n)  # Initial state (equilibrium)
u0[19] = u0[19] + 0.01  # Add small perturbation to 20th variable
#time integration
u0True = u0
nt1 = int(20/dt)
for k in range(nt1):
    u0True = RK4(Lorenz96,u0True,dt,F)

# Define observations
m = 9    
dt_m = 0.2 #time period between observations
tm_m = 20 #maximum time for observations
nt_m = int(tm_m/dt_m) #number of observation instants
ind_m = (np.linspace(int(dt_m/dt),int(tm_m/dt),nt_m)).astype(int)
t_m = t[ind_m]

sig_m= 0.1  # standard deviation for measurement noise
R = sig_m**2*np.eye(m) #covariance matrix for measurement noise

#time integration
uTrue = np.zeros([n,nt+1])
uTrue[:,0] = u0True
km = 0
w = np.zeros([m,nt_m])
for k in range(nt):
    uTrue[:,k+1] = RK4(Lorenz96,uTrue[:,k],dt,F)
    if (km<nt_m) and (k+1==ind_m[km]):
        w[:,km] = h(uTrue[:,k+1]) + np.random.normal(0,sig_m,[m,])
        km = km+1
        
########################### Data Assimilation #################################
# perturb IC
sig_b= 1
u0b = u0True + np.random.normal(0,sig_b,[n,])
B = sig_b**2*np.eye(n)

sig_p= 0.1
Q = sig_p**2*np.eye(n)

#time integration
ub = np.zeros([n,nt+1])
ub[:,0] = u0b
ua = np.zeros([n,nt+1])
ua[:,0] = u0b


# ensemble size 
N = 50
#initialize ensemble
uai = np.zeros([n,N])
for i in range(N):
    uai[:,i] = u0b + np.random.multivariate_normal(np.zeros(n), B)

km = 0
for k in range(nt):
    # Forecast Step
    #background trajectory [without correction]
    ub[:,k+1] = RK4(Lorenz96,ub[:,k],dt,F) 
    #EnKF trajectory [with correction at observation times]
    for i in range(N): # forecast ensemble
        uai[:,i] = RK4(Lorenz96,uai[:,i],dt,F) \
                 + np.random.multivariate_normal(np.zeros(n), Q)

    # compute the mean of forecast ensemble
    ua[:,k+1] = np.mean(uai,1)
    # compute forecast error covariance matrix
    B = (1/(N-1)) * (uai - ua[:,k+1].reshape(-1,1)) @ (uai - ua[:,k+1].reshape(-1,1)).T

    if (km<nt_m) and (k+1==ind_m[km]):
        # Analysis Step
        uai,B = EnKF(uai,w[:,km],h,Dh,R,B)
        # compute the mean of analysis ensemble
        ua[:,k+1] = np.mean(uai,1)    
        km = km+1


        
#%%
############################### Plotting ######################################
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font)

fig, ax = plt.subplots(nrows=3,ncols=1, figsize=(10,8))
ax = ax.flat

ax[0].plot(t,uTrue[8,:], label=r'\bf{True}', linewidth = 3, color='C0')
ax[0].plot(t,ub[8,:], ':', label=r'\bf{Background}', linewidth = 3, color='C1')
ax[0].plot(t,ua[8,:], '--', label=r'\bf{Analysis}', linewidth = 3, color='C3')
ax[0].set_xlabel(r'$t$',fontsize=22)
    
ax[1].plot(t,uTrue[17,:], label=r'\bf{True}', linewidth = 3, color='C0')
ax[1].plot(t,ub[17,:], ':', label=r'\bf{Background}', linewidth = 3, color='C1')
ax[1].plot(t,ua[17,:], '--', label=r'\bf{Analysis}', linewidth = 3, color='C3')
ax[1].set_xlabel(r'$t$',fontsize=22)

ax[2].plot(t,uTrue[35,:], label=r'\bf{True}', linewidth = 3, color='C0')
ax[2].plot(t,ub[35,:], ':', label=r'\bf{Background}', linewidth = 3, color='C1')
ax[2].plot(t[ind_m],w[8,:], 'o', fillstyle='none', \
               label=r'\bf{Observation}', markersize = 8, markeredgewidth = 2, color='C2')
ax[2].plot(t,ua[35,:], '--', label=r'\bf{Analysis}', linewidth = 3, color='C3')
ax[2].set_xlabel(r'$t$',fontsize=22)


ax[2].legend(loc="center", bbox_to_anchor=(0.5,4.2),ncol =4,fontsize=15)

ax[0].set_ylabel(r'$x_{9}(t)$')
ax[1].set_ylabel(r'$x_{18}(t)$', labelpad=9)
ax[2].set_ylabel(r'$x_{36}(t)$', labelpad=7)
fig.subplots_adjust(hspace=0.5)

plt.savefig('L96_EnKF.png', dpi = 500, bbox_inches = 'tight')
