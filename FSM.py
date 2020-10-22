# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:09:08 2020

@author: Shady
"""


import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt
from examples import *
from time_integrators import *

def fsm1st(rhs,Jrhs,ObsOp,JObsOp,t,ind_m,u0b,w,R,opt,*args):
    
    # Implementation of the first-order forward sensitivity method (FSM) to
    # correct the initial conditions based on the forecast sensitivity matrices

    # Inputs:
        #rhs: defines the right-hand side of the continuous time forward model f(u;mu)
        #Jrhs: defines the Jacobian matrix of rhs D_f(u)
        #ObsOp: defines the observation operator h(u)
        #JObsOp: defines the Jacobian of the observation operator D_h(u)
        #t: vector of time
        #ind_m: indices of measurement instants
        #u0b: initial condition for base trajectory
        #w: matrix of measurements
        #R: covariance matrix of measurement noise
        #opt: [0=euler] or [1=RK4] defines the time integration scheme to 
              #comptue the discrete-time forward map and its Jacobian
    # Output: the correction vector du0
    
    n = len(u0b)
    #determine the assimilation window
    t = t[:ind_m[-1]+1] #cut the time till the last observation point 
    nt = len(t)-1
    dt = t[1] - t[0]
    ub = np.zeros([n,nt+1]) #base trajectory   
    Ri = np.linalg.inv(R)

    ub[:,0] = u0b
    U = np.eye(n,n) #Initialization of U
    Q = np.zeros((1,n))  #Dh*U
    ef = np.zeros((1,1)) #w-h(u)
    W = np.zeros((1,1))  #weighting matrix
    km = 0
    nt_m = len(ind_m)
    if opt == 0: #Euler
        #forward model
        for k in range(nt):
            ub[:,k+1] = euler(rhs,ub[:,k],dt,*args)
            DM = Jeuler(rhs,Jrhs,ub[:,k],dt,*args)
            U = DM @ U
            if (km<nt_m) and (k+1==ind_m[km]):
                tmp = w[:,km] - ObsOp(ub[:,k+1])
                ek = tmp.reshape(-1,1)
                ef = np.vstack((ef,ek))
                Qk = JObsOp(ub[:,k+1]) @ U
                Q = np.vstack((Q,Qk))
                W = block_diag(W,Ri)
                km = km + 1
            
    elif opt == 1: #RK4
        # forward model
        for k in range(nt):
            ub[:,k+1] = RK4(rhs,ub[:,k],dt,*args)
            DM = JRK4(rhs,Jrhs,ub[:,k],dt,*args)
            U = DM @ U
            if (km<nt_m) and (k+1==ind_m[km]):
                tmp = w[:,km] - ObsOp(ub[:,k+1])
                ek = tmp.reshape(-1,1)
                ef = np.vstack((ef,ek))
                Qk = JObsOp(ub[:,k+1]) @ U
                Q = np.vstack((Q,Qk))
                W = block_diag(W,Ri)
                km = km + 1
            
    Q = np.delete(Q, (0), axis=0)
    ef = np.delete(ef, (0), axis=0)
    W = np.delete(W, (0), axis=0)
    W = np.delete(W, (0), axis=1)
    
    # solve weighted least-squares
    W1 = sqrtm(W) 
    du0 = np.linalg.lstsq(W1@Q, W1@ef, rcond=None)[0]

    return du0.ravel()


# Loss function (z-h(x))^T * R^{-1} * (z-h(x))
def loss(rhs,ObsOp,t,ind_m,u0,w,R,opt,*args):
    
    n = len(u0)
    #determine the assimilation window
    t = t[:ind_m[-1]+1] #cut the time till the last observation point 
    nt = len(t)-1
    dt = t[1] - t[0]
    u = np.zeros([n,nt+1]) #trajectory
    u[:,0] = u0

    Ri = np.linalg.inv(R)
    floss = 0 
    km = 0
    nt_m = len(ind_m)
    if opt == 0: #Euler
        #forward model
        for k in range(nt):
            u[:,k+1] = euler(rhs,u[:,k],dt,*args)
            
            if (km<nt_m) and (k+1==ind_m[km]):
                tmp = w[:,km] - ObsOp(u[:,k+1])
                tmp = tmp.reshape(-1,1)
                floss = floss + np.linalg.multi_dot(( tmp.T, Ri , tmp ))
                km = km + 1

    elif opt == 1: #RK4
        # forward model
        for k in range(nt):
            u[:,k+1] = RK4(rhs,u[:,k],dt,*args)
            if (km<nt_m) and (k+1==ind_m[km]):
                tmp = w[:,km] - ObsOp(u[:,k+1])
                tmp = tmp.reshape(-1,1)
                floss = floss + np.linalg.multi_dot(( tmp.T, Ri , tmp ))
                km = km + 1

    floss = floss[0,0]/2
    return floss
        

def GoldenAlpha(p,rhs,ObsOp,t,ind_m,u0,w,R,opt,*args):
    
    # p is the optimization direction
    a0=0
    b0=1
    r=(3-np.sqrt(5))/2
    
    uncert = 1e-5 # Specified uncertainty
    
    a1= a0 + r*(b0-a0);
    b1= b0 - r*(b0-a0);
    while (b0-a0) > uncert:
    
        if loss(rhs,ObsOp,t,ind_m,u0+a1*p,w,R,opt,*args)  <  loss(rhs,ObsOp,t,ind_m,u0+b1*p,w,R,opt,*args):  
            b0=b1;
            b1=a1;
            a1= a0 + r*(b0-a0);
        else:
            a0=a1;
            a1=b1;
            b1= b0 - r*(b0-a0);
    alpha = (b0+a0)/2

    return alpha

#%% Application: Lorenz 63

# parameters
sigma = 10.0     
beta = 8.0/3.0
rho = 28.0     
dt = 0.01
tm = 10 
nt = int(tm/dt)
t = np.linspace(0,tm,nt+1)

############################ Twin experiment ##################################
# Observation operator
def h(u):
    w = u
    return w

def Dh(u):
    n = len(u)
    D = np.eye(n)
    return D


u0True = np.array([1,1,1]) # True initial conditions
np.random.seed(seed=1)
sig_m= 0.15  # standard deviation for measurement noise
R = sig_m**2*np.eye(3) #covariance matrix for measurement noise

dt_m = 0.2 #time period between observations
tm_m = 2 #maximum time for observations
nt_m = int(tm_m/dt_m) #number of observation instants

ind_m = (np.linspace(int(dt_m/dt),int(tm_m/dt),nt_m)).astype(int)
t_m = t[ind_m]

#time integration
uTrue = np.zeros([3,nt+1])
uTrue[:,0] = u0True
km = 0
w = np.zeros([3,nt_m])
for k in range(nt):
    uTrue[:,k+1] = RK4(Lorenz63,uTrue[:,k],dt,sigma,beta,rho)
    if (km<nt_m) and (k+1==ind_m[km]):
        w[:,km] = h(uTrue[:,k+1]) + np.random.normal(0,sig_m,[3,])
        km = km+1
   
########################### Data Assimilation #################################
u0b = np.array([2.0,3.0,4.0])
u0a = u0b
J0 = loss(Lorenz63,h,t,ind_m,u0a,w,R,1,sigma,beta,rho)
for iter in range(200):
    
    #computing the gradient of cost functional with base trajectory
    du0 = fsm1st(Lorenz63,JLorenz63,h,Dh,t,ind_m,u0a,w,R,1,sigma,beta,rho)
    #minimization direction
    p = du0#/np.linalg.norm(du0) 
    #Golden method for linesearch
    alpha = GoldenAlpha(p,Lorenz63,h,t,ind_m,u0a,w,R,1,sigma,beta,rho)   
    #update initial condition with gradient descent
    u0a = u0a + alpha*p  
    #u0a = u0a + 0.25*du0
    J = loss(Lorenz63,h,t,ind_m,u0a,w,R,1,sigma,beta,rho)
    
    plt.plot(iter,J,'*')

    if np.abs(J0-J) < 1e-2:
        print('Convergence: loss function')
        print(u0a)
        print(iter)
        break
    #else:
        J0=J
    if np.linalg.norm(du0) < 1e-4:
        print('Convergence: correction vector')
        print(u0a)
        print(iter)
        break
#%%
##################### Time Integration [Comparison] ###########################

ub = np.zeros([3,nt+1])
ub[:,0] = u0b
ua = np.zeros([3,nt+1])
ua[:,0] = u0a
km = 0
for k in range(nt):
    ub[:,k+1] = RK4(Lorenz63,ub[:,k],dt,sigma,beta,rho)
    ua[:,k+1] = RK4(Lorenz63,ua[:,k],dt,sigma,beta,rho)


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

for k in range(3):
    ax[k].plot(t,uTrue[k,:], label=r'\bf{True}', linewidth = 3)
    ax[k].plot(t,ub[k,:], ':', label=r'\bf{Background}', linewidth = 3)
    ax[k].plot(t[ind_m],w[k,:], 'o', fillstyle='none', \
               label=r'\bf{Observation}', markersize = 8, markeredgewidth = 2)
    ax[k].plot(t,ua[k,:], '--', label=r'\bf{Analysis}', linewidth = 3)
    ax[k].set_xlabel(r'$t$',fontsize=22)
    ax[k].axvspan(0, tm_m, color='y', alpha=0.4, lw=0)

ax[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =4,fontsize=15)

ax[0].set_ylabel(r'$x(t)$', labelpad=5)
ax[1].set_ylabel(r'$y(t)$', labelpad=-12)
ax[2].set_ylabel(r'$z(t)$')
fig.subplots_adjust(hspace=0.5)

plt.savefig('L63_FSM.png', dpi = 500, bbox_inches = 'tight')

