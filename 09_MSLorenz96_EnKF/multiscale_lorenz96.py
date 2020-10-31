# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:41:10 2019

@author: Suraj
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps
import pyfftw

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
import time as clck
import os

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def rhs_dy_dt(J,u,y,fr,h,c,b,dt):
    v = np.zeros((ne+2,J+4))
    
    v[1:ne+1,2:J+2] = y[:,:]
    
    v[0,:] = v[ne,:]
    v[ne+1,:] = v[1,:]
    
    v[1:ne+2,1] = v[0:ne+1,J+1]
    v[1:ne+2,0] = v[0:ne+1,J]
    v[0:ne+1,J+2] = v[1:ne+2,2]
    v[0:ne+1,J+3] = v[1:ne+2,3]
    
    x = np.reshape(u,[-1,1])
    r = np.zeros((ne,J))
    
#    for i in range(ne):
#        for j in range(2,J+2):
#            r[i,j-2] = -b*c*v[i,j+1]*(v[i,j+2] - v[i,j+1]) - c*v[i,j] + h*c*x[i]/J
            
    r = -b*c*v[1:ne+1,3:J+3]*(v[1:ne+1,4:J+4] - v[1:ne+1,1:J+1]) - c*v[1:ne+1,2:J+2] + h*c*x/b
    
    return r*dt    
    
def rhs_dx_dt(ne,u,y,fr,h,c,dt):
    v = np.zeros(ne+3)
    v[2:ne+2] = u
    v[1] = v[ne+1]
    v[0] = v[ne]
    v[ne+2] = v[2]
    
    r = np.zeros(ne)
    
#    for i in range(2,ne+2):
#        r[i-2] = v[i-1]*(v[i+1] - v[i-2]) - v[i] + fr
    
    ysum = np.sum(y,axis=1)
    r = -v[1:ne+1]*(v[0:ne] - v[3:ne+3]) - v[2:ne+2] + fr - (h*c/b)*ysum
    
    return r*dt
    
    
def rk4c(ne,J,u,y,fr,h,c,b,dt):
    k1x = rhs_dx_dt(ne,u,y,fr,h,c,dt)
    k1y = rhs_dy_dt(J,u,y,fr,h,c,b,dt)
    
    k2x = rhs_dx_dt(ne,u+0.5*k1x,y+0.5*k1y,fr,h,c,dt)
    k2y = rhs_dy_dt(J,u+0.5*k1x,y+0.5*k1y,fr,h,c,b,dt)
    
    k3x = rhs_dx_dt(ne,u+0.5*k2x,y+0.5*k2y,fr,h,c,dt)
    k3y = rhs_dy_dt(J,u+0.5*k2x,y+0.5*k2y,fr,h,c,b,dt)
    
    k4x = rhs_dx_dt(ne,u+0.5*k3x,y+0.5*k3y,fr,h,c,dt)
    k4y = rhs_dy_dt(J,u+0.5*k3x,y+0.5*k3y,fr,h,c,b,dt)
    
    un = u + (k1x + 2.0*(k2x + k3x) + k4x)/6.0
    yn = y + (k1y + 2.0*(k2y + k3y) + k4y)/6.0
    
    return un,yn

def rk4uc(ne,J,u,y,fr,h,c,b,dt):
    k1x = rhs_dx_dt(ne,u,y,fr,h,c,dt)
    k2x = rhs_dx_dt(ne,u+0.5*k1x,y,fr,h,c,dt)
    k3x = rhs_dx_dt(ne,u+0.5*k2x,y,fr,h,c,dt)
    k4x = rhs_dx_dt(ne,u+0.5*k3x,y,fr,h,c,dt)
    
    # update y with an unupdated x
    k1y = rhs_dy_dt(J,u,y,fr,h,c,b,dt)    
    k2y = rhs_dy_dt(J,u,y+0.5*k1y,fr,h,c,b,dt)
    k3y = rhs_dy_dt(J,u,y+0.5*k2y,fr,h,c,b,dt)
    k4y = rhs_dy_dt(J,u,y+0.5*k3y,fr,h,c,b,dt)
    
    un = u + (k1x + 2.0*(k2x + k3x) + k4x)/6.0
    yn = y + (k1y + 2.0*(k2y + k3y) + k4y)/6.0
    
    return un,yn



#%% Main program:
ne = 36
J = 10
fr = 10.0
c = 10.0
b = 10.0
h = 1.0

fact = 0.1
std = 1.0

dt = 0.001
tmax = 20.0
tinit = 5.0
ns = int(tinit/dt)
nt = int(tmax/dt)

nf = 10         # frequency of observation
nb = int(nt/nf) # number of observation time
oib = [nf*k for k in range(nb+1)]

u = np.zeros(ne)
utrue = np.zeros((ne,nt+1))
uinit = np.zeros((ne,ns+1))
ysuminit = np.zeros((ne,ns+1))
ysum = np.zeros((ne,nt+1))
yall = np.zeros((ne,J,nt+1))

ti = np.linspace(-tinit,0,ns+1)
t = np.linspace(0,tmax,nt+1)
tobs = np.linspace(0,tmax,nb+1)
x = np.linspace(1,ne,ne)

X,T = np.meshgrid(x,t,indexing='ij')
Xi,Ti = np.meshgrid(x,ti,indexing='ij')

#%%
#-----------------------------------------------------------------------------#
# generate true solution trajectory
#-----------------------------------------------------------------------------#
u[:] = fr
u[int(ne/2)-1] = fr + 0.01
#u = 2*fr*np.random.random_sample(ne) - fr
uinit[:,0] = u

y = 2*fact*fr*np.random.random_sample((ne, J)) - fact*fr 

#%%

# generate initial condition at t = 0
for k in range(1,ns+1):
    un, yn = rk4uc(ne,J,u,y,fr,h,c,b,dt)
    uinit[:,k] = un
    ysuminit[:,k] = np.sum(yn,axis=1)
    u = np.copy(un)
    y = np.copy(yn)

# assign inital condition
u = uinit[:,-1]
utrue[:,0] = uinit[:,-1]
ysum[:,0] = ysuminit[:,-1]
yall[:,:,0] = yn

#initX = np.load('./Lorenz-Online/data/initX.npy')
#initY = np.load('./Lorenz-Online/data/initY.npy')
#y = np.reshape(initY,[ne,J])

# generate true forward solution
for k in range(1,nt+1):
    un, yn = rk4uc(ne,J,u,y,fr,h,c,b,dt)
    utrue[:,k] = un
    ysum[:,k] = np.sum(yn,axis=1)
    yall[:,:,k] = yn
    u = np.copy(un)
    y = np.copy(yn)

np.savez('data_cyclic.npz',utrue=utrue,ysum=ysum,yall=yall)
    
#%%
vmin = -12
vmax = 12
fig, ax = plt.subplots(2,1,figsize=(8,5))
cs = ax[0].contourf(Ti,Xi,uinit,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(uinit)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

cs = ax[1].contourf(T,X,utrue,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))

fig.tight_layout()
plt.show()

#%%
data = np.load('data_cyclic.npz')
yp = data['ysum']
fig, ax = plt.subplots(4,1,sharex=True,figsize=(10,6))

n = [0,17]
for i in range(2):
    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i+2].plot(t,ysum[n[i],:],'r-')
#    ax[i+2].plot(t,yp[n[i],:],'g-')
    ax[i].set_xlim([0,tmax])
    ax[i].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')
    ax[i+2].set_ylabel(r'$y_{'+str(n[i]+1)+'}$')
    
ax[i].set_xlabel(r'$t$')
line_labels = ['$X$','$Y$']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 