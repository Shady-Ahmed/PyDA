#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:15:00 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(22)
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import random

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def rhs(ne,u,fr):
    v = np.zeros(ne+3)
    v[2:ne+2] = u
    v[1] = v[ne+1]
    v[0] = v[ne]
    v[ne+2] = v[2]
    
    r = np.zeros(ne)
    
#    for i in range(2,ne+2):
#        r[i-2] = v[i-1]*(v[i+1] - v[i-2]) - v[i] + fr
    
    r = v[1:ne+1]*(v[3:ne+3] - v[0:ne]) - v[2:ne+2] + fr
    
    return r
    
    
def rk4(ne,dt,u,fr):
    r1 = rhs(ne,u,fr)
    k1 = dt*r1
    
    r2 = rhs(ne,u+0.5*k1,fr)
    k2 = dt*r2
    
    r3 = rhs(ne,u+0.5*k2,fr)
    k3 = dt*r3
    
    r4 = rhs(ne,u+k3,fr)
    k4 = dt*r4
    
    un = u + (k1 + 2.0*(k2 + k3) + k4)/6.0
    
    return un
       
#%%
ne = 36

dt = 0.001
tmax = 10.0
tini = 15.0
ns = int(tini/dt)
nt = int(tmax/dt)
fr = 10.0
nf = 10         # frequency of observation
nb = int(nt/nf) # number of observation time

u = np.zeros(ne)
utrue = np.zeros((ne,nt+1))
uinit = np.zeros((ne,ns+1))

#-----------------------------------------------------------------------------#
# generate true solution trajectory
#-----------------------------------------------------------------------------#
ti = np.linspace(-tini,0,ns+1)
t = np.linspace(0,tmax,nt+1)
tobs = np.linspace(0,tmax,nb+1)
x = np.linspace(1,ne,ne)

X,T = np.meshgrid(x,t,indexing='ij')
Xi,Ti = np.meshgrid(x,ti,indexing='ij')

data = np.load('data_cyclic.npz')
uclosure_all = data['utrue']

nts = int(10.0/dt)

uclosure = uclosure_all[:,nts:]
utrue[:,0] = uclosure[:,0]

# generate true forward solution
for k in range(1,nt+1):
    u = utrue[:,k-1]
    un = rk4(ne,dt,u,fr)
    utrue[:,k] = un

#%%    
#fort = np.loadtxt('true_trajectory.plt',skiprows=1)
#u1p = utrue[19,:]
#u1f = fort[500:,2]
#aa = u1p - u1f
#
#field = np.loadtxt('true_field.plt',skiprows=2) 
#ufort = field[:,2].reshape(ns+nt+1,ne,order='f')
#
#ufort_i = ufort[:501,:].T
#ufort_e = ufort[500:,:].T
#
#aa = utrue - ufort_e

#%%
vmin = -12
vmax = 12
fig, ax = plt.subplots(2,1,figsize=(6,5))
cs = ax[0].contourf(T,X,uclosure,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(uinit)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

#cs = ax[1].contourf(T,X,ufort_e,cmap='jet',vmin=-vmin,vmax=vmax)
#m = plt.cm.ScalarMappable(cmap='jet')
#m.set_array(uinit)
#m.set_clim(vmin, vmax)
#fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

cs = ax[1].contourf(T,X,utrue,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))

fig.tight_layout()
plt.show()

#%%
#-----------------------------------------------------------------------------#
# generate observations
#-----------------------------------------------------------------------------#
mean = 0.0
sd2 = 1.0e0 # added noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)

oib = [nf*k for k in range(nb+1)]

uobs = uclosure[:,oib] + np.random.normal(mean,sd1,[ne,nb+1])

#-----------------------------------------------------------------------------#
# generate erroneous soltions trajectory
#-----------------------------------------------------------------------------#
uw = np.zeros((ne,nt+1))
k = 0
si2 = 1.0e-2
si1 = np.sqrt(si2)

u = utrue[:,0] + np.random.normal(mean,si1,ne)
uw[:,0] = u

for k in range(1,nt+1):
    un = rk4(ne,dt,u,fr)
    uw[:,k] = un
    u = np.copy(un)

#%%
#-----------------------------------------------------------------------------#
# EnKF model
#-----------------------------------------------------------------------------#    

# number of observation vector
me = 24
freq = int(ne/me)
oin = sorted(random.sample(range(ne), me)) 
roin = np.int32(np.linspace(0,me-1,me))
print(oin)

dh = np.zeros((me,ne))
dh[roin,oin] = 1.0

H = np.zeros((me,ne))
H[roin,oin] = 1.0

#%%
# number of ensemble 
npe = 50
cn = 1.0/np.sqrt(npe-1)

lambd = 1.05

z = np.zeros((me,nb+1))
#zf = np.zeros((me,npe,nb+1))
DhX = np.zeros((me,npe))
DhXm = np.zeros(me)

ua = np.zeros((ne,nt+1)) # mean analyssi solution (to store)
uf = np.zeros(ne)        # mean forecast
sc = np.zeros((ne,npe))   # square-root of the covariance matrix
Af = np.zeros((ne,npe))   # Af data
ue = np.zeros((ne,npe,nt+1)) # all ensambles
ph = np.zeros((ne,me))

km = np.zeros((ne,me))
kmd = np.zeros((ne,npe))

cc = np.zeros((me,me))
ci = np.zeros((me,me))

for k in range(nb+1):
    z[:,k] = uobs[oin,k]
#    for n in range(npe):
#        zf[:,n,k] = z[:,k] + np.random.normal(mean,sd1,me)

#%%
# initial ensemble
k = 0
se2 = 0.0 #np.sqrt(sd2)
se1 = np.sqrt(se2)

for n in range(npe):
    ue[:,n,k] = uw[:,k] + np.random.normal(mean,si1,ne)       
    
ua[:,k] = np.sum(ue[:,:,k],axis=1)
ua[:,k] = ua[:,k]/npe

kobs = 1

# RK4 scheme
for k in range(1,nt+1):
    
    # forecast afor all ensemble fields
    for n in range(npe):
        u[:] = ue[:,n,k-1]
        un = rk4(ne,dt,u,fr)
        ue[:,n,k] = un[:] + np.random.normal(mean,se1,ne)
    
    # mean analysis for plotting
    ua[:,k] = np.sum(ue[:,:,k],axis=1)
    ua[:,k] = ua[:,k]/npe
    
    if k == oib[kobs]:
        # compute mean of the forecast fields
        uf[:] = np.sum(ue[:,:,k],axis=1)   
        uf[:] = uf[:]/npe
        
        # compute Af dat
        for n in range(npe):
            Af[:,n] = ue[:,n,k] - uf[:]
        
        da = dh @ Af
        
        cc = da @ da.T/(npe-1)  
        
        for i in range(me):
            cc[i,i] = cc[i,i] + sd2 
        
        ci = np.linalg.pinv(cc)
        
        km = Af @ da.T @ ci/(npe-1)
        
#        pf = Af @ Af.T
#        pf[:,:] = pf[:,:]/(npe-1)
#        
#        dp = dh @ pf
#        cc = dp @ dh.T     
#
#        for i in range(me):
#            cc[i,i] = cc[i,i] + sd2     
#        
#        ph = pf @ dh.T
#        
#        ci = np.linalg.pinv(cc) # ci: inverse of cc matrix
#        
#        km = ph @ ci # compute Kalman gain
        
        # analysis update    
        kmd = km @ (z[:,kobs] - uf[oin])
        ua[:,k] = uf[:] + kmd[:]
        
        # ensemble correction
        ha = dh @ Af
        
        ue[:,:,k] = Af[:,:] - 0.5*(km @ dh @ Af) + ua[:,k].reshape(-1,1)
        
        #multiplicative inflation (optional): set lambda=1.0 for no inflation
        ue[:,:,k] = ua[:,k].reshape(-1,1) + lambd*(ue[:,:,k] - ua[:,k].reshape(-1,1))
        
        kobs = kobs+1

np.savez('data_'+str(me)+'_sr.npz',t=t,tobs=tobs,T=T,X=X,utrue=utrue,uobs=uobs,uw=uw,ua=ua,oin=oin)
    
#%%
fig, ax = plt.subplots(3,1,sharex=True,figsize=(6,5))

n = [9,14,34]
for i in range(3):
    if i == 0:
        ax[i].plot(tobs,uobs[n[i],:],'ro',fillstyle='none', markersize=2,markeredgewidth=1)
        
#    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i].plot(t,uclosure[n[i],:],'k-')
    ax[i].plot(t,uw[n[i],:],'b--')
    ax[i].plot(t,ua[n[i],:],'g-.')
    

    ax[i].set_xlim([0,tmax])
    ax[i].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')

ax[i].set_xlabel(r'$t$')
line_labels = ['Observation','True','Wrong','EnKF']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 
fig.savefig('m_'+str(me)+'.pdf')

#%%
vmin = -12
vmax = 12
fig, ax = plt.subplots(3,1,figsize=(6,7.5))

cs = ax[0].contourf(T,X,uclosure,30,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))
ax[0].set_title('True')

cs = ax[1].contourf(T,X,ua,30,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(ua)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))
ax[1].set_title('DEnKF')

diff = ua - uclosure
cs = ax[2].contourf(T,X,diff,30,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(ua)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))
ax[2].set_title('Difference')

fig.tight_layout()
plt.show() 
fig.savefig('fc_'+str(me)+'_'+str(sd2)+'.png',dpi=300)   

print(np.linalg.norm(diff))
    

 
























































