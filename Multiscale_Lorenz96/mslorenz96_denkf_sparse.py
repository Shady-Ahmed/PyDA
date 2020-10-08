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

ti = np.linspace(-tinit,0,ns+1)
t = np.linspace(0,tmax,nt+1)
tobs = np.linspace(0,tmax,nb+1)
x = np.linspace(1,ne,ne)

X,T = np.meshgrid(x,t,indexing='ij')
Xi,Ti = np.meshgrid(x,ti,indexing='ij')

ttrain = 10.0
ntrain = int(ttrain/dt)

#%%    
data = np.load('data_cyclic.npz')
utrue = data['utrue']
ysum = data['ysum']
yall = data['yall']

mean = 0.0
sd2 = 1.0e0 # added noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)

uobsfull = utrue[:,:] + np.random.normal(mean,sd1,[ne,nt+1])


#%%
tstart = 10.0
tend = 20.0
nts = int(tstart/dt) 
ntest = int((tend-tstart)/dt)

#%%
nt = ntest
nf = 10         # frequency of observation
nb = int(nt/nf) # number of observation time
oib = [nf*k for k in range(nb+1)]

uobs = uobsfull[:,int(nts):][:,oib]
tobs = np.array([tstart+nf*k*dt for k in range(nb+1)])

uw = np.zeros((ne,nt+1))
k = 0
mean = 0.0

si2 = 1.0e-2
si1 = np.sqrt(si2)

u = utrue[:,nts] + np.random.normal(mean,si1,ne)
uw[:,k] = u
y = yall[:,:,nts]

#%%
for k in range(1,nt+1):
    un, yn = rk4uc(ne,J,u,y,fr,h,c,b,dt)
    uw[:,k] = un
    u = np.copy(un)
    y = np.copy(yn)

#%%
print('---------- Solution with wrong initial condition ----------------')
t = np.linspace(tstart,tend,nt+1)
x = np.linspace(1,ne,ne)

X,T = np.meshgrid(x,t,indexing='ij')

vmin = -12
vmax = 12
fig, ax = plt.subplots(3,1,figsize=(6,7.5))
cs = ax[0].contourf(T,X,utrue[:,nts:],40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))
ax[0].set_title('True')

cs = ax[1].contourf(T,X,uw,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))
ax[0].set_title('Wrong')

diff = utrue[:,nts:] - uw
cs = ax[2].contourf(T,X,diff,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))
ax[0].set_title('Difference')

fig.tight_layout()
plt.show()

print(np.linalg.norm(diff))

#np.savez('data_'+str(0)+'.npz',T=T,X=X,utrue=utrue,uw=uw)

#%%
#-----------------------------------------------------------------------------#
# EnKF model
#-----------------------------------------------------------------------------#    
# number of observation vector
me = 18
freq = int(ne/me)
oin = sorted(random.sample(range(ne), me)) 
roin = np.int32(np.linspace(0,me-1,me))
#print(oin)

dh = np.zeros((me,ne))
dh[roin,oin] = 1.0

H = np.zeros((me,ne))
H[roin,oin] = 1.0

#%%
# number of ensemble 
npe = 20
cn = 1.0/np.sqrt(npe-1)
lambd = 1.0

z = np.zeros((me,nb+1))
#zf = np.zeros((me,npe,nb+1))
DhX = np.zeros((me,npe))
DhXm = np.zeros(me)

ua = np.zeros((ne,nt+1)) # mean analyssi solution (to store)
uf = np.zeros(ne)        # mean forecast
sc = np.zeros((ne,npe))   # square-root of the covariance matrix
Af = np.zeros((ne,npe))   # Af data
ue = np.zeros((ne,npe,nt+1)) # all ensambles
yse = np.zeros((ne,J,npe,nt+1)) # all ensambles
ph = np.zeros((ne,me))

km = np.zeros((ne,me))
kmd = np.zeros((ne,npe))

cc = np.zeros((me,me))
ci = np.zeros((me,me))

for k in range(nb+1):
    z[:,k] = uobs[oin,k]


#%%
# initial ensemble
k = 0
se2 = 0.0 #np.sqrt(sd2)
se1 = np.sqrt(se2)

for n in range(npe):
    ue[:,n,k] = uw[:,k] + np.random.normal(mean,si1,ne)       
    yse[:,:,n,k] = yall[:,:,k+nts]  + np.random.normal(mean,si1,[ne,J])     

#%%    
ua[:,k] = np.sum(ue[:,:,k],axis=1)
ua[:,k] = ua[:,k]/npe

kobs = 1

print('---------- Data assimilation stage ----------------')
# RK4 scheme
for k in range(1,nt+1):
    
    # forecast afor all ensemble fields
    for n in range(npe):
        u = ue[:,n,k-1] 
        y = yse[:,:,n,k-1] 
        un, yn = rk4uc(ne,J,u,y,fr,h,c,b,dt)
        ue[:,n,k] = un
        yse[:,:,n,k] = yn
              
    # mean analysis for plotting
    ua[:,k] = np.sum(ue[:,:,k],axis=1)
    ua[:,k] = ua[:,k]/npe
    
    if k == oib[kobs]:
#        print(k)
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

        # analysis update    
        kmd = km @ (z[:,kobs] - uf[oin])
        ua[:,k] = uf[:] + kmd[:]
        
        # ensemble correction
        ha = dh @ Af
        
        ue[:,:,k] = Af[:,:] - 0.5*(km @ dh @ Af) + ua[:,k].reshape(-1,1)
        
        #multiplicative inflation (optional): set lambda=1.0 for no inflation
        #ue[:,:,k] = ua[:,k] + lambd*(ue[:,:,k] - ua[:,k])
        
        kobs = kobs+1

np.savez('data_'+str(me)+'.npz',t=t,tobs=tobs,T=T,X=X,utrue=utrue,uobs=uobs,uw=uw,ua=ua,oin=oin)
    
#%%
fig, ax = plt.subplots(3,1,sharex=True,figsize=(6,5))

n = [9,14,34]
for i in range(3):
    if i == 0:
        ax[i].plot(tobs,uobs[n[i],:],'ro',fillstyle='none', markersize=3,markeredgewidth=1)
    ax[i].plot(t,utrue[n[i],nts:],'k-')
    ax[i].plot(t,uw[n[i],:],'b--')
    ax[i].plot(t,ua[n[i],:],'g-.')
    

    ax[i].set_xlim([np.min(t),np.max(t)])
    ax[i].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')

ax[i].set_xlabel(r'$t$')
line_labels = ['Observation','True','Wrong','EnKF']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 
fig.savefig('m_'+str(me)+'_5.png')

#%%
vmin = -12
vmax = 12
fig, ax = plt.subplots(3,1,figsize=(6,7.5))

cs = ax[0].contourf(T,X,utrue[:,nts:],40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))
ax[0].set_title('True')

cs = ax[1].contourf(T,X,ua,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(ua)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))
ax[1].set_title('DEnKF')

diff = ua - utrue[:,nts:]
cs = ax[2].contourf(T,X,diff,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(ua)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))
ax[2].set_title('Difference')

fig.tight_layout()
plt.show() 
fig.savefig('f_'+str(me)+'_5.png',dpi=300)   

print(np.linalg.norm(diff))


l2norm = np.array([me,nf,sd2,np.linalg.norm(diff)])
f=open('l2norm.dat','ab')
np.savetxt(f,l2norm.reshape([1,-1]))    
f.close()

