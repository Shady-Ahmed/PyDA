#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:04:03 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os
from numba import jit
from scipy.fftpack import dst, idst
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
font = {'family' : 'normal',
        'size'   : 16}
mpl.rc('font', **font)

def fst(nx,ny,dx,dy,f):
    fd = f[2:nx+3,2:ny+3]
    data = fd[1:-1,1:-1]
        
#    e = dst(data, type=2)
    data = dst(data, axis = 1, type = 1)
    data = dst(data, axis = 0, type = 1)
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
    
    data1 = np.zeros((nx-1,ny-1))
    
#    for i in range(1,nx):
#        for j in range(1,ny):
    alpha = (2.0/(dx*dx))*(np.cos(np.pi*m/nx) - 1.0) + (2.0/(dy*dy))*(np.cos(np.pi*n/ny) - 1.0)           
    data1 = data/alpha
    
#    u = idst(data1, type=2)/((2.0*nx)*(2.0*ny))
    data1 = idst(data1, axis = 1, type = 1)
    data1 = idst(data1, axis = 0, type = 1)
    
    u = data1/((2.0*nx)*(2.0*ny))
    
    ue = np.zeros((nx+5,ny+5))
    ue[3:nx+2,3:ny+2] = u
    
    return ue

#%%
re = 100.0
ro = 0.00175
npe = 20
nx = 128
ny = 256
pi = np.pi
lx = 1.0
ly = 2.0

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

filename = 'qg_'+ str(re) + '_' + str(ro) + '_' +str(npe)+ '_' + str(nx) + '_' + str(ny) + '.npz'
data = np.load(filename)
x = data['x']
y = data['y']
w_true = data['w_true']
w_wrong = data['w_wrong']
w_da = data['w_da']

#%%
shift = 0.5

npx = 16
npy = 32

xp = np.arange(0,npx)*int(nx/npx) + int(shift*nx/npx)
yp = np.arange(0,npy)*int(ny/npy) + int(shift*ny/npy) 
xprobe, yprobe = np.meshgrid(xp,yp)

#%%
w = w_true[:,:,50]
s = fst(nx, ny, dx, dy, -w)

fig, axs = plt.subplots(1,2,figsize=(11,5))

cs = axs[0].contourf(x,y,w[2:nx+3,2:ny+3],120,cmap='jet')
#cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
fig.colorbar(cs, ax=axs[0], orientation='vertical',)
for i in xp:
    for j in yp:
        axs[0].plot(x[i-1,j-1],y[i-1,j-1],'ko',ms=1)

cs = axs[1].contourf(x,y,s[2:nx+3,2:ny+3],120,cmap='jet')
#cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
fig.colorbar(cs, ax=axs[1], orientation='vertical')

for i in range(2):
    axs[i].set_xlabel('$x$')
    axs[i].set_ylabel('$y$')

    
fig.tight_layout()    
plt.show()
fig.savefig('qg_field.pdf',dpi=300)
fig.savefig('qg_field.png',dpi=300)

#%%
fig, ax = plt.subplots(2,3,figsize=(12,8))

axs = ax.flat     

cs = axs[0].contourf(x,y,w_true[2:nx+3,2:ny+3,0],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[0], orientation='vertical')
axs[0].set_title('True Initial')

cs = axs[1].contourf(x,y,w_wrong[2:nx+3,2:ny+3,0],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[1], orientation='vertical')
axs[1].set_title('Wrong Initial')

cs = axs[2].contourf(x,y,w_da[2:nx+3,2:ny+3,0],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[2], orientation='vertical')
axs[2].set_title('DA Initial')

cs = axs[3].contourf(x,y,w_true[2:nx+3,2:ny+3,-1],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[3], orientation='vertical')
axs[3].set_title('True Forecast')

cs = axs[4].contourf(x,y,w_wrong[2:nx+3,2:ny+3,-1],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[4], orientation='vertical')
axs[4].set_title('Wrong Forecast')

cs = axs[5].contourf(x,y,w_da[2:nx+3,2:ny+3,-1],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[5], orientation='vertical')
axs[5].set_title('DA Forecast')

plt.show()
fig.tight_layout()
fig.savefig('contour_da_all.png', dpi=200)

#%%
fig, ax = plt.subplots(3,3,figsize=(12,12))
axs = ax.flat    

k = 0
for i in [100,200,300]:
    cs = axs[k].contourf(x,y,w_true[2:nx+3,2:ny+3,i],120,cmap='jet',vmin=-600,vmax=400)
    k = k + 1
    cs = axs[k].contourf(x,y,w_da[2:nx+3,2:ny+3,i],120,cmap='jet',vmin=-600,vmax=400)
    k = k + 1
    diff = w_true[2:nx+3,2:ny+3,i] - w_da[2:nx+3,2:ny+3,i]
    cs = axs[k].contourf(x,y,diff,120,cmap='jet',vmin=-600,vmax=400)
    k = k +1

for i in range(9):
    axs[i].set_xlabel('$x$')
    axs[i].set_ylabel('$y$')

m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(w_true[2:nx+3,2:ny+3,100])
m.set_clim(-600, 400)

fig.subplots_adjust(bottom=0.4)
cbar_ax = fig.add_axes([0.25, -0.02, 0.5, 0.025])
fig.colorbar(m, cax=cbar_ax,orientation='horizontal')
    
fig.tight_layout()       
plt.show()
fig.savefig('qg_results_da.png', bbox_inches='tight',dpi=200)
 