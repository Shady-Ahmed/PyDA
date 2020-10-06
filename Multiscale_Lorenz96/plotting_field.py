#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:17:22 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
data = np.load('data_cyclic.npz')
uclosure = data['utrue']
nts = 10000
utrue = uclosure[:,nts:]

data = np.load('data_9_sr.npz')
T = data['T']
X = data['X']
utrue_9 = data['utrue']
uw_9 = data['uw']
ua_9 = data['ua']

data = np.load('data_18_sr.npz')
T = data['T']
X = data['X']
utrue_18 = data['utrue']
uw_18 = data['uw']
ua_18 = data['ua']

data = np.load('data_36_sr.npz')
T = data['T']
X = data['X']
utrue_36 = data['utrue']
uw_36 = data['uw']
ua_36 = data['ua']


diff_36 = utrue - ua_36
diff_9 = utrue - ua_9
diff_18 = utrue - ua_18

#%%
vmin = -12
vmax = 12
fig, ax = plt.subplots(7,1,figsize=(8,13.4),sharex=True)

axs = ax.flat

field = [utrue,ua_9,ua_18,ua_36,diff_9,diff_18,diff_36]
#label = ['True','True','True','EnKF','EnKF','EnKF','Error','Error','Error']
title = ['(a) True', '(b) DEnKF ($m=9$)', '(c) DEnKF ($m=18$)', '(d) DEnKF ($m=36$)', '(e) Error (DEnKF ($m=9$))',
         '(f) Error (DEnKF ($m=18$))','(g) Error (DEnKF ($m=36$))']
ylabel = [r'$u$',r'$u$',r'$u$',r'$u$',r'$\epsilon$',r'$\epsilon$',r'$\epsilon$']


for i in range(7):
    cs = axs[i].contourf(T,X,field[i],60,cmap='viridis',vmin=vmin,vmax=vmax,zorder=-9)
    axs[i].set_rasterization_zorder(-1)
#    axs[i].set_title(label[i])
#    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(ylabel[i],size=20)
    for c in cs.collections:
        c.set_edgecolor("face")
    axs[i].set_title(title[i],size=16)

axs[i].set_xlabel(r'$t$',size=20)
m = plt.cm.ScalarMappable(cmap='viridis')
m.set_array(utrue)
m.set_clim(vmin, vmax)
#fig.colorbar(m,ax=axs[0],ticks=np.linspace(vmin, vmax, 6))

fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.025])
fig.colorbar(m, cax=cbar_ax,orientation='horizontal')

fig.tight_layout()
plt.show() 
fig.savefig('field_plot_no_closure.pdf',bbox_inches='tight',dpi=300)
fig.savefig('field_plot_no_closure.eps',bbox_inches='tight')
fig.savefig('field_plot_no_closure.png',bbox_inches='tight',dpi=300)

#%%
ne,nt = utrue.shape
l2norm = np.zeros((3,2))
l2norm[0,0], l2norm[1,0], l2norm[2,0] = 9, 18, 36
l2norm[2,1] = np.linalg.norm(diff_36)/np.sqrt(ne*nt)
l2norm[0,1] = np.linalg.norm(diff_9)/np.sqrt(ne*nt)
l2norm[1,1] = np.linalg.norm(diff_18)/np.sqrt(ne*nt)
np.savetxt('l2norm.csv',l2norm,delimiter=',')
