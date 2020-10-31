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

import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font)

#%%
data = np.load('data_s_16.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_8 = data['utrue']
uobs_8 = data['uobs']
uw_8 = data['uw']
ua_8 = data['ua']
oin8  = data['oin']

data = np.load('data_s_32.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_12 = data['utrue']
uobs_12 = data['uobs']
uw_12 = data['uw']
ua_12 = data['ua']
oin12  = data['oin']

data = np.load('data_s_64.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_20 = data['utrue']
uobs_20 = data['uobs']
uw_20 = data['uw']
ua_20 = data['ua']
oin20  = data['oin']

#%%
fig, ax = plt.subplots(3,3,figsize=(10,7),sharex=True,sharey=True)
ymin = -5
ymax = 5
n = [7,50,100]

c = 0
i = 0
ax[i,c].plot(t,utrue_8[n[i],:],label=r'\bf{True}', linewidth = 3)
ax[i,c].plot(t,uw_8[n[i],:],':', label=r'\bf{Background}', linewidth = 3)
ax[i,c].plot(t,ua_8[n[i],:],'--', label=r'\bf{Analysis}', linewidth = 3)
if i == 0:
    ax[i,c].plot(tobs,uobs_8[n[i],:],'o', fillstyle='none', \
           label=r'\bf{Observation}', markersize = 8, markeredgewidth = 2,zorder=0)

ax[i,c].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')
i=1
ax[i,c].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')
i=2
ax[i,c].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')
        
for i in range(1,3):
    ax[i,c].plot(t,utrue_8[n[i],:], linewidth = 3)
    ax[i,c].plot(t,uw_8[n[i],:],':',  linewidth = 3)
    ax[i,c].plot(t,ua_8[n[i],:],'--',  linewidth = 3)
    if i == 0:
        ax[i,c].plot(tobs,uobs_8[n[i],:],'o', fillstyle='none', \
                markersize = 8, markeredgewidth = 2,zorder=0)
    

    ax[i,c].set_xlim([0,np.max(t)])
#    ax[i,c].set_ylim([ymin,ymax])
#    ax[i,c].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')
    
ax[i,c].set_xlabel(r'$t$')

c = 1

for i in range(3):
    ax[i,c].plot(t,utrue_12[n[i],:],linewidth = 3)
    ax[i,c].plot(t,uw_12[n[i],:],':', linewidth = 3)
    ax[i,c].plot(t,ua_12[n[i],:],'--', linewidth = 3)
    if i == 0:
        ax[i,c].plot(tobs,uobs_8[n[i],:],'o', fillstyle='none', \
               markersize = 8, markeredgewidth = 2,zorder=0)
    
    ax[i,c].set_xlim([0,np.max(t)])
#    ax[i,c].set_ylim([ymin,ymax])
#    ax[i,c].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')
ax[i,c].set_xlabel(r'$t$')

c = 2
for i in range(3):
    ax[i,c].plot(t,utrue_20[n[i],:], linewidth = 3)
    ax[i,c].plot(t,uw_20[n[i],:],':', linewidth = 3)
    ax[i,c].plot(t,ua_20[n[i],:],'--', linewidth = 3)
    if i == 0:
        ax[i,c].plot(tobs,uobs_20[n[i],:],'o', fillstyle='none', \
                markersize = 8, markeredgewidth = 2,zorder=0)

    ax[i,c].set_xlim([0,np.max(t)])
#    ax[i,c].set_ylim([ymin,ymax])
#    ax[i,c].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')
ax[i,c].set_xlabel(r'$t$')

fig.subplots_adjust(top=0.2)
fig.legend(loc="center", bbox_to_anchor=(0.5,1.0),ncol =4,fontsize=15)

fig.tight_layout()
plt.show() 
fig.savefig('time_series_enkfs2.pdf',bbox_inches='tight')
fig.savefig('time_series_enkfs2.eps',bbox_inches='tight')
fig.savefig('time_series_enkfs2.png',bbox_inches='tight',dpi=300, pad_inches=0.2)