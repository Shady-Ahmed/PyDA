#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:16:04 2020

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

from scipy.ndimage import gaussian_filter

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


#%%
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
# fast poisson solver using second-order central difference scheme
def fps(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[:] = hx*np.float64(np.arange(0, nx))

    ky[:] = hy*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f[2:nx+2,2:ny+2],0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
    
    #periodicity
    u = np.empty((nx+5,ny+5)) 
    u[2:nx+2,2:ny+2] = ut
    u[:,ny+2] = u[:,2]
    u[nx+2,:] = u[2,:]
    u[nx+2,ny+2] = u[2,2]
    
    return u

#%%
# set periodic boundary condition for ghost nodes. Index 0 and (n+2) are the ghost boundary locations
def bc(nx,ny,u):
    u[:,0] = 0.0
    u[:,1] = 0.0
    u[:,2] = 0.0
    u[:,ny+2] = 0.0
    u[:,ny+3] = 0.0
    u[:,ny+4] = 0.0
    
    u[0,:] = 0.0
    u[1,:] = 0.0
    u[2,:] = 0.0
    u[nx+2,:] = 0.0
    u[nx+3,:] = 0.0
    u[nx+4,:] = 0.0
    
    return u

#%%
def grad_spectral(nx,ny,u):
    
    '''
    compute the gradient of u using spectral differentiation
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    u : solution field 
    
    Output
    ------
    ux : du/dx (size = [nx+1,ny+1])
    uy : du/dy (size = [nx+1,ny+1])
    '''
    
    ux = np.empty((nx+1,ny+1))
    uy = np.empty((nx+1,ny+1))
    
    uf = np.fft.fft2(u[0:nx,0:ny])

    kx = np.fft.fftfreq(nx,1/nx)
    ky = np.fft.fftfreq(ny,1/ny)
    
    kx = kx.reshape(nx,1)
    ky = ky.reshape(1,ny)
    
    uxf = 1.0j*kx*uf
    uyf = 1.0j*ky*uf 
    
    ux[0:nx,0:ny] = np.real(np.fft.ifft2(uxf))
    uy[0:nx,0:ny] = np.real(np.fft.ifft2(uyf))
    
    # periodic bc
    ux[:,ny] = ux[:,0]
    ux[nx,:] = ux[0,:]
    ux[nx,ny] = ux[0,0]
    
    # periodic bc
    uy[:,ny] = uy[:,0]
    uy[nx,:] = uy[0,:]
    uy[nx,ny] = uy[0,0]
    
    return ux,uy

#%%
def les_filter(nx,ny,nxc,nyc,u):
    
    '''
    coarsen the solution field keeping the size of the data same
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    u : solution field on fine grid
    
    Output
    ------
    uc : coarsened solution field [nx+1, ny+1]
    '''
    
    uf = np.fft.fft2(u[0:nx,0:ny])
        
    uf[int(nxc/2):int(nx-nxc/2),:] = 0.0
    uf[:,int(nyc/2):int(ny-nyc/2)] = 0.0 
    utc = np.real(np.fft.ifft2(uf))
    
    uc = np.zeros((nx+1,ny+1))
    uc[0:nx,0:ny] = utc
    
    # periodic bc
    uc[:,ny] = uc[:,0]
    uc[nx,:] = uc[0,:]
    uc[nx,ny] = uc[0,0]
    
    return uc

#%%  
def dyn_smag(nx,ny,kappa,sc,wc):
    '''
    compute the eddy viscosity using Germanos dynamics procedure with Lilys 
    least square approximation
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    kapppa : sub-filter grid filter ratio
    wc : vorticity on LES grid
    sc : streamfunction on LES grid
    
    Output
    ------
    ev : (cs*delta)**2*|S| (size = [nx+1,ny+1])
    '''
    
    nxc = int(nx/kappa) 
    nyc = int(ny/kappa)
    
    scc = les_filter(nx,ny,nxc,nyc,sc[2:nx+3,2:ny+3])
    wcc = les_filter(nx,ny,nxc,nyc,wc[2:nx+3,2:ny+3])
    
    scx,scy = grad_spectral(nx,ny,sc[2:nx+3,2:ny+3])
    wcx,wcy = grad_spectral(nx,ny,wc[2:nx+3,2:ny+3])
    
    wcxx,wcxy = grad_spectral(nx,ny,wcx)
    wcyx,wcyy = grad_spectral(nx,ny,wcy)
    
    scxx,scxy = grad_spectral(nx,ny,scx)
    scyx,scyy = grad_spectral(nx,ny,scy)
    
    dac = np.sqrt(4.0*scxy**2 + (scxx - scyy)**2) # |\bar(s)|
    dacc = les_filter(nx,ny,nxc,nyc,dac)        # |\tilde{\bar{s}}| = \tilde{|\bar(s)|}
    
    sccx,sccy = grad_spectral(nx,ny,scc)
    wccx,wccy = grad_spectral(nx,ny,wcc)
    
    wccxx,wccxy = grad_spectral(nx,ny,wccx)
    wccyx,wccyy = grad_spectral(nx,ny,wccy)
    
    scy_wcx = scy*wcx
    scx_wcy = scx*wcy
    
    scy_wcx_c = les_filter(nx,ny,nxc,nyc,scy_wcx)
    scx_wcy_c = les_filter(nx,ny,nxc,nyc,scx_wcy)
    
    h = (sccy*wccx - sccx*wccy) - (scy_wcx_c - scx_wcy_c)
    
    t = dac*(wcxx + wcyy)
    tc = les_filter(nx,ny,nxc,nyc,t)
    
    m = kappa**2*dacc*(wccxx + wccyy) - tc
    
    hm = h*m
    mm = m*m
    
    CS2 = (np.sum(0.5*(hm + abs(hm)))/np.sum(mm))
    
    ev = CS2*dac
    
    return ev

#%%
def stat_smag(nx,ny,dx,dy,s,cs):
        
        
    dsdxy = (1.0/(4.0*dx*dy))*(s[1:nx+2,1:ny+2] + s[3:nx+4,3:ny+4] \
                                             -s[3:nx+4,1:ny+2] - s[1:nx+2,3:ny+4])
    
    dsdxx = (1.0/(dx*dx))*(s[3:nx+4,2:ny+3] - 2.0*s[2:nx+3,2:ny+3] \
                                         +s[1:nx+2,2:ny+3])
    
    dsdyy = (1.0/(dy*dy))*(s[2:nx+3,3:ny+4] - 2.0*s[2:nx+3,2:ny+3] \
                                         +s[2:nx+3,1:ny+2])
    
    ev = cs*cs*dx*dy*np.sqrt(4.0*dsdxy*dsdxy + (dsdxx-dsdyy)*(dsdxx-dsdyy))
    
    return ev    

#%% 
# compute rhs using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def rhs_arakawa(nx,ny,dx,dy,re,ro,st,w,s,fs,ifm):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    dd = 1.0/(2.0*dx)
    
    f = np.zeros((nx+5,ny+5))
    
    #Arakawa    
    j1 = gg*( (w[3:nx+4,2:ny+3]-w[1:nx+2,2:ny+3])*(s[2:nx+3,3:ny+4]-s[2:nx+3,1:ny+2]) \
             -(w[2:nx+3,3:ny+4]-w[2:nx+3,1:ny+2])*(s[3:nx+4,2:ny+3]-s[1:nx+2,2:ny+3]))

    j2 = gg*( w[3:nx+4,2:ny+3]*(s[3:nx+4,3:ny+4]-s[3:nx+4,1:ny+2]) \
            - w[1:nx+2,2:ny+3]*(s[1:nx+2,3:ny+4]-s[1:nx+2,1:ny+2]) \
            - w[2:nx+3,3:ny+4]*(s[3:nx+4,3:ny+4]-s[1:nx+2,3:ny+4]) \
            + w[2:nx+3,1:ny+2]*(s[3:nx+4,1:ny+2]-s[1:nx+2,1:ny+2]))
    
    j3 = gg*( w[3:nx+4,3:ny+4]*(s[2:nx+3,3:ny+4]-s[3:nx+4,2:ny+3]) \
            - w[1:nx+2,1:ny+2]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[1:nx+2,3:ny+4]*(s[2:nx+3,3:ny+4]-s[1:nx+2,2:ny+3]) \
            + w[3:nx+4,1:ny+2]*(s[3:nx+4,2:ny+3]-s[2:nx+3,1:ny+2]) )

    jac = (j1+j2+j3)*hh
    
    lap = aa*(w[3:nx+4,2:ny+3]-2.0*w[2:nx+3,2:ny+3]+w[1:nx+2,2:ny+3]) \
        + bb*(w[2:nx+3,3:ny+4]-2.0*w[2:nx+3,2:ny+3]+w[2:nx+3,1:ny+2])
    
    cor = dd*(s[3:nx+4,2:ny+3]-s[1:nx+2,2:ny+3])
    
    if ifm == 0:
#        f[2:nx+3,2:ny+3] = -jac + lap/re + fs[2:nx+3,2:ny+3] + cor/ro #+ st*w[2:nx+3,2:ny+3] 
        f[3:nx+2,3:ny+2] = -jac[1:nx,1:ny] + lap[1:nx,1:ny]/re + fs[3:nx+2,3:ny+2] + cor[1:nx,1:ny]/ro
        
    elif ifm == 1:
        kappa = 2
        ev = dyn_smag(nx,ny,kappa,s,w)
        f[2:nx+3,2:ny+3] = -jac + lap/re + ev*lap
                        
    return f

#%%
# set initial condition for decay of turbulence problem
def ic_decay(nx,ny,dx,dy):
    #w = np.empty((nx+3,ny+3))
    
    epsilon = 1.0e-6
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    ksi = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    eta = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    
    phase = np.zeros((nx,ny), dtype='complex128')
    wf =  np.empty((nx,ny), dtype='complex128')
    
    phase[1:int(nx/2),1:int(ny/2)] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,1:int(ny/2)] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]))

    phase[1:int(nx/2),ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                 eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                eta[1:int(nx/2),1:int(ny/2)]))

    k0 = 10.0
    c = 4.0/(3.0*np.sqrt(np.pi)*(k0**5))           
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es = c*(kk**4)*np.exp(-(kk/k0)**2)
    wf[:,:] = np.sqrt((kk*es/np.pi)) * phase[:,:]*(nx*ny)
            
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    ut = np.real(fft_object_inv(wf)) 
    
    #w = np.zeros((nx+3,ny+3))
    
    #periodicity
    w = np.zeros((nx+5,ny+5)) 
    w[2:nx+2,2:ny+2] = ut
    w[:,ny+2] = w[:,2]
    w[nx+2,:] = w[2,:]
    w[nx+2,ny+2] = w[2,2] 
    
    w = bc(nx,ny,w)    
    
    return w

#%%
# compute the energy spectrum numerically
def energy_spectrum(nx,ny,w):
    epsilon = 1.0e-6

    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    wf = fft_object(w[2:nx+2,2:ny+2]) 
    
    es =  np.empty((nx,ny))
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk
    
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    
    en = np.zeros(n+1)
    
    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.size
        ii = ii+1
        jj = jj+1
        en[k] = np.sum(es[ii,jj])
#        for i in range(1,nx):
#            for j in range(1,ny):          
#                kk1 = np.sqrt(kx[i,j]**2 + ky[i,j]**2)
#                if ( kk1>(k-0.5) and kk1<(k+0.5) ):
#                    ic = ic+1
#                    en[k] = en[k] + es[i,j]
                    
        en[k] = en[k]/ic
        
    return en, n

#%%
def export_data(nx,ny,re,ro,n,w,s,isolver,ifm):
    if isolver == 1:
        if ifm == 0:
            folder = 'data_'+str(re) + '_' + str(ro) + '_' + str(nx) + '_' + str(ny)
        elif ifm == 1:
            folder = 'data_'+str(int(re)) + '_' +'dsm' + '_'+ str(nx) + '_' + str(ny)
        
    if not os.path.exists('./results_da/'+folder):
        os.makedirs('./results_da/'+folder)
        
    filename = './results_da/'+folder+'/results_' + str(int(n))+'.npz'
    np.savez(filename,w=w)#,s=s)
    
def export_data_wrong(nx,ny,re,ro,n,w,s,isolver,ifm):
    if isolver == 1:
        if ifm == 0:
            folder = 'data_'+str(re) + '_' + str(ro) + '_' + str(nx) + '_' + str(ny)
        elif ifm == 1:
            folder = 'data_'+str(int(re)) + '_' +'dsm' + '_'+ str(nx) + '_' + str(ny)
        
    if not os.path.exists('./results_wrong/'+folder):
        os.makedirs('./results_wrong/'+folder)
        
    filename = './results_wrong/'+folder+'/results_' + str(int(n))+'.npz'
    np.savez(filename,w=w)#,s=s)        
    
#%% 
# read input file
l1 = []
with open('input_da.txt') as f:
    for l in f:
        l1.append((l.strip()).split("\t"))

nx = np.int64(l1[0][0])
ny = np.int64(l1[1][0])
re = np.float64(l1[2][0])
ro = np.float64(l1[3][0])
st = np.float64(l1[4][0])
nt = np.int64(l1[5][0])
dt = np.float64(l1[6][0])
ns = np.int64(l1[7][0])
isolver = np.int64(l1[8][0])
ifm = np.int64(l1[9][0])
ipr = np.int64(l1[10][0])
npe = np.int64(l1[11][0])
npx = np.int64(l1[12][0])
npy = np.int64(l1[13][0])
sd2 = np.float64(l1[14][0])
nf = np.int64(l1[15][0])         # frequency of observation

freq = int(nt/ns)
filename_sn = 'data_'+str(re)+'_'+str(ro)+'_'+str(nx)+'_'+str(ny)

#%% 
# assign parameters
tmax = nt*dt

nb = int(nt/nf) # number of observation time
ns = int(nt/nf)

freq = int(nt/ns)

nx_dns = nx
ny_dns = ny

pi = np.pi
lx = 1.0
ly = 2.0

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

ifile = 0
time = 0.0

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(-0.5*ly,0.5*ly,ny+1)

x, y = np.meshgrid(x, y, indexing='ij')

#%% 
# QG parameters
fs = np.zeros((nx+5, ny+5))

fs[2:nx+3,2:ny+3] = (1.0/ro)*np.sin(np.pi*y)

#%% 
# DA parameters
ne = (nx-1)*(ny-1)
me = npx*npy
mean = 0.0
sd1 = np.sqrt(sd2)

shift = 0.5

xp = np.arange(0,npx)*int(nx/npx) + int(shift*nx/npx)
yp = np.arange(0,npy)*int(ny/npy) + int(shift*ny/npy) -1
xprobe, yprobe = np.meshgrid(xp,yp)

oin = []
for i in xp:
    for j in yp:
        n = i*(nx-1) + j
        oin.append(n)
#        print(n)

roin = np.int32(np.linspace(0,npx*npy-1,npx*npy))
dh = np.zeros((me,ne))
dh[roin,oin] = 1.0

oib = [nf*k for k in range(nb+1)]

z = np.zeros((me,ns+1))

ist_sn = 5000

for k in range(ns+1):
    snap_id = ist_sn + k
    data = np.load('./results/'+filename_sn+'/results_'+str(snap_id)+'.npz')
    wobs = data['w']
    wo = np.reshape(wobs,[nx+5,ny+5])
    wobs_tr = np.reshape(wo[3:nx+2,3:ny+2], [ne])
    z[:,k] = wobs_tr[oin] + np.random.normal(mean,sd1,[me])
    
kobs = 1

#%% 
# allocate the vorticity and streamfunction arrays
we = np.zeros(((nx+5)*(ny+5),npe)) 
wet = np.zeros(((nx-1)*(ny-1),npe,1)) 

wen = np.zeros((nx+5,ny+5,npe)) 
sen = np.zeros((nx+5,ny+5,npe))

ten = np.zeros((nx+5,ny+5,npe))
ren = np.zeros((nx+5,ny+5,npe))

k = 0

n_snaps = 200
aa = np.arange(n_snaps)
np.random.shuffle(aa)
rand_sn = aa[:npe]

for n in range(npe):
    snap_id = ist_sn - rand_sn[n]
    data = np.load('./results/'+filename_sn+'/results_'+str(snap_id)+'.npz')
    w0 = data['w']
        
    wen[:,:,n] = w0

    sen[:,:,n] = fst(nx, ny, dx, dy, -wen[:,:,n])
    sen[:,:,n] = bc(nx,ny,sen[:,:,n])
       
    we[:,n] = np.reshape(w0,[(nx+5)*(ny+5)])    

wa = np.zeros(((nx+5)*(ny+5),nt+1))
wat = np.zeros(((nx-1)*(ny-1),nt+1))

wa[:,k] = np.sum(we[:,:],axis=1)
wa[:,k] = wa[:,k]/npe

data = np.load('./results/'+filename_sn+'/results_'+str(ist_sn)+'.npz')
w_true = data['w']
   
#%% 
def rhs(nx,ny,dx,dy,re,ro,st,w,s,fs,ifm,isolver):
    if isolver == 1:
        return rhs_arakawa(nx,ny,dx,dy,re,ro,st,w,s,fs,ifm)

# time integration using third-order Runge Kutta method
aa = 1.0/3.0
bb = 2.0/3.0
clock_time_init = tm.time()

#%%
w_wrong = np.zeros((nx+5,ny+5,ns+1))
w_wrong[:,:,0] = np.reshape(wa[:,0],[nx+5,ny+5])

#%%
fig, axs = plt.subplots(1,2,figsize=(11,5))
    
cs = axs[0].contourf(x,y,w_true[2:nx+3,2:ny+3],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[0], orientation='vertical')
axs[0].set_title('True')

cs = axs[1].contourf(x,y,w_wrong[2:nx+3,2:ny+3,0],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[1], orientation='vertical')
axs[1].set_title('Wrong')

plt.show()

ww = np.zeros((nx+5,ny+5)) 
sw = np.zeros((nx+5,ny+5))

ww = np.copy(w_wrong[:,:,0])
sw[:,:] = fst(nx, ny, dx, dy, -ww[:,:])
sw[:,:] = bc(nx,ny,sw[:,:])
    
tw = np.zeros((nx+5,ny+5))
rw = np.zeros((nx+5,ny+5))

for k in range(1,nt+1):
    time = time + dt
    rw[:,:] = rhs(nx,ny,dx,dy,re,ro,st,ww[:,:],sw[:,:],fs,ifm,isolver)
    rw[:,:] = bc(nx,ny,rw)
    
    #stage-1
    tw[2:nx+3,2:ny+3] = ww[2:nx+3,2:ny+3] + dt*rw[2:nx+3,2:ny+3]
    tw[:,:] = bc(nx,ny,tw[:,:])
    
    sw[:,:] = fst(nx, ny, dx, dy, -tw[:,:])
    sw[:,:] = bc(nx,ny,sw[:,:])
    
    rw[:,:] = rhs(nx,ny,dx,dy,re,ro,st,tw[:,:],sw[:,:],fs,ifm,isolver)
    rw[:,:] = bc(nx,ny,rw[:,:])
    
    #stage-2
    tw[2:nx+3,2:ny+3] = 0.75*ww[2:nx+3,2:ny+3] + 0.25*tw[2:nx+3,2:ny+3] + 0.25*dt*rw[2:nx+3,2:ny+3]
    tw[:,:] = bc(nx,ny,tw[:,:])
    
    sw[:,:] = fst(nx, ny, dx, dy, -tw[:,:])
    sw[:,:] = bc(nx,ny,sw[:,:])
    
    rw[:,:] = rhs(nx,ny,dx,dy,re,ro,st,tw[:,:],sw[:,:],fs,ifm,isolver)
    rw[:,:] = bc(nx,ny,rw[:,:])
    
    #stage-3
    ww[2:nx+3,2:ny+3] = aa*ww[2:nx+3,2:ny+3] + bb*tw[2:nx+3,2:ny+3] + bb*dt*rw[2:nx+3,2:ny+3]
    ww[:,:] = bc(nx,ny,ww[:,:])
    
    sw[:,:] = fst(nx, ny, dx, dy, -ww[:,:])
    sw[:,:] = bc(nx,ny,sw[:,:])
            
    sx, sy = grad_spectral(nx,ny,sw[2:nx+3,2:ny+3])
    
    umax = np.max(np.abs(sy))
    vmax = np.max(np.abs(sx))
    
    cfl = np.max([umax*dt/dx, vmax*dt/dy])
    
    if (k%freq == 0):
        w_wrong[:,:,int(k/freq)] = ww[:,:]
        print(int(k/freq), " ", time, " ", np.max(ww), " ", cfl)
        export_data_wrong(nx,ny,re,ro,int(k/freq),ww,sw,isolver,ifm)
        
#%%
fig, axs = plt.subplots(1,2,figsize=(11,5))
    
cs = axs[0].contourf(x,y,w_wrong[2:nx+3,2:ny+3,0],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[0], orientation='vertical')
axs[0].set_title('Wrong ($t$ = 0)')

cs = axs[1].contourf(x,y,w_wrong[2:nx+3,2:ny+3,-1],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[1], orientation='vertical')
axs[1].set_title('Wrong ($t$ = ' + str(nt*dt)+')')

plt.show()
        
#%%
wf = np.zeros(ne)
Af = np.zeros((ne,npe))   # Af data
w_ = np.zeros((nx+5,ny+5))

w = np.reshape(wa[:,k], [nx+5,ny+5])
s = fst(nx, ny, dx, dy, -w)
s = bc(nx,ny,s)
export_data(nx,ny,re,ro,0,w,s,isolver,ifm)

#%%
k = 0
time = 0.0   
data_probe = np.zeros((nt+1,4))
probe_list = [3,2+int(nx/2),nx+1]

data_probe[k,0] = time
data_probe[k,1:4] = np.reshape(wa[:,k],[nx+5,ny+5])[probe_list,int(ny/2)+2]
#data_probe[k,4:] = sen[probe_list,int(ny/2)+2,0]

#%%
def rhs(nx,ny,dx,dy,re,ro,st,w,s,fs,ifm,isolver):
    if isolver == 1:
        return rhs_arakawa(nx,ny,dx,dy,re,ro,st,w,s,fs,ifm)


# time integration using third-order Runge Kutta method
aa = 1.0/3.0
bb = 2.0/3.0
clock_time_init = tm.time()

for k in range(1,nt+1):
    time = time + dt
    for n in range(npe):
        ren[:,:,n] = rhs(nx,ny,dx,dy,re,ro,st,wen[:,:,n],sen[:,:,n],fs,ifm,isolver)
        ren[:,:,n] = bc(nx,ny,ren[:,:,n])
        
        #stage-1
        ten[2:nx+3,2:ny+3,n] = wen[2:nx+3,2:ny+3,n] + dt*ren[2:nx+3,2:ny+3,n]
        ten[:,:,n] = bc(nx,ny,ten[:,:,n])
        
        sen[:,:,n] = fst(nx, ny, dx, dy, -ten[:,:,n])
        sen[:,:,n] = bc(nx,ny,sen[:,:,n])
        
        ren[:,:,n] = rhs(nx,ny,dx,dy,re,ro,st,ten[:,:,n],sen[:,:,n],fs,ifm,isolver)
        ren[:,:,n] = bc(nx,ny,ren[:,:,n])
        
        #stage-2
        ten[2:nx+3,2:ny+3,n] = 0.75*wen[2:nx+3,2:ny+3,n] + 0.25*ten[2:nx+3,2:ny+3,n] + 0.25*dt*ren[2:nx+3,2:ny+3,n]
        ten[:,:,n] = bc(nx,ny,ten[:,:,n])
        
        sen[:,:,n] = fst(nx, ny, dx, dy, -ten[:,:,n])
        sen[:,:,n] = bc(nx,ny,sen[:,:,n])
        
        ren[:,:,n] = rhs(nx,ny,dx,dy,re,ro,st,ten[:,:,n],sen[:,:,n],fs,ifm,isolver)
        ren[:,:,n] = bc(nx,ny,ren[:,:,n])
        
        #stage-3
        wen[2:nx+3,2:ny+3,n] = aa*wen[2:nx+3,2:ny+3,n] + bb*ten[2:nx+3,2:ny+3,n] + bb*dt*ren[2:nx+3,2:ny+3,n]
        wen[:,:,n] = bc(nx,ny,wen[:,:,n])
        
        sen[:,:,n] = fst(nx, ny, dx, dy, -wen[:,:,n])
        sen[:,:,n] = bc(nx,ny,sen[:,:,n])
        
        we[:,n] = np.reshape(wen[:,:,n],[(nx+5)*(ny+5)])  
        wet[:,n,0] = np.reshape(wen[3:nx+2,3:ny+2,n],[(nx-1)*(ny-1)])
    
    wa[:,k] = np.sum(we[:,:],axis=1)
    wa[:,k] = wa[:,k]/npe
    
    wat[:,k] = np.sum(wet[:,:,0],axis=1)
    wat[:,k] = wat[:,k]/npe
    
    sx, sy = grad_spectral(nx,ny,sen[2:nx+3,2:ny+3,0])
    
    umax = np.max(np.abs(sy))
    vmax = np.max(np.abs(sx))
    
    cfl = np.max([umax*dt/dx, vmax*dt/dy])
    
    data_probe[k,0] = time
    data_probe[k,1:4] = np.reshape(wa[:,k],[nx+5,ny+5])[probe_list,int(ny/2)+2]
#    data_probe[k,4:] = sen[probe_list,int(ny/2)+2,0]
    
    #print(k, " ", time, " ", np.max(wat[:,k]), " ", cfl)
    
    if k == oib[kobs]:
        print(k, np.max(wat[:,k]))
        # compute mean of the forecast fields
        wf[:] = np.sum(wet[:,:,0],axis=1)   
        wf[:] = wf[:]/npe
        
        # compute Af dat
        for n in range(npe):
            Af[:,n] = wet[:,n,0] - wf[:]
            
        da = dh @ Af
        
        cc = da @ da.T/(npe-1)  
        
        for i in range(me):
            cc[i,i] = cc[i,i] + sd2 
        
        ci = np.linalg.pinv(cc)
        
        km = Af @ da.T @ ci/(npe-1)
        
        # analysis update    
        kmd = km @ (z[:,kobs] - wf[oin])
        wat[:,k] = wf[:] + kmd[:]
        
        w_[3:nx+2,3:ny+2] = np.reshape(wat[:,k],[nx-1,ny-1])
        w_ = bc(nx,ny,w_)
        
        wa[:,k] = np.reshape(w_,[(nx+5)*(ny+5)])
        
        # ensemble correction
#        ha = dh @ Af
        
        wet[:,:,0] = Af[:,:] - 0.5*(km @ dh @ Af) + wat[:,k].reshape(-1,1)
        
        for n in range(npe):
            w_[3:nx+2,3:ny+2] = np.reshape(wet[:,n,0],[nx-1,ny-1])
            w_ = bc(nx,ny,w_)
            wen[:,:,n] = w_
            
        #multiplicative inflation (optional): set lambda=1.0 for no inflation
        #ue[:,:,k] = ua[:,k] + lambd*(ue[:,:,k] - ua[:,k])
        
        kobs = kobs + 1
        
    if (k%freq == 0):
#        #ws_all[int(k/freq),:,:,0] = w  
#        #ws_all[int(k/freq),:,:,1] = s
#        wa[:,int(k/freq)] = np.average(we[:,:],axis=1)
#        w = np.reshape(wa[:,int(k/freq)], [nx+5,ny+5])
#        s = fst(nx, ny, dx, dy, -w)
#        s = bc(nx,ny,s)
#        print(k, " ", time, " ", np.max(w), " ", cfl)
#        
        export_data(nx,ny,re,ro,int(k/freq),w,s,isolver,ifm)


total_clock_time = tm.time() - clock_time_init
print('Total clock time=', total_clock_time)
np.save('cpu_time.npy',total_clock_time)

#%%
snap_id = ist_sn 
data = np.load('./results/'+filename_sn+'/results_'+str(snap_id)+'.npz')
w_true_init = data['w']

snap_id = ist_sn + ns
data = np.load('./results/'+filename_sn+'/results_'+str(snap_id)+'.npz')
w_true_forecast = data['w']

w_da_init = np.reshape(wa[:,0],[nx+5,ny+5])
w_da_forecast = np.reshape(wa[:,-1],[nx+5,ny+5])

fig, ax = plt.subplots(2,3,figsize=(12,8))

axs = ax.flat     

cs = axs[0].contourf(x,y,w_true_init[2:nx+3,2:ny+3],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[0], orientation='vertical')
axs[0].set_title('True Initial')

cs = axs[1].contourf(x,y,w_wrong[2:nx+3,2:ny+3,0],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[1], orientation='vertical')
axs[1].set_title('Wrong Initial')

cs = axs[2].contourf(x,y,w_da_init[2:nx+3,2:ny+3],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[2], orientation='vertical')
axs[2].set_title('DA Initial')

cs = axs[3].contourf(x,y,w_true_forecast[2:nx+3,2:ny+3],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[3], orientation='vertical')
axs[3].set_title('True Forecast')

cs = axs[4].contourf(x,y,w_wrong[2:nx+3,2:ny+3,-1],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[4], orientation='vertical')
axs[4].set_title('Wrong Forecast')

cs = axs[5].contourf(x,y,w_da_forecast[2:nx+3,2:ny+3],60,cmap='jet',vmin=-500,vmax=350)
fig.colorbar(cs, ax=axs[5], orientation='vertical')
axs[5].set_title('DA Forecast')

plt.show()
fig.tight_layout()
fig.savefig('contour_da.png', dpi=200)

#%%
data_probe_true = np.zeros((ns+1,4))
data_probe_wrong = np.zeros((ns+1,4))
data_probe_da = np.zeros((ns+1,4))

for k in range(ns+1):
    snap_id = ist_sn + k
    data = np.load('./results/'+filename_sn+'/results_'+str(snap_id)+'.npz')
    w = data['w']
    data_probe_true[k,0] = k*dt*nf
    data_probe_true[k,1:] = w[[probe_list,int(ny/2)+2]]

for k in range(ns+1):
    w = w_wrong[:,:,k]
    data_probe_wrong[k,0] = k*dt*nf
    data_probe_wrong[k,1:] = w[[probe_list,int(ny/2)+2]]
    
fig, axs = plt.subplots(3,1,figsize=(8,8))
    
axs[0].plot(data_probe_true[:,0], data_probe_true[:,1], 'k', label = 'True')
axs[1].plot(data_probe_true[:,0], data_probe_true[:,2], 'k', label = 'True')
axs[2].plot(data_probe_true[:,0], data_probe_true[:,3], 'k', label = 'True')

axs[0].plot(data_probe_true[:,0], data_probe_wrong[:,1], label = 'Wrong')
axs[1].plot(data_probe_true[:,0], data_probe_wrong[:,2], label = 'Wrong')
axs[2].plot(data_probe_true[:,0], data_probe_wrong[:,3], label = 'Wrong')

axs[0].plot(data_probe[:,0], data_probe[:,1], 'r--',  label = 'DA')
axs[1].plot(data_probe[:,0], data_probe[:,2], 'r--', label = 'DA')
axs[2].plot(data_probe[:,0], data_probe[:,3], 'r--', label = 'DA')

axs[0].legend()
axs[2].legend()
axs[1].legend()

fig.tight_layout()
plt.show()    

fig.savefig('time_series_da.png', dpi=200)

#%%
data_probe_true = np.zeros((ns+1,4))

qq1 = 1
qq2 = 10

for k in range(ns+1):
    snap_id = ist_sn + k
    data = np.load('./results/'+filename_sn+'/results_'+str(snap_id)+'.npz')
    w = data['w']
    wr = np.reshape(w[3:nx+2,3:ny+2], [ne])
    data_probe_true[k,0] = k*dt*nf
    data_probe_true[k,1] = wr[oin[qq1]]
    data_probe_true[k,2] = wr[oin[qq2]]

fig, axs = plt.subplots(2,1,figsize=(8,6))
    
axs[0].plot(data_probe_true[:,0], data_probe_true[:,1], 'k', label = 'True')
axs[0].plot(data_probe_true[:,0], z[qq1,:], 'ro', label = 'Observations')


axs[1].plot(data_probe_true[:,0], data_probe_true[:,2], 'k', label = 'True')
axs[1].plot(data_probe_true[:,0], z[qq2,:], 'ro', label = 'Observations')

axs[0].legend()
axs[1].legend()

fig.tight_layout()
plt.show()    

fig.savefig('time_series_da_v2.png', dpi=200)

#%%
if ipr == 5:
    avg_start = 20
    wavg = np.average(wa[:,avg_start:], axis=1)
    wavg = np.reshape(wavg, [nx+5,ny+5])
    savg = fst(nx, ny, dx, dy, -wavg)
    savg = bc(nx,ny,savg)
    
    fig, axs = plt.subplots(1,2,figsize=(10,5))

    cs = axs[0].contour(x,y,wavg[2:nx+3,2:ny+3],60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[0], orientation='vertical')
    
    cs = axs[1].contour(x,y,savg[2:nx+3,2:ny+3],60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[1], orientation='vertical')
    
    plt.show()
    fig.tight_layout()
    fig.savefig('qg_'+ str(re) +'_' + str(ro) + '_'+str(npe) + '_' + str(nx) + '_' + str(nt) + '.png', 
                bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    
    fig, axs = plt.subplots(1,1,figsize=(8,4))
    
    axs.plot(data_probe[int(nt/2):,0], data_probe[int(nt/2):,1], label = 'Left')
    axs.plot(data_probe[int(nt/2):,0], data_probe[int(nt/2):,2], label = 'Center')
    axs.plot(data_probe[int(nt/2):,0], data_probe[int(nt/2):,3], label = 'Right')
    
    axs.legend()
    fig.tight_layout()
    plt.show()
    fig.savefig('qgs_trajectory_'+ str(re) +'_' + str(ro) + '_'+str(npe) + '_' + str(nx) + '_' + str(nt) + '.png', 
                bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    
#%%
if (ipr == 4):

    fig, axs = plt.subplots(1,2,figsize=(12,5))
    
    for ne in range(npe):
        w0 = np.reshape(we[:,ne], [nx+5,ny+5])
        wh = np.reshape(we[:,ne], [nx+5,ny+5])
        wn = np.reshape(we[:,ne], [nx+5,ny+5])
        en0, n = energy_spectrum(nx,ny,w0)    
        enh, n = energy_spectrum(nx,ny,wh) 
        ent, n = energy_spectrum(nx,ny,wn)
        k = np.linspace(1,n,n)
            
        #axs[0].loglog(k,en0[1:],'r', ls = '-', lw = 2, label='$t = 0.0$')
        
        #axs[0].loglog(k,enh[1:],'g', ls = '-', lw = 2, alpha = 0.2,label='$t = '+str(0.5*dt*nt)+'$')
        #axs[0].loglog(k,ent[1:], 'b', lw = 2, alpha = 0.5, label = '$t = '+str(dt*nt)+'$')
        
        axs[1].loglog(k,ent[1:], 'b', lw = 2, alpha = 0.2, label = '$t = '+str(dt*nt)+'$')
    
    w0 = np.reshape(wa[:,0], [nx+5,ny+5])
    wh = np.reshape(wa[:,int(ns/2)], [nx+5,ny+5])
    wn = np.reshape(wa[:,-1], [nx+5,ny+5])
    en0, n = energy_spectrum(nx,ny,w0)    
    enh, n = energy_spectrum(nx,ny,wh) 
    ent, n = energy_spectrum(nx,ny,wn)
    k = np.linspace(1,n,n)
    
    axs[0].loglog(k,enh[1:],'r', ls = '-', lw = 2, alpha = 1.0, label='$t = '+str(0.5*dt*nt)+'$')
    axs[0].loglog(k,ent[1:], 'k', lw = 2, alpha = 1.0, label = '$t = '+str(dt*nt)+'$')
    #axs[0].loglog(k,ent[1:], 'b', lw = 2, alpha = 0.5, label = '$t = '+str(dt*nt)+'$')
    
    axs[1].loglog(k,ent[1:], 'k', lw = 2, alpha = 1.0, label = '$t = '+str(dt*nt)+'$')
    
    line = 100*k**(-3.0)
    axs[0].loglog(k,line, 'k--', lw = 2, )
    axs[1].loglog(k,line, 'k--', lw = 2, )
    
    axs[0].text(0.75, 0.75, '$k^{-3}$', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')
    axs[1].text(0.75, 0.75, '$k^{-3}$', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')
    
    
    axs[0].set_xlabel('$K$')
    axs[0].set_ylabel('$E(K)$')
    axs[0].legend(loc=3)
    axs[0].set_ylim(1e-10,1e0)
    #axs[0].set_title('$t$ = ' + str(0.5*nt*dt))
    
    axs[1].set_xlabel('$K$')
    axs[1].set_ylabel('$E(K)$')
    #axs[0].legend(loc=3)
    axs[1].set_ylim(1e-10,1e0)
    axs[1].set_title('$t$ = ' + str(1.0*nt*dt))
    
    plt.show()
    fig.savefig('e_dns'+ str(re) + '_'+str(npe) + '_' + str(nx) + '_' + str(ifm) + '.png', 
                bbox_inches = 'tight', pad_inches = 0, dpi = 300)

#%%    
w_true = np.zeros((nx+5,ny+5,ns+1))    
for k in range(ns+1):
    snap_id = ist_sn + k
    data = np.load('./results/'+filename_sn+'/results_'+str(snap_id)+'.npz')
    w_true[:,:,k] = data['w']

w_da = np.zeros((nx+5,ny+5,ns+1))    
for k in range(nt+1):
    if (k%freq == 0):
        print(k, int(k/nf))
        w_da[:,:,int(k/nf)] = np.reshape(wa[:,k],[nx+5,ny+5])
        
#%%    
np.savez('qg_'+ str(re) + '_' + str(ro) + '_' +str(npe)+ '_' + str(nx) + '_' + str(ny) + '.npz', 
         x = x,
         y = y,
         w_true = w_true,
         w_wrong = w_wrong,
         w_da = w_da)