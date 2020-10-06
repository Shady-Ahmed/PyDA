# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:50:06 2020

@author: Shady
"""

import numpy as np

def Lorenz63(state,*args): # Lorenz 96 model
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0
    x, y, z = state # Unpack the state vector
    f = np.zeros(3) #Derivatives
    f[0] = sigma * (y - x)
    f[1] = x * (rho - z) - y
    f[2] = x * y - beta * z
    print(*args)
    return f 


def Lorenz96(state,*args): # Lorenz 96 model
    x = state
    F = args[0]
    n = len(x)    
    f = np.zeros(n)
    # bounday points: i=0,1,N-1
    f[0] = (x[1] - x[n-2]) * x[n-1] - x[0]
    f[1] = (x[2] - x[n-1]) * x[0] - x[1]
    f[n-1] = (x[0] - x[n-3]) * x[n-2] - x[n-1]
    for i in range(2, n-1):
        f[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
    # Add the forcing term
    f = f + F

    return f


def Burgers1D(state,*args):
    
    u = state
    nu = args[0]
    dx = args[1]
    nx = len(u) - 1
    f = np.zeros(nx+1)
    f[1:nx] = (nu/(dx*dx))*(u[2:nx+1] - 2*u[1:nx] + u[0:nx-1]) \
             - (1.0/3.0)*(u[2:nx+1]+u[0:nx-1]+u[1:nx])*(u[2:nx+1]-u[0:nx-1])/(2.0*dx) 
             
    return f


u=np.zeros(3)

ff = Lorenz63(u)