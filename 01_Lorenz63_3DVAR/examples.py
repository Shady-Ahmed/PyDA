# -*- coding: utf-8 -*-
"""
examples file for a tutorial paper on data assimilation
"PyDA: A hands-on introduction to dynamical data assimilation with Python"
@authors: Shady E. Ahmed, Suraj Pawar, Omer San
"""

import numpy as np

def Lorenz63(state,*args): #Lorenz 96 model
    #rho = 28.0     #sigma = 10.0     #beta = 8.0 / 3.0
    
    sigma = args[0]
    beta = args[1]
    rho = args[2]
    x, y, z = state #Unpack the state vector
    f = np.zeros(3) #Derivatives
    f[0] = sigma * (y - x)
    f[1] = x * (rho - z) - y
    f[2] = x * y - beta * z
    return f 

def JLorenz63(state,*args): #Jacobian of Lorenz 63 model
    #rho = 28.0     #sigma = 10.0     #beta = 8.0 / 3.0
    
    sigma = args[0]
    beta = args[1]
    rho = args[2]
    x, y, z = state #Unpack the state vector
    df = np.zeros([3,3]) #Derivatives
   
    df[0,0] = sigma * (-1)
    df[0,1] = sigma * (1)
    df[0,2] = sigma * (0)

    df[1,0] = 1 * (rho - z) 
    df[1,1] = -1
    df[1,2] = x * (-1)

    df[2,0] = 1 * y 
    df[2,1] = x * 1 
    df[2,2] = - beta

    return df 



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


def JLorenz96(state,*args): #Jacobian of Lorenz 96 model
    x = state
    F = args[0]
    n = len(x)    
    df = np.zeros([n,n])
    # bounday points: i=0,1,N-1
    df[0,0] = -1
    df[0,1] = x[n-1]
    df[0,n-1] = (x[1] - x[n-2])
    df[0,n-2] = -x[n-1]

    df[1,0] = (x[2] - x[n-1])
    df[1,1] = -1
    df[1,2] = x[0]
    df[1,n-1] = -x[0]

    df[n-1,0] =  x[n-2]
    df[n-1,n-1] = -1
    df[n-1,n-2] = (x[0] - x[n-3])
    df[n-1,n-3] = -x[n-2]
    
    for i in range(2, n-1):
        df[i,i] = -1
        df[i,i+1] = x[i-1]
        df[i,i-1] = x[i+1] - x[i-2]
        df[i,i-2] = -x[i-1]

    return df



def Burgers1D(state,*args):
    
    u = state
    nu = args[0]
    dx = args[1]
    nx = len(u) - 1
    f = np.zeros(nx+1)
    f[1:nx] = (nu/(dx*dx))*(u[2:nx+1] - 2*u[1:nx] + u[0:nx-1]) \
             - (1.0/3.0)*(u[2:nx+1]+u[0:nx-1]+u[1:nx])*(u[2:nx+1]-u[0:nx-1])/(2.0*dx) 
             
    return f

def JBurgers1D(state,*args):
    
    u = state
    nu = args[0]
    dx = args[1]
    nx = len(u) - 1
    df = np.zeros([nx+1,nx+1])
         
    for i in range(1,nx):
        df[i,i] = (nu/(dx*dx))*(-2) \
                 - (1.0/3.0)*(u[i+1]-u[i-1])/(2.0*dx) 
        
        df[i,i+1] = (nu/(dx*dx))*(1) \
                  - (1.0/3.0)*(u[i+1]+u[i-1]+u[i])*(1)/(2.0*dx) \
                  - (1.0/3.0)*(1)*(u[i+1]-u[i-1])/(2.0*dx) 
                               
        df[i,i-1] = (nu/(dx*dx))*(1) \
                  - (1.0/3.0)*(u[i+1]+u[i-1]+u[i])*(-1)/(2.0*dx) \
                  - (1.0/3.0)*(1)*(u[i+1]-u[i-1])/(2.0*dx) 

    return df