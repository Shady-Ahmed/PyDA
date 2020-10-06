# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:09:08 2020

@author: Shady
"""


import numpy as np

def euler(rhs,state,dt,*args):
    
    k1 = rhs(state,*args)   
    new_state = state + dt*k1
    return new_state


def RK4(rhs,state,dt,*args):
    
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    k4 = rhs(state+k3*dt,*args)

    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state