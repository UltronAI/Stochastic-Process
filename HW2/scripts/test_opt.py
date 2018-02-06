# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:09:02 2018

@author: gaof
"""

import numpy as np
from scipy.optimize import minimize

def f(p):
    x,y = p
    z = (x-5)**3 + y**2
    return z

def fprime(p):
    x,y = p
    dx = 3*(x-5)**2
    dy = 2*y
    return np.array([dx, dy])

init = [-5, 5]

result = minimize(f, init, method='Newton-CG', jac=fprime)

print(result.x)