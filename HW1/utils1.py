import numpy as np
import numpy.linalg as la
from math import factorial as fact

def A(_):
    return min(1, _)

def Poisson(k):
    lamb = 0.5
    return np.exp(-lamb) * lamb ** k / fact(k)


