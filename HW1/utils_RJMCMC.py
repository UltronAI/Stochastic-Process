import numpy as np
import numpy.linalg as la
from math import factorial, ceil, floor
from scipy.special import gamma
import random

alpha = 1
beta = 200 / 365.24
lamb = 3

def A(_):
    return min(1, _)

def b(k): # , k_max):
    return 3.6 / 7 * min(1, lamb / k)
    """
    if k == k_max:
        return 0
    else:
        return 3.6 / 7 * min(1, lamb / k)
    """

def d(k):
    return 3.6 / 7 * min(1, (k - 1) / lamb)
    """
    if k == 1:
        return 0
    else:
        return 3.6 / 7 * min(1, k / lamb)
    """

def pi(k):
    if k == 1:
        return 0
    else:
        return 0.5 * (1 - b(k) - d(k))

def yita(k):
    return 1 - b(k, k_max) - d(k) - pi(k)

def M(s, j, y):
    return Disasters(s[j], s[j + 1], y)

def Disasters(s1, s2, y):
    start = ceil(s1 - 1851)
    end = ceil(s2 - 1851)
    return sum(y[start : end])

def Poisson(k):
    lamb = 3
    return np.exp(-lamb) * lamb ** (k - 1) / factorial(k - 1)

def Gamma(h):
    alpha = 1
    beta = 200 / 365.24
    return beta ** alpha * h ** (alpha - 1) * np.exp(-beta * h) / gamma(alpha)

def HMove(h, s, k, y):
    alpha = 1
    beta = 200 / 365.24 
    for j in range(k):
        un = random.uniform(-0.5, 0.5)
        h_ = h
        h_[j] = h[j] * np.exp(un)
        tmp = min((h[j] - h_[j]) * (s[j + 1] - s[j] + beta) + (Disasters(s[j], s[j + 1], y) + alpha) * (np.log(h_[j]) - np.log(h[j])), 0)
        a = np.exp(tmp)
        u = np.random.rand()
        if u <= a:
            h = h_
        else:
            pass
    return h, s, k

def SMove(h, s, k, y):
    for j in range(k):
        if j == 0:
            continue
        s_ = s
        s_[j] = random.uniform(s[j-1], s[j+1])
        u = np.random.rand()
        tmp = min(0,
            (h[j]-h[j - 1]) * (s_[j]-s[j]) + (Disasters(s_[j - 1], s_[j], y) - Disasters(s[j - 1], s[j], y)) * np.log(h[j - 1])
            + (Disasters(s_[j], s_[j + 1], y) - Disasters(s[j], s[j + 1], y)) * np.log(h[j]) + np.log(s_[j] - s[j - 1]) + np.log(s[j + 1]-s_[j])
            - np.log(s[j]-s[j - 1]) - np.log(s[j + 1]-s[j]))
        a = np.exp(tmp)
        if u <= a:
            s = s_
        else:
            pass
    return h, s, k

def Birth(h, s, k, y):
    j = random.randint(0, k - 1)
    s_star = random.uniform(s[j], s[j + 1])
    s_ = np.insert(s, j + 1, s_star)
    u = np.random.rand()
    h_j = h[j] * (u / (1 - u)) ** ((s[j + 1] - s_star) / (s[j + 1] - s[j]))
    h_j1 = h[j] * ((1 - u) / u) ** ((s_star - s[j]) / (s[j + 1] - s[j]))
    h_ = np.insert(h, j + 1, h_j1)
    h_[j] = h_j
    tmp1 = np.exp(-h_[j] * (s_star - s[j]) - h_[j + 1] * (s[j + 1] - s_star) + M(s_, j, y) * np.log(h_[j]) + M(s_, j + 1, y) * np.log(h_[j + 1]) + h[j] * (s[j + 1] - s[j]) - M(s, j, y) * np.log(h[j]))
    tmp2 = 2 * lamb * (2 * k + 1) / (s[-1] - s[0]) ** 2 * (s_star - s[j]) * (s[j + 1] - s_star) / (s[j + 1] - s[j]) * beta ** alpha / gamma(alpha) * (h_[j] * h_[j + 1] / h[j]) ** (alpha - 1)
    tmp3 = np.exp(-beta * (h_[j] + h_[j + 1] - h[j]))
    tmp4 = d(k + 1) * (s[-1] - s[0]) / (b(k) * k) * (h_[j] + h_[j + 1]) ** 2 / h[j]
    a = min(np.log(tmp1 * tmp2 * tmp3 * tmp4), 0)
    """
    print(tmp1)
    print(tmp2)
    print(tmp3)
    print(tmp4)
    print(a)
    """
    u = np.random.rand()
    if u <= np.exp(a):
        h = h_
        s = s_
        k = k + 1
    else:
        pass
    return h, s, k

def Death(h, s, k, y):
    alpha = 1
    beta = 200 / 365.24
    lamb = 3
    j = random.randint(0, k - 2)
    h_j = h[j] ** ((s[j + 1] - s[j]) / (s[j + 2] - s[j])) * h[j + 1] ** ((s[j + 2] - s[j + 1]) / (s[j + 2] - s[j]))
    s_ = np.delete(s, j + 1)
    h_ = np.delete(h, j + 1)
    h_[j] = h_j
    tmp1 = np.exp(-h_[j] * (s[j + 2] - s[j]) + M(s_, j, y) * np.log(h_[j]) + h[j] * (s[j + 1] - s[j]) + h[j + 1] * (s[j + 2] - s[j + 1])
            - M(s, j, y) * np.log(h[j]) - M(s, j + 1, y) * np.log(h[j + 1]))
    tmp2 = ((s[-1] - s[0]) ** 2 / (2 * lamb * (2 * k + 1)) * (s[j + 2] - s[j]) / ((s[j + 1] - s[j]) * (s[j + 2] - s[j + 1])) 
            * gamma(alpha) / beta ** alpha * (h_[j] / (h[j] * h[j + 1])) ** (alpha - 1) * np.exp(beta * (h[j] + h[j + 1] - h_[j])))
    tmp3 = b(k) * k / (d(k+1) * (s[-1] - s[0]))
    tmp4 = h_[j] / (h[j] + h[j + 1]) ** 2
    a = min(np.log(tmp1 * tmp2 * tmp3 * tmp4), 0)
    u = np.random.rand()
    if u <= np.exp(a):
        h = h_
        s = s_
        k = k - 1
    else:
        pass
    return h, s, k

def Loss(h, s, y):
    k = h.shape[0]
    err = 0
    for i in range(k):
        start = ceil(s[i] - s[0])
        end = ceil(s[i + 1] - s[0])
        y_ = y[start : end]
        err += sum((y_ - h[i]) ** 2)
    
    return err / y.shape[0]