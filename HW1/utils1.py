import numpy as np
import numpy.linalg as la
from math import factorial, ceil
from scipy.special import gamma
import random

def A(_):
    return min(1, _)

def d(k):
    pass

def b(k):
    pass

def M(s, j, y):
    return Disasters(s[j], s[j+1], y)

def Disasters(s1, s2, y):
    start = ceil(s1 - 1851)
    end = ceil(s2 - 1851)
    return sum(y[start : end])

def Poisson(k):
    lamb = 0.5
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
        tmp = min((h[j] - h_[j]) * (s[j+1] - s[j] + beta) + (Disasters(s[j], s[j+1]) + alpha, y) * (np.log(h_[j]) - np.log(h[j])), 0)
        a = np.exp(tmp)
        u = np.random.rand()
        if u <= a:
            h = h_
        else:
            pass

def SMove(h, s, k, y):
    for j in range(k):
        if j == 0:
            continue
        s_ = s
        s_[j] = random.uniform(s[j-1], s[j+1])
        u = np.random.rand()
        tmp = min(0,
            (h[j]-h[j-1]) * (s_[j]-s[j]) + (Disasters(s_[j-1], s_[j], y) - Disasters(s[j-1], s[j], y)) * np.log(h[j-1])
            + (Disasters(s_[j], s_[j+1], y) - Disasters(s[j], s[j+1], y)) * np.log(h[j]) + np.log(s_[j] - s[j-1]) + np.log(s[j+1]-s_[j])
            - np.log(s[j]-s[j-1]) - np.log(s[j+1]-s[j]))
        a = np.exp(tmp)
        if u <= a:
            s = s_
        else:
            pass

def Birth(h, s, k, y):
    alpha = 1
    beta = 200 / 365.24
    lamb = 0.5
    j = random.randint(0, k - 1)
    s_star = random.uniform(s[j], s[j + 1])
    s_ = np.insert(s, j + 1, s_star)
    u = np.random.rand()
    h_j = h[j] * (u / (1-u)) ** ((s[j + 1] - s_star) / (s[j + 1] - s[j]))
    h_j1 = h[j] * ((1-u) / u) ** ((s_star - s[j]) / (s[j + 1] - s[j]))
    h_ = h
    h_[j] = h_j
    h_ = np.insert(h_, j + 1, h_j1)
    tmp1 = np.exp(-h_[j] * (s_star - s[j]) - h_[j + 1] * (s[j + 1] - s_star) + M(s_, j, y) * np.log(h_[j]) + M(s_, j + 1, y) * np.log(h_[j + 1])
            + h[j] * (s[j + 1] - s[j]) - M(s, j, y) * np.log(h[j]))
    tmp2 = (2 * lamb * (2 * k + 1) / (s[-1] - s[0]) * (s_star - s[j]) * (s[j + 1] - s_star) / (s[j + 1] - s[j])
            * beta ** alpha / gamma(alpha) * ((h_[j] * h_[j + 1]) / h[j]) ** (alpha - 1) * np.exp(-beta * (h_[j] + h_[j+1] -h[j])))
    tmp3 = d(k) * (s[-1] - s[0]) / (b(k - 1) * k)
    tmp4 = (h_[j] + h_[j + 1]) ** 2 / h[j]
    a = min(tmp1 * tmp2 * tmp3 * tmp4, 1)
    u = np.random.rand()
    if u <= a:
        h = h_
        s = s_
        k = k + 1
    else:
        pass

def Death(h, s, k, y):
    pass