import numpy as np
import numpy.linalg as la
from math import factorial
from scipy.special import gamma
import random

def A(_):
    return min(1, _)

def Disasters(s1, s2, y):
    start = s1 - 1851
    end = s2 - 1851
    return sum(y[start : end])

def Poisson(k):
    lamb = 0.5
    return np.exp(-lamb) * lamb ** (k - 1) / factorial(k - 1)

def Gamma(h):
    alpha = 1
    beta = 200 / 365.24
    return beta ** alpha * h ** (alpha - 1) * np.exp(-beta * h) / gamma(alpha)

def P(h, s, y):
    L = h.shape[0]
    out = 0
    for i in range(L):
        out += -h[i] * (s[i + 1] - s[i])
        out += np.log(h[i]) * Disasters(s[i], s[i + 1], y)
    out = np.exp(out)


def HMove(h, s, y):
    iter = 10
    L = h.shape[0]
    for j in range(L):
        u = random.uniform(-0.5, 0.5)
        h_ = h
        h_[j] = h[j] * np.exp(u)
        u_ = np.random.rand()
        if u_ <= A(P(h_, s, y) * Gamma(h_[j]) * h[j] / (P(h, s, y) * Gamma(h[j]) * h_[j])):
            h[j] = h_[j]
        else:
            pass

def SMove(h, s, y):
    iter = 10
    L = s.shape[0]
    for j in range(L):
        if j == 0 or j == L - 1:
            continue 
        s_ = s
        s_[j] = random.randint(s[j - 1], s[j + 1])
        u = np.random.rand()
        if u <= A(P(h, s_, y) * (1 / (s[j + 1] - s[j - 1])) \
                / (P(h, s, y) * (1 / (s[j + 1] - s[j - 1])) * (s[j] - s[j - 1]) * (s[j + 1] - s[j]))):
            s[j] = s_[j]
        else:
            pass

