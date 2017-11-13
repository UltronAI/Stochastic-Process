import numpy as np
import numpy.linalg as la
import random

def A(_):
    return min(1, _)

def FindClosest(Mu, j1):
    mu1 = Mu[j1, :]
    d_min = 999999
    j2 = j1
    for i in range(Mu.shape[0]):
        if i == j1:
            continue
        mu2 = Mu[i, :]
        d = la.norm(mu1 - mu2)
        if d < d_min:
            d_min = d
            j2 = i
    return j2

def Generate(X):
    d = X.shape[1]
    mean = X.mean(axis = 0)
    var = np.cov(X, rowvar=False)
    delta = np.random.rand(1, d)
    u = np.random.rand()
    if u > 0.5:
        return np.random.multivariate_normal((mean + delta).reshape(d), var).reshape(1, d)
    else:
        return np.random.multivariate_normal((mean - delta).reshape(d), var).reshape(1, d)

def C(c, method = "AIC", N = np.e**2):
    if method == "AIC":
        return c + 1
    else:
        return (c + 1) * np.log(N) / 2

def Linear(s):
    return s

def Cubic(s):
    return s**3

def ThinPlateSpline(s):
    return s**2 * np.log(s)

def Multiquadric(s):
    lamb = 0.5
    return (s**2 + lamb) ** 0.5

def Gauss(s):
    lamb = 0.5
    return np.exp(-lamb * s**2)

def Phi(X, Mu, phi = "Gauss"):
    N = X.shape[0]
    k = Mu.shape[0]
    out = np.empty((N, k))
    for i in range(N):
        s = la.norm(X[i, :] - Mu, axis = 1)
        if phi == "Liearn":
            out[i, :] = Linear(s)
        elif phi == "Cubic":
            out[i, :] = Cubic(s)
        elif phi == "ThinPlateSpline":
            out[i, :] = ThinPlateSpline(s)
        elif phi == "Multiquadric":
            out[i, :] = Multiquadric(s)
        else:
            out[i, :] = Gauss(s)
    return out

def D(X, Mu, phi = "Gauss"):
    D1 = np.ones((X.shape[0], 1))
    D2 = X
    if Mu.shape[0] > 0:
        D3 = Phi(X, Mu, phi)
        return np.concatenate((D1, D2, D3), axis = 1)
    else:
        return np.concatenate((D1, D2), axis = 1)

def P(X, Mu, phi = "Gauss"):
    N = X.shape[0]
    I = np.identity(N)
    D_ = D(X, Mu, phi)
    return I - D_.dot(la.pinv(D_.T.dot(D_))).dot(D_.T)

def Birth(X, Mu, y, mu):
    N = X.shape[0]
    c = y.shape[1]
    k = Mu.shape[0]
    d = mu.shape[1]
    C_ = C(c + 1, "AIC") # using AIC criterion here
    S = 1
    out = 1
    if k == 0:
        Mu = Mu.reshape(0)
        mu = mu.reshape(d)
        Mu_ = np.concatenate((Mu, mu)).reshape(1, d)
    else:
        Mu_ = np.concatenate((Mu, mu))
    P_ = P(X, Mu)
    P_1 = P(X, Mu_)
    for i in range(c):
        out *= (y[:, i].T.dot(P_).dot(y[:, i]) / y[:, i].T.dot(P_1).dot(y[:, i])) ** (N/2)
    out *= S * np.exp(-C_) / (k + 1)
    return out

def Death(X, Mu, y, j):
    N = X.shape[0]
    c = y.shape[1]
    k = Mu.shape[0]
    C_ = C(c + 1, "AIC") # using AIC criterion here
    S = 1
    out = 1
    Mu_ = np.concatenate((Mu[:j, :], Mu[j + 1:, :]))
    P_ = P(X, Mu)
    P_1 = P(X, Mu_)
    for i in range(c):
        out *= (y[:, i].T.dot(P_).dot(y[:, i]) / y[:, i].T.dot(P_1).dot(y[:, i])) ** (N/2)
    out *= k * np.exp(C_) / S
    return out

def Split(X, Mu, y, s, j, mu1, mu2):
    N = X.shape[0]
    c = y.shape[1]
    k = Mu.shape[0]
    d = Mu.shape[1]
    C_ = C(c + 1, "AIC") # using AIC criterion here
    out = 1
    Mu_ = np.concatenate((Mu[:j, :], mu1, mu2, Mu[j + 1:, :]))
    P_ = P(X, Mu)
    P_1 = P(X, Mu_) 
    for i in range(c):
        out *= (y[:, i].T.dot(P_).dot(y[:, i]) / y[:, i].T.dot(P_1).dot(y[:, i])) ** (N/2)
    out *= k * s * np.exp(C_) / (k + 1)
    return out

def Merge(X, Mu, y, s, j1, j2, mu):
    N = X.shape[0]
    c = y.shape[1]
    k = Mu.shape[0]
    d = Mu.shape[1]
    C_ = C(c + 1, "AIC") # using AIC criterion here
    out = 1
    [j1, j2] = [j2, j1] if j1 > j2 else [j1, j2]
    Mu_ = np.concatenate((Mu[:j1, :], Mu[j1 + 1 : j2, :], Mu[j2 + 1:, :], mu))
    P_ = P(X, Mu)
    P_1 = P(X, Mu_) 
    for i in range(c):
        out *= (y[:, i].T.dot(P_).dot(y[:, i]) / y[:, i].T.dot(P_1).dot(y[:, i])) ** (N/2)
    out *= k * np.exp(C_) / (s * (k - 1))
    return out

def Update1(X, Mu, y, mu):
    d = Mu.shape[1]
    Min = X.min(axis = 0)
    Max = X.max(axis = 0)
    W = Max - Min
    t = 0.5
    mu = np.zeros((1, d))
    for i in range(d):
        mu[0, i] = random.uniform(Min[i] - t * W[i], Max[i] + t * W[i])
    return mu

def Update2(X, Mu, y, mu):
    d = Mu.shape[1]
    I = np.identity(d)
    mean = mu
    sigma_ = np.cov(mu)
    sigma = I * sigma_
    mu = np.random.multivariate_normal(mean, sigma).reshape(1, d)
    return mu

def Update(X, Mu, y):
    N = X.shape[0]
    c = y.shape[1]
    k = Mu.shape[0]
    d = Mu.shape[1]
    threshold = 0.5 # np.random.rand()
    sigma = np.cov(Mu, rowvar = False)
    for j in range(Mu.shape[0]):
        mu = Mu[j, :]
        w = np.random.rand()
        mu_ = Update1(X, Mu, y, mu) if w <= threshold else Update2(X, Mu, y, mu)
        Mu_ = Mu
        Mu_[j, :] = mu_
        P_ = P(X, Mu)
        P_1 = P(X, Mu_)
        RJSA = 1
        for i in range(c):
            RJSA *= (y[:, i].T.dot(P_).dot(y[:, i]) / y[:, i].T.dot(P_1).dot(y[:, i])) ** (N/2)
        u = np.random.rand()
        if u <= A(RJSA):
            Mu = Mu_
        else:
            pass
    return Mu

def Loss(X, Mu, y):
    N = X.shape[0]
    c = y.shape[1]
    tao = np.zeros((c, c))

    D_ = D(X, Mu)
    P_ = P(X, Mu)

    alpha = la.pinv(D_.T.dot(D_)).dot(D_.T).dot(y)

    for t in range(c):
        tao[t, t] = 1 / N * y[:, t].T.dot(P_).dot(y[:, t])

    n = np.zeros((N, c))
    for t in range(N):
        nt = np.random.multivariate_normal(np.zeros(c), tao).reshape(1, c)
        n[t, :] = nt  
    
    predict = D_.dot(alpha) + n

    return (la.norm(y - predict)) / la.norm(y)

def Pi(z):
    # using Gauss Distribution here
    # mean =
    pass 
