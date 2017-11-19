import numpy as np
import numpy.linalg as la
import random
from math import ceil

def A(_):
    return min(1, _)

def GenerateMu(X):
    x_min = X.min(axis = 1)
    x_max = X.max(axis = 1)
    delta = x_max - x_min
    d = X.shape[1]
    t = 0.5
    mu = np.zeros((1, d))
    for i in range(d):
        mu[0, i] = random.uniform(x_min[i] - t * delta[i], x_max[i] + t * delta[i])
    # mu = np.array([np.random.uniform(x_min[i] - delta, x_max[i] + delta, 1) for i in range(d)]).reshape(1, d)
    return mu

def Init(X, y):
    [k, d] = X.shape
    k = ceil(k / 2)
    Theta = np.zeros((k, d + 1))
    for i in range(k):
        mu = GenerateMu(X)
        s = np.cov(X[i, :]).reshape(1, 1)
        Theta[i, :] = np.concatenate((mu, s), axis = 1)
    return k, Theta

def Gauss(x, Mu, s):
    return np.exp(- la.norm(x - Mu, axis = 1) * s ** 2)

def Phi(X, Theta):
    N = X.shape[0]
    k = Theta.shape[0]
    out = np.empty((N, k))
    Mu = Theta[:, 0 : -1]
    s = Theta[:, -1].T
    for i in range(N):
        out[i, :] = Gauss(X[i, :], Mu, s)
    return out

def D(X, Theta):
    D1 = np.ones((X.shape[0], 1))
    D2 = X
    if Theta.shape[0] > 0:
        D3 = Phi(X, Theta)
        return np.concatenate((D1, D2, D3), axis = 1)
    else:
        return np.concatenate((D1, D2), axis = 1)

def P(X, Theta):
    N = X.shape[0]
    I = np.identity(N)
    D_ = D(X, Theta)
    return I - D_.dot(la.pinv(D_.T.dot(D_))).dot(D_.T)

def UpdateTheta(X, Theta, y):
    d = Theta.shape[1] - 1
    x_min = X.min(axis = 0)
    x_max = X.max(axis = 0)
    S = np.cov(X)
    s_min = min([S[i, i] for i in range(X.shape[0])])
    s_max = max([S[i, i] for i in range(X.shape[0])])
    s = np.array([random.uniform(s_min, s_max)]).reshape(1, 1)
    W = x_max - x_min
    t = 0.5
    mu = np.zeros((1, d))
    for i in range(d):
        mu[0, i] = random.uniform(x_min[i] - t * W[i], x_max[i] + t * W[i])
    theta = np.concatenate((mu, s), axis = 1)
    return theta

def Update(X, Theta, y):
    [N, d] = X.shape
    k = Theta.shape[0]
    c = y.shape[1]
    for j in range(k):
        theta_ = UpdateTheta(X, Theta, y)
        Theta_ = Theta
        Theta_[j, :] = theta_
        P_ = P(X, Theta)
        P_1 = P(X, Theta_)
        MDL = 1
        for i in range(c):
            MDL *= (y[:, i].T.dot(P_).dot(y[:, i]) / y[:, i].T.dot(P_1).dot(y[:, i])) ** (N/2)
        u = np.random.rand()
        if u <= A(MDL):
            Theta = Theta_
        else:
            pass
    return Theta

def Alpha(X, Theta, y):
    D_ = D(X, Theta)
    alpha = la.pinv(D_.T.dot(D_)).dot(D_.T).dot(y)
    return alpha

def Tao(X, Theta, y):
    N = X.shape[0]
    c = y.shape[1]
    tao = np.zeros((c, c))

    D_ = D(X, Theta)
    P_ = P(X, Theta)

    for t in range(c):
        tao[t, t] = 1 / N * y[:, t].T.dot(P_).dot(y[:, t])
    return tao

def Predict(X, Theta, alpha, tao, c):
    N = X.shape[0]
    D_ = D(X, Theta)

    n = np.zeros((N, c))
    for t in range(N):
        nt = np.random.multivariate_normal(np.zeros(c), tao ** 2).reshape(1, c)
        n[t, :] = nt  
    
    predict = D_.dot(alpha) + n
    
    return predict

def Loss(X, Theta, y, alpha, tao):
    predict = Predict(X, Theta, alpha, tao, y.shape[1])
    mse = ((predict - y) ** 2).mean()
    return mse # 0.5 * np.sum((predict - y) ** 2)

def T(i):
    return 2 / i ** 2 if i > 0 else 666

def Pi(T, f):
    return np.exp(-f / T)

def Reduce(X, Theta, y, threshold, iter):
    k = Theta.shape[0]
    alpha = Alpha(X, Theta, y)
    tao = Tao(X, Theta, y)
    R = np.sum(alpha[-k:, :], axis = 1)
    phi = Phi(X, Theta)
    N = phi.shape[0]
    r = np.sum(phi * R.T, axis = 0) / N
    print(r.max())
    print(r.min())
    index = np.where(r >= threshold)
    print(index)
    print(index[0].shape)
    Theta_ = Theta[index]
    print(Theta_.shape)
    alpha_ = Alpha(X, Theta_, y)
    tao_ = Tao(X, Theta_, y)
    T_ = T(iter)
    loss = Loss(X, Theta, y, alpha, tao)
    loss_ = Loss(X, Theta_, y, alpha_, tao_) 
    u = np.random.rand()
    if loss_ < loss:
        Theta = Theta_
    else:
        pass