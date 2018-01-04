import numpy as np
import numpy.linalg as la
import random

def Linear(s):
    return s

def Cubic(s):
    return s**3

def Pow4(s):
    return s**4

def ThinPlateSpline(s):
    return s**2 * np.log(s)

def Multiquadric(s):
    lamb = 0.5
    return (s**2 + lamb) ** 0.5

def Gauss(s):
    lamb = 1 / 2.5 ** 2
    return np.exp(-lamb * s**2)

def CubicGauss(s):
    lamb = 0.5
    return np.exp(-lamb * s**3)

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
        elif phi == "CubicGauss":
            out[i, :] = CubicGauss(s)
        elif phi == "Pow4":
            out[i, :] = Pow4(s)
        else:
            out[i, :] = Gauss(s)
    return out

def D(X, W = np.array([])):
    D1 = np.ones((X.shape[0], 1))
    D2 = X
    if W.shape[0] > 0:
        D3 = W
        return np.concatenate((D1, D2, D3), axis = 1)
    else:
        return np.concatenate((D1, D2), axis = 1)

def P(X, W = np.array([])):
    N = X.shape[0]
    I = np.identity(N)
    D_ = D(X, W)
    return I - D_.dot(la.pinv(D_.T.dot(D_))).dot(D_.T)

def Alpha(X, W, y, phi = "Gauss"):
    D_ = D(X, W)
    alpha = la.pinv(D_.T.dot(D_)).dot(D_.T).dot(y)
    return alpha

def Tao(X, W, y, phi = "Gauss"):
    N = X.shape[0]
    c = y.shape[1]
    tao = np.zeros((c, c))

    D_ = D(X, W)
    P_ = P(X, W)

    for t in range(c):
        tao[t, t] = 1 / N * y[:, t].T.dot(P_).dot(y[:, t])
    
    return tao

def Predict(X, W, alpha, tao, c, phi = "Gauss"):
    N = X.shape[0]
    D_ = D(X, W)

    n = np.zeros((N, c))
    for t in range(N):
        nt = np.random.multivariate_normal(np.zeros(c), tao ** 2).reshape(1, c)
        n[t, :] = nt  
    
    predict = D_.dot(alpha) + n
    
    return predict

def Loss(X, W, y, alpha, tao, phi = "Gauss"):
    predict = Predict(X, W, alpha, tao, y.shape[1], phi)
    mse = ((predict - y) ** 2).mean()
    return mse # 0.5 * np.sum((predict - y) ** 2)       
            
def ArgErrMax(W, d, Arg):
    c = d.shape[1]
    M = W.shape[1]
    g = np.zeros((c, M))
    e = np.zeros(M)
    for j in range(M):
        for i in range(c):
            g[i, j] = W[:, j].T.dot(d[:, i]) / W[:, j].T.dot(W[:, j])
        e[j] = (g[:, j] ** 2).sum() * W[:, j].T.dot(W[:, j]) / np.trace(d.T.dot(d))
    for i in range(len(e)):
        arg = e.argmax()
        if arg not in Arg:
            break
        else:
            e[arg] = 0
    return arg, e[arg]

def GetW(W, W_, Q, Arg):
    k = W.shape[1]
    [N, M] = W_.shape
    alpha = np.zeros(k)
    for i in range(M):
        if i in Arg:
            continue
        for j in range(k):
            alpha[j] = W[:, j].T.dot(Q[:, i]) / W[:, j].T.dot(W[:, j])
        W_[:, i] = Q[:, i] - (alpha * W).sum(axis = 1)
    return W_

def GetA(W, Q, A, arg):
    if W.shape[0] == 0:
        return np.array([])
    elif W.shape[1] == 1:
        return np.array([1])
    k = W.shape[1]
    k_old = A.shape[0]
    if k == k_old:
        return A
    A_ = np.identity(k)
    A_[0 : k - 1, 0 : k - 1] = A
    for i in range(k - 1):
        A_[i, k - 1] = W[:, -1].T.dot(Q[:, arg]) / W[:, -1].T.dot(W[:, -1])
    return A_