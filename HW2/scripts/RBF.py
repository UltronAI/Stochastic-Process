# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy.linalg as LA
from utils import mse
import numpy as np
from utils_rbf import *
import warnings
warnings.filterwarnings("ignore")

result = np.array([]).reshape(0,1)

phi = "Gauss"
method = "RJSA"

for year in [2010, 2011, 2012, 2013, 2014]:
    print(">>>>>")
    data_name = "data{}".format(year)
    data_path = "../data2017/{}.mat".format(data_name)
    
    print("Loading data from \"{}\" ...".format(data_path))
    data = sio.loadmat(data_path)[data_name]
    
    x = data["Score"][0][0]
    y = data["TargetScore1"][0][0]
    
    print("shape of x:", x.shape)
    print("shape of y:", y.shape)
    
    id1 = np.where(y==-1)[0]
    id2 = np.where(y!=-1)[0]
    
    N = x.shape[0]
    d = x.shape[1]
    c = y.shape[1]
    n_test = id1.shape[0]
    n_val = int((N-n_test)*0.1)
    n_train = N-n_val-n_test
    
    print("DataNum for train:", n_train)
    print("DataNum for test:", n_test)
    print("DataNum for validation:", n_val)
    
    x_test = x[id1]
    y_test = y[id1]
    x = x[id2]
    y = y[id2]
    
    min_loss = 999999
    min_id = -1
    
    for ii in range(0,N):
        start = ii*n_val
        if N-n_test-start < n_val:
            break
        x_train = np.concatenate((x[:start,:], x[start+n_val:,:]))
        y_train = np.concatenate((y[:start,:], y[start+n_val:,:]))
        x_val = x[start:start+n_val, :]
        y_val = y[start:start+n_val, :]
        
        k = 0
        iter = 2000
        Mu = np.array([])

        valLoss = np.array([])
        trainLoss = np.array([])

        s = 1
        k_max = 100

        b_ = lambda k, k_max: 0 if k == k_max else 1 if k == 0 else 0.2
        d_ = lambda k: 0 if k == 0 else 0.2
        s_ = lambda k, k_max: 0 if k == 0 or k == k_max else 0.2
        m_ = lambda k: 0 if k == 0 or k ==1 else 0.2
        
        for i in range(iter):
            k = Mu.shape[0]
            Mu_old = Mu
            alpha = Alpha(x_train, Mu, y_train, phi)
            tao = Tao(x_train, Mu, y_train, phi)
            loss_val = Loss(x_val, Mu, y_val, alpha, tao, phi)
            valLoss = np.append(valLoss, loss_val)
            loss_train = Loss(x_train, Mu, y_train, alpha, tao, phi)
            trainLoss = np.append(trainLoss, loss_train)

            if k >= k_max:
                iter = i + 1
                break
            if valLoss.shape[0] >= 100 and abs(valLoss[-1] - valLoss[-5]) < 1e-5 and abs(trainLoss[-1] - trainLoss[-5]) < 1e-5:
                iter = i + 1
                break

            [bi, di, si, mi] = [b_(k, k_max), d_(k), s_(k, k_max), m_(k)]
            u = np.random.rand()
            if u <= bi:
                u_ = np.random.rand()
                mu = Generate2(x_train)
                if u_ <= A(Birth(x_train, Mu, y_train, mu, phi)):
                    if k == 0:
                        Mu = Mu.reshape(0)
                        mu = mu.reshape(d)
                        Mu = np.concatenate((Mu, mu)).reshape(1, d)
                    else:
                        Mu = np.concatenate((Mu, mu))
                    k = k + 1
                else:
                    pass
            elif u <= bi + di:
                u_ = np.random.rand()
                j = np.random.randint(0, k)
                if u_ <= A(Death(x_train, Mu, y_train, j, phi)):
                    k = k - 1
                    Mu = np.concatenate((Mu[:j, :], Mu[j + 1:, :]))
                else:
                    pass
            elif u <= bi + di + si:
                u_ = np.random.rand()
                j = np.random.randint(0, k)
                mu = Mu[j, :]
                u_1 = np.random.rand(1, d)
                mu1 = mu - u_1 * s
                mu2 = mu + u_1 * s
                if u_ <= A(Split(x_train, Mu, y_train, s, j, mu1, mu2, phi)):
                    k = k + 1
                    Mu = np.concatenate((Mu[:j, :], Mu[j + 1:, :], mu1, mu2))
                else:
                    pass
            elif u <= bi + di + si + mi:
                u_ = np.random.rand()
                j1 = np.random.randint(0, k)
                mu1 = Mu[j1, :]
                j2 = FindClosest(Mu, j1)
                mu2 = Mu[j2, :]
                [j1, j2] = [j2, j1] if j1 > j2 else [j1, j2]
                if la.norm(j1 - j2) > 2 * s:
                    i -= 1
                    continue
                mu = ((mu1 + mu2) * 0.5).reshape(1, d)
                if u_ <= A(Merge(x_train, Mu, y_train, s, j1, j2, mu, phi)):
                    k = k - 1
                    Mu = np.concatenate((Mu[:j1, :], Mu[j1 + 1: j2, :], Mu[j2 + 1:, :], mu))
                else:
                    pass
            else:
                if k == 1:
                    pass
                else:
                    Mu = Update(x_train, Mu, y_train, phi)

            if method == "RJSA":
                # perform a MH step with the annealed acceptance ratio
                Mu = SA3(x_train, y_train, i, Mu, Mu_old, phi) 
            elif method == "AIC":
                alpha = Alpha(x_train, Mu, y_train, phi)
                tao = Tao(x_train, Mu, y_train, phi)
                alpha_old = Alpha(x_train, Mu_old, y_train, phi)
                tao_old = Tao(x_train, Mu_old, y_train, phi)
                Mu = AIC(x_train, y_train, Mu, Mu_old, alpha, alpha_old, tao, tao_old, phi)
            elif method == "BIC":
                alpha = Alpha(x_train, Mu, y_train, phi)
                tao = Tao(x_train, Mu, y_train, phi)
                alpha_old = Alpha(x_train, Mu_old, y_train, phi)
                tao_old = Tao(x_train, Mu_old, y_train, phi)
                Mu = BIC(x_train, y_train, Mu, Mu_old, alpha, alpha_old, tao, tao_old, phi)

        mean_loss = loss_train*0.25+loss_val*0.75
        
        if min_loss > mean_loss:
            min_id = ii+1
            min_loss = (loss_train+loss_val)/2
        print("#%d \t valLoss=%.3f \t trainLoss=%.3f \t meanLoss=%.3f" % (ii+1, loss_val, loss_train, mean_loss))
        
#    print("[use #{}] total loss={}".format(min_id, loss))
#    
#    if result.shape[0] > 0:
#        result = np.concatenate((result, y_pred), axis=0)
#    else:
#        result = y_pred
#        
#sio.savemat("../results/a.mat", {'a':result})