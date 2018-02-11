# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import numpy.linalg as LA
from utils import mse

result = np.array([]).reshape(0,1)

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
    
    x = np.concatenate((x, x**2, x**3), axis=1)
    
    id1 = np.where(y==-1)[0]
    id2 = np.where(y!=-1)[0]
    
    d = x.shape[1]
    N = x.shape[0]
    n_test = id1.shape[0]
    n_val = int((N-n_test)*0.1)
    n_train = N-n_val-n_test
    
    print("length of x:", d)
    print("DataNum for train:", n_train)
    print("DataNum for test:", n_test)
    print("DataNum for validation:", n_val)
    
    x_test = x[id1]
    y_test = y[id1]
    x = x[id2]
    y = y[id2]
    
    min_loss = 999999
    min_id = -1
    
    for i in range(0,N):
        start = i*n_val
        if N-n_test-start < n_val:
            break
        x_train = np.concatenate((x[:start,:], x[start+n_val:,:]))
        y_train = np.concatenate((y[:start,:], y[start+n_val:,:]))
        x_val = x[start:start+n_val, :]
        y_val = y[start:start+n_val, :]
        
        Sn = 1
        Sw = np.identity(d)
        
        A = Sw.dot(x_train.T).dot(LA.inv(Sn*np.identity(n_train)+x_train.dot(Sw).dot(x_train.T)))
        w_mean = A.dot(y_train)
        w_cov = Sw-A.dot(x_train).dot(Sw)
        
        y_train_pred = np.empty(n_train)
        for j in range(n_train):
            x_ = x_train[j]
            f_ = np.random.normal(x_.dot(w_mean), x_.dot(w_cov).dot(x_.T))
            n_ = np.random.normal(0, Sn)
            y_train_pred[j] = f_+n_
        train_loss = mse(y_train_pred, y_train)
        
        y_val_pred = np.empty(n_val)
        for j in range(n_val):
            x_ = x_val[j]
            f_ = np.random.normal(x_.dot(w_mean), x_.dot(w_cov).dot(x_.T))
            n_ = np.random.normal(0, Sn)
            y_val_pred[j] = f_+n_
        val_loss = mse(y_val_pred, y_val)
        
        mean_loss = train_loss*0.25+val_loss*0.75
        
        if min_loss > mean_loss:
            min_id = i+1
            min_wmean = w_mean
            min_wcov = w_cov
            min_loss = mean_loss
            
        print("#%d \t valLoss=%.3f \t trainLoss=%.3f \t meanLoss=%.3f" % (i+1, val_loss, train_loss, mean_loss))
    
    y_total_pred = np.empty(N-n_test)
    for j in range(N-n_test):
        x_ = x[j]
        f_ = np.random.normal(x_.dot(min_wmean), x_.dot(min_wcov).dot(x_.T))
        n_ = np.random.normal(0, Sn)
        y_total_pred[j] = f_+n_
    loss = mse(y_total_pred, y)
    print("[use #{}] total loss={}".format(min_id, loss))
    
    y_pred = np.empty(n_test)
    for j in range(n_test):
        x_ = x_test[j]
        f_ = np.random.normal(x_.dot(min_wmean), x_.dot(min_wcov).dot(x_.T))
        n_ = np.random.normal(0, Sn)
        y_pred[j] = f_+n_
        
    sio.savemat("../results/c_{}_{}.mat".format(year, loss), {'Ypred': y_pred})
    
    if result.shape[0] > 0:
        result = np.concatenate((result, y_pred), axis=0)
    else:
        result = y_pred
        
sio.savemat("../results/c.mat", {'c':result})