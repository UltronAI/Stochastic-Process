# -*- coding: utf-8 -*-

import scipy.io as sio
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
    
    id1 = np.where(y==-1)[0]
    id2 = np.where(y!=-1)[0]
    
    N = x.shape[0]
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
    
    for i in range(0,N):
        start = i*n_val
        if N-n_test-start < n_val:
            break
        x_train = np.concatenate((x[:start,:], x[start+n_val:,:]))
        y_train = np.concatenate((y[:start,:], y[start+n_val:,:]))
        x_val = x[start:start+n_val, :]
        y_val = y[start:start+n_val, :]
        
        w = LA.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
        w = w/LA.norm(w)
        
        train_loss = mse(x_train.dot(w), y_train)
        val_loss = mse(x_val.dot(w), y_val)
        
        if min_loss > (train_loss+val_loss)/2:
#            print("^^^^")
            min_id = i+1
            min_w = w
            min_loss = (train_loss+val_loss)/2
        print("#%d\tval_loss=%.3f\ttrain_loss=%.3f" % (i+1, val_loss, train_loss))
        
    loss = mse(x.dot(min_w), y)
    y_pred = x_test.dot(min_w)
    print("[use #{}] total loss={}".format(min_id, loss))
    
    if result.shape[0] > 0:
        result = np.concatenate((result, y_pred), axis=0)
    else:
        result = y_pred
        
sio.savemat("../results/a.mat", {'a':result})