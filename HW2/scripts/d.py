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
        
        pass
    
    if result.shape[0] > 0:
        result = np.concatenate((result, y_pred), axis=0)
    else:
        result = y_pred
        
sio.savemat("../results/d.mat", {'d':result})