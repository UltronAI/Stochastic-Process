# -*- coding: utf-8 -*-

import tensorflow as tf
import scipy.io as sio
import numpy.linalg as LA
from utils import mse
import numpy as np

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
    D = x.shape[1]
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
    H = 25
    
    min_loss = 999999
    min_id = -1
    
    xx = tf.placeholder(tf.float32, shape=(None,D))
    yy = tf.placeholder(tf.float32, shape=(None,1))

    init = tf.contrib.layers.xavier_initializer()
    h = tf.layers.dense(inputs=xx, units=H, activation=tf.nn.relu,
            kernel_initializer=init)
    yy_pred = tf.layers.dense(inputs=h, units=1,
            kernel_initializer=init)

    loss = tf.losses.mean_squared_error(yy_pred, yy)
    optimizer = tf.train.RMSPropOptimizer(1e-3)
    updates = optimizer.minimize(loss)
        
    saver = tf.train.Saver()
    
    for i in range(0,N):
        start = i*n_val
        if N-n_test-start < n_val:
            break
        x_train = np.concatenate((x[:start,:], x[start+n_val:,:]))
        y_train = np.concatenate((y[:start,:], y[start+n_val:,:]))
        x_val = x[start:start+n_val, :]
        y_val = y[start:start+n_val, :]
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            values = {xx: x_train, yy: y_train}
            for t in range(2000):
                train_loss, _ = sess.run([loss, updates], feed_dict=values)
            val_loss = sess.run(loss, feed_dict={xx: x_val, yy: y_val})

            mean_loss = train_loss*0.25+val_loss*0.75

            if min_loss > mean_loss:
                min_id = i+1
                min_loss = (train_loss+val_loss)/2
                save_path = saver.save(sess, "../results/nn_{}.ckpt".format(year))
            print("#%d \t valLoss=%.3f \t trainLoss=%.3f \t meanLoss=%.3f" % (i+1, val_loss, train_loss, mean_loss))
    
    with tf.Session() as sess:
        saver.restore(sess, "../results/nn_{}.ckpt".format(year))
        total_loss = sess.run(loss, feed_dict={xx: x, yy: y})
        y_pred = sess.run(yy_pred, feed_dict={xx: x_test, yy:y})
        
        print("[use #{}] total loss={}".format(min_id, total_loss))
                
        sio.savemat("../results/nn_{}_{}.mat".format(year, total_loss), {'Ypred': y_pred})
        
        if result.shape[0] > 0:
            result = np.concatenate((result, y_pred), axis=0)
        else:
            result = y_pred
            
sio.savemat("../results/nn.mat", {'nn':result})
