import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio
import random

for order in [1, 2]:

    iter = 10000 if order == 1 else 40000

    print(">> Loading data from {} ...".format("data/data{}.mat".format(order)))
    data = sio.loadmat("data/data{}.mat".format(order))

    X = data["x"]
    Y = data["y"]
    xtest = data["xtest"]

    index_max = Y[:, 1].argsort()[-2:][::-1]
    X = np.delete(X, index_max, 0)
    Y = np.delete(Y, index_max, 0)
    index_min = Y[:, 1].argsort()[:2][::-1]
    X = np.delete(X, index_min, 0)
    Y = np.delete(Y, index_min, 0)

    print("xtrain:", X.shape)
    print("ytrain:", Y.shape)
    print("xtest:", xtest.shape)

    [N, d] = X.shape
    c = Y.shape[1]

    val_index = random.sample(range(N), 200)
    train_index = [i for i in range(N) if i not in val_index]

    x_val = X[val_index, :]
    y_val = Y[val_index, :]
    x_train = X[train_index, :]
    y_train = Y[train_index, :]

    N_train = x_train.shape[0]
    N_val = x_val.shape[0]

    print("[N_train, N_val, d, c] = [{0}, {1}, {2}, {3}]".format(N_train, N_val, d, c))

    H1 = 50
    H2 = 25
    H3 = 50
    H4 = 25

    with tf.device('/gpu:0'):
        x = tf.placeholder(tf.float32, shape = (None, d))
        y = tf.placeholder(tf.float32, shape = (None, c))

        init = tf.contrib.layers.xavier_initializer()
        h1 = tf.layers.dense(inputs=x, units=H1, activation=tf.nn.relu,
                        kernel_initializer=init)
        h2 = tf.layers.dense(inputs=h1, units=H2, activation=tf.nn.tanh,
                        kernel_initializer=init)
        h3 = tf.layers.dense(inputs=h2, units=H3, activation=tf.nn.relu,
                        kernel_initializer=init)
        h4 = tf.layers.dense(inputs=h3, units=H4, activation=tf.nn.tanh,
                        kernel_initializer=init)
        y_pred = tf.layers.dense(inputs=h4, units=c, # activation=tf.nn.sigmoid,
                        kernel_initializer=init)

    loss = tf.losses.mean_squared_error(y_pred, y)

    optimizer = tf.train.RMSPropOptimizer(1e-3)
    updates = optimizer.minimize(loss)

    saver = tf.train.Saver()

    trainLoss = np.array([])
    valLoss = np.array([])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        values = { x: x_train,
                   y: y_train,
                }
        print(">> starting iteration ...")
        for t in range(iter):
            loss_train, _ = sess.run([loss, updates],
                                   feed_dict=values)
            trainLoss = np.append(trainLoss, loss_train)
            loss_val = sess.run(loss, feed_dict = {x: x_val, y: y_val})
            valLoss = np.append(valLoss, loss_val)
            if np.mod(t, 100) == 0:
                print("[ %d ] [ Iteration %d] [ loss_train = %.5f ] [ loss_val = %.5f ]" % (order, t, loss_train, loss_val))
                if t == 0:
                    continue
                plt.figure()
                plt.plot(np.arange(t + 1), valLoss, "r", label = "loss_val")
                plt.plot(np.arange(t + 1), trainLoss, "b", label = "loss_train")
                plt.legend()
                plt.savefig("model/NN/loss{0}.png".format(order))
                save_path = saver.save(sess, "model/NN/nn-model{0}.ckpt".format(order))
        output = sess.run(y_pred, feed_dict = {x: xtest, y: y_train})