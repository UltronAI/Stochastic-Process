import scipy.io as sio
import numpy as np
from utils2 import *
import warnings, time, os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# data1_path = "data/data1.mat" : x: 1000 x 2 | y: 1000 x 2 | xtest: 200 x 2
# data2_path = "data/data2.mat" : x: 1000 x 3 | y: 1000 x 2 | xtets: 200 x 3

for order in [2]: # [1, 2]:

    print(">> Loading data from {} ...".format("data/data{}.mat".format(order)))
    data = sio.loadmat("data/data{}.mat".format(order))

    x = data["x"]
    y = data["y"]
    xtest = data["xtest"]

    print("x:", x.shape)
    print("y:", y.shape)
    print("xtest:", xtest.shape)

    iter = 10

    [N, d] = x.shape
    c = y.shape[1]

    k, Theta = Init(x, y)
    threshold = 0
    loss = np.array([])

    phi = Phi(x, Theta)

    t0 = time.time()
    for i in range(iter):
        k = Theta.shape[0]
        alpha = Alpha(x, Theta, y)
        tao = Tao(x, Theta, y)
        loss_ = Loss(x, Theta, y, alpha, tao)
        loss = np.append(loss, loss_)
        t1 = time.time()
        t = t1 - t0
        print("[ %d ] [ Iteration %d ] [ time = %.4f ] [ k = %d ] [ Theta.row_num = %d ] [ loss = %.5f ]" % (order, i, t, k, Theta.shape[0], loss_))
        if k <= 50:
            break
        t0 = time.time()
        if np.mod(i, 100) == 0: # save model per 50 iter
            if i != 0:
                plt.figure()
                plt.plot(np.arange(i + 1), loss)
                plt.savefig("model/MDL/loss{0}.png".format(order))
            # t1 = time.time()
            # t = t1 - t0
            np.save("model/MDL/Loss{0}.npy".format(order), loss)
            np.save("model/MDL/Theta{0}.npy".format(order), Theta)
            np.save("model/MDL/Alpha{0}.npy".format(order), alpha)
            np.save("model/MDL/Tao{0}.npy".format(order), tao)
            # print("[ %d ] [ Iteration %d ] [ time = %.4f ] [ k = %d ] [ Theta.row_num = %d ] [ loss = %.5f ]" % (order, i, t, k, Theta.shape[0], loss_))
            # t0 = time.time()
        # update Theta
        Theta = Update(x, Theta, y)
        # delete useless nodes
        Theta = Reduce(x, Theta, y, threshold, i)

    alpha = Alpha(x, Theta, y)
    tao = Tao(x, Theta, y)
    t1 = time.time()
    t = t1 - t0
    np.save("model/MDL/Loss{0}.npy".format(order), loss)
    np.save("model/MDL/Theta{0}.npy".format(order), Theta)
    np.save("model/MDL/Alpha{0}.npy".format(order), alpha)
    np.save("model/MDL/Tao{0}.npy".format(order), tao)
    print("[ %d ] [ Iteration %d ] [ time = %.4f ] [ k = %d ] [ Theta.row_num = %d ] [ loss = %.5f ]" % (order, i, t, k, Theta.shape[0], loss_))
    plt.figure()
    plt.plot(np.arange(i + 1), loss)
    plt.savefig("model/MDL/loss{0}.png".format(order))

    print("************ Do testing ************")
    ytest = Predict(xtest, Theta, alpha, tao, c)
    np.save("model/MDL/v{0}.npy".format(order), ytest)
    print("******** model-MDL-v{} is saved ********".format(order))