import scipy.io as sio
import numpy as np
import numpy.linalg as la
from utils_OLS import *
import warnings, time, os, random
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

for order in [1, 2]:

    print("*** Loading data from {} ***".format("data/data{}.mat".format(order)))
    data = sio.loadmat("data/data{}.mat".format(order))

    phi = "Gauss"

    x = data["x"]
    y = data["y"]
    xtest = data["xtest"]

    index_max = y[:, 1].argsort()[-2:][::-1]
    x = np.delete(x, index_max, 0)
    y = np.delete(y, index_max, 0)
    index_min = y[:, 1].argsort()[:2][::-1]
    x = np.delete(x, index_min, 0)
    y = np.delete(y, index_min, 0)

    print("x:", x.shape)
    print("y:", y.shape)
    print("xtest:", xtest.shape)

    N = x.shape[0]
    val_index = random.sample(range(N), 200)
    train_index = [i for i in range(N) if i not in val_index]

    x_val = x[val_index, :]
    y_val = y[val_index, :]
    x_train = x[train_index, :]
    y_train = y[train_index, :]

    [N_train, d] = x_train.shape
    N_val = x_val.shape[0]
    c = y_train.shape[1]
    print("[N_train, N_val, d, c] = [{0}, {1}, {2}, {3}]".format(N_train, N_val, d, c))

    Q = Phi(x_train, x_train, phi)

    M = Q.shape[1]
    rho = 1e-4 if order == 1 else 5e-5
    ERR = 0
    iter = 2000
    W = np.array([])
    Arg = np.array([])
    A = np.array([])
    C = np.array([])

    k_max = 300
    valLoss = np.array([])
    trainLoss = np.array([])

    t0 = time.time()
    print(">> starting iterations ...")
    for it in range(iter):
        if W.shape[0] == 0:
            k = 0
            W_train = W
            W_val = np.array([])
        else:
            k = W.shape[1]
            W_train = W
            Q_val = Phi(x_val, C, phi)
            if k == 1:
                W_val = Q_val
            else:
                W_val = Q_val.dot(la.inv(A))

        alpha_train = Alpha(x_train, W_train, y_train, phi)
        alpha_val = Alpha(x_val, W_val, y_val, phi)
        tao_train = Tao(x_train, W_train, y_train, phi)
        tao_val = Tao(x_val, W_val, y_val, phi)
        loss_train = Loss(x_train, W_train, y_train, alpha_train, tao_train, phi)
        trainLoss = np.append(trainLoss, loss_train)
        loss_val = Loss(x_val, W_val, y_val, alpha_val, tao_val, phi)   
        valLoss = np.append(valLoss, loss_val)

        if np.mod(it, 50) == 0: # save model per 50 iter
            if it != 0:
                plt.close()
                plt.figure()
                plt.plot(np.arange(it + 1), valLoss, "r", label = "loss_val")
                plt.plot(np.arange(it + 1), trainLoss, "b", label = "loss_train")
                plt.legend()
                plt.savefig("model/OLS/loss{0}_{1}.png".format(order, phi))
            np.save("model/OLS/valLoss{0}_{1}.npy".format(order, phi), valLoss)
            np.save("model/OLS/trainLoss{0}_{1}.npy".format(order, phi), trainLoss)
            np.save("model/OLS/C{0}_{1}.npy".format(order, phi), C)
            np.save("model/OLS/A{0}_{1}.npy".format(order, phi), A)
            t1 = time.time()
            t = t1 - t0
            print("[ %d ] [ %s ] [ Iteration %d ] [ time = %.4f ] [ k = %d ] [ ERR = %.5f ] [ val_loss = %.5f ] [ train_loss = %.5f ]" % (order, phi, it, t, k, ERR, loss_val, loss_train))
            t0 = time.time()    

        if k >= k_max or 1 - ERR < rho:
            print("ERR", ERR)
            iter = it + 1
            break
        
        if k>= 10 and abs(valLoss[-1] - valLoss[-2]) < 1e-5 and abs(trainLoss[-1] - trainLoss[-2]) < 1e-5:
            iter = it + 1
            break

        if it == 0:
            W_ = Q
        else:
            W_ = GetW(W, W_, Q, Arg)
        arg, err = ArgErrMax(W_, y_train, Arg)
        if arg is None or arg in Arg:
            continue
        ERR += err
        Arg = np.append(Arg, arg)
        if W.shape[0] == 0:
            W = np.append(W, W_[:, arg]).reshape(N_train, 1)
            A = GetA(W, Q, A, arg)
            C = np.append(C, x[arg, :]).reshape(1, d)
        else:
            W = np.concatenate((W, W_[:, arg].reshape(N_train, 1)), axis = 1)
            A = GetA(W, Q, A, arg)
            C = np.concatenate((C, x[arg, :].reshape(1, d)), axis = 0)

    np.save("model/OLS/valLoss{0}_{1}.npy".format(order, phi), valLoss)
    np.save("model/OLS/trainLoss{0}_{1}.npy".format(order, phi), trainLoss)
    np.save("model/OLS/Mu{0}_{1}.npy".format(order, phi), C)
    np.save("model/OLS/A{0}_{1}.npy".format(order, phi), A)
    plt.close()    
    plt.figure()
    plt.plot(np.arange(iter), valLoss, "r", label = "loss_val")
    plt.plot(np.arange(iter), trainLoss, "b", label = "loss_train")
    plt.legend()
    plt.savefig("model/OLS/loss{0}_{1}.png".format(order, phi))
    t1 = time.time()
    t = t1 - t0
    print("[ %d ] [ %s ] [ Iteration %d ] [ time = %.4f ] [ k = %d ] [ val_loss = %.5f ] [ train_loss = %.5f ]" % (order, phi, it, t, k, loss_val, loss_train))

    Q_total = Phi(x, C, phi)
    W_total = Q_total.dot(la.inv(A))

    alpha_total = Alpha(x, W_total, y, phi)
    tao_total = Tao(x, W_total, y, phi)
    loss_total = Loss(x, W_total, y, alpha_total, tao_total, phi)

    print("total loss = %.5f, k = %d" % (loss_total, k))

    print("************ Do testing ************")

    # test
    Q_test = Phi(xtest, C, phi)
    W_test = Q_test.dot(la.inv(A))

    ytest = Predict(xtest, W_test, alpha_total, tao_total, c, phi)
    np.save("model/OLS/v{0}_{1}.npy".format(order, phi), ytest)
    print("******** model-OLS-v{} is saved ********".format(order))
