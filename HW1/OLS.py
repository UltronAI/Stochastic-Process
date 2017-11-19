import scipy.io as sio
import numpy as np
from utils_OLS import *
import warnings, time, os, random
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

order = 1

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
rho = 1e-5
ERR = 0
iter = 2000
W = np.array([])
Arg = np.array([])

k_max = 400
valLoss = np.array([])
trainLoss = np.array([])

t0 = time.time()
for it in range(iter):
    if W.shape[0] == 0:
        k = 0
        Mu = np.array([])
    else:
        k = W.shape[1]
        if Mu.shape[0] == 0:
            Mu = x[arg, :].reshape(1, d)
        else:
            Mu = np.concatenate((Mu, x[arg, :].reshape(1, d)), axis = 0)
    alpha = Alpha(x_train, Mu, y_train, phi)
    tao = Tao(x_train, Mu, y_train, phi)
    loss_val = Loss(x_val, Mu, y_val, alpha, tao, phi)
    valLoss = np.append(valLoss, loss_val)
    loss_train = Loss(x_train, Mu, y_train, alpha, tao, phi)
    trainLoss = np.append(trainLoss, loss_train)
    print("[ %d ] [ %s ] [ Iteration %d ] [ k = %d ] [ val_loss = %.5f ] [ train_loss = %.5f ]" % (order, phi, it, k, loss_val, loss_train))
    if np.mod(it, 100) == 0: # save model per 50 iter
        if it != 0:
            plt.figure()
            plt.plot(np.arange(it + 1), valLoss, "r", label = "loss_val")
            plt.plot(np.arange(it + 1), trainLoss, "b", label = "loss_train")
            plt.legend()
            plt.savefig("model/OLS/loss{0}_{1}.png".format(order, phi))
        t1 = time.time()
        t = t1 - t0
        np.save("model/OLS/valLoss{0}_{1}.npy".format(order, phi), valLoss)
        np.save("model/OLS/trainLoss{0}_{1}.npy".format(order, phi), trainLoss)
        np.save("model/OLS/Mu{0}_{1}.npy".format(order, phi), Mu)
        np.save("model/OLS/Alpha{0}_{1}.npy".format(order, phi), alpha)
        np.save("model/OLS/Tao{0}_{1}.npy".format(order, phi), tao)
        # print("[ %d ] [ %s ] [ Iteration %d ] [ time = %.4f ] [ k = %d ] [ val_loss = %.5f ] [ train_loss = %.5f ]" % (order, phi, it, t, k, loss_val, loss_train))
        t0 = time.time()    

    if k >= k_max:
        print("ERR", ERR)
        break
    if it == 0:
        W_ = Q
    else:
        W_ = GetW(W, W_, Q, Arg)
    arg, err = ArgErrMax(W_, y_train, Arg)
    ERR += err
    if arg in Arg:
        continue
    Arg = np.append(Arg, arg)
    if W.shape[0] == 0:
        W = np.append(W, W_[:, arg]).reshape(N_train, 1)
    else:
        W = np.concatenate((W, W_[:, arg].reshape(N_train, 1)), axis = 1)
