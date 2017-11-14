import scipy.io as sio
import numpy as np
from utils import *
import warnings, time, os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# data1_path = "data/data1.mat" : x: 1000 x 2 | y: 1000 x 2 | xtest: 200 x 2
# data2_path = "data/data2.mat" : x: 1000 x 3 | y: 1000 x 2 | xtets: 200 x 3

for order in [1, 2]:

    print(">> Loading data from {} ...".format("data/data{}.mat".format(order)))
    data = sio.loadmat("data/data{}.mat".format(order))

    x = data["x"]
    y = data["y"]
    xtest = data["xtest"]

    print("x:", x.shape)
    print("y:", y.shape)
    print("xtest:", xtest.shape)

    [N, d] = x.shape
    c = y.shape[1]

    print("[N, d, c] = [{0}, {1}, {2}]".format(N, d, c))

    print(">> Initializing parameters ...")
    iter = 3000
    s = 1
    k = 0
    k_max = 200
    Mu = np.array([])
    loss = np.array([])

    b_ = lambda k, k_max: 0 if k == k_max else 1 if k == 0 else 0.2
    d_ = lambda k: 0 if k == 0 else 0.2
    s_ = lambda k, k_max: 0 if k == 0 or k == k_max else 0.2
    m_ = lambda k: 0 if k == 0 or k ==1 else 0.2 

    print(">> Starting iteration ...")
    t0 = time.time()
    for i in range(iter):
        k = Mu.shape[0]
        Mu_old = Mu
        alpha = Alpha(x, Mu, y)
        tao = Tao(x, Mu, y)
        loss_ = Loss(x, Mu, y, alpha, tao)
        loss = np.append(loss, loss_)
        if np.mod(i, 50) == 0: # save model per 50 iter
            if i != 0:
                plt.figure()
                plt.plot(np.arange(i + 1), loss)
                plt.savefig("model/RJSA/loss{}.png".format(order))
            os.system("echo $i > log.txt")
            t1 = time.time()
            t = t1 - t0
            np.save("model/RJSA/Loss{}.npy".format(order), loss)
            np.save("model/RJSA/Mu{}.npy".format(order), Mu)
            np.save("model/RJSA/Alpha{}.npy".format(order), alpha)
            np.save("model/RJSA/Tao{}.npy".format(order), tao)
            print("[ Iteration %d ] [ time = %.4f ] [ k = %d ] [ loss = %.5f ]" % (i, t, k, loss_))
            t0 = time.time()
        [bi, di, si, mi] = [b_(k, k_max), d_(k), s_(k, k_max), m_(k)]
        u = np.random.rand()
        if u <= bi:
            # birth move
            # print("birth move")
            u_ = np.random.rand()
            mu = Generate(x)
            if u_ <= A(Birth(x, Mu, y, mu)):
                if k == 0:
                    mu = mu.reshape(d)
                    Mu = np.concatenate((Mu, mu)).reshape(1, d)
                else:
                    Mu = np.concatenate((Mu, mu))
                k = k + 1
            else:
                pass
        elif u <= bi + di:
            # death move
            # print("death move")
            u_ = np.random.rand()
            j = np.random.randint(0, k)
            if u_ <= A(Death(x, Mu, y, j)):
                k = k - 1
                Mu = np.concatenate((Mu[:j, :], Mu[j + 1:, :]))
            else:
                pass
        elif u <= bi + di + si:
            # split move
            # print("split move")
            u_ = np.random.rand()
            j = np.random.randint(0, k)
            mu = Mu[j, :]
            u_1 = np.random.rand(1, d)
            mu1 = mu - u_1 * s
            mu2 = mu + u_1 * s
            if u_ <= A(Split(x, Mu, y, s, j, mu1, mu2)):
                k = k + 1
                Mu = np.concatenate((Mu[:j, :], Mu[j + 1:, :], mu1, mu2))
            else:
                pass
        elif u <= bi + di + si + mi:
            # merge move
            # print("merge move")
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
            if u_ <= A(Merge(x, Mu, y, s, j1, j2, mu)):
                k = k - 1
                Mu = np.concatenate((Mu[:j1, :], Mu[j1 + 1: j2, :], Mu[j2 + 1:, :], mu))
            else:
                pass
        else:
            # update RBF centre
            # print("update move")
            if k == 1:
                pass
            else:
                Mu = Update(x, Mu, y)

        # perform a MH step with the annealed acceptance ratio
        # SA method 1
        """
        alpha = Alpha(x, Mu, y)
        tao = Tao(x, Mu, y)
        Mu = SA1(x, Mu, y, alpha, tao, i)
        """
        # SA method 2
        alpha = Alpha(x, Mu, y)
        tao = Tao(x, Mu, y)
        alpha_old = Alpha(x, Mu_old, y)
        tao_old = Tao(x, Mu_old, y)
        Mu = SA2(x, y, i, Mu, Mu_old, alpha, alpha_old, tao, tao_old)

    # save model
    os.system("echo $i > log.txt")
    alpha = Alpha(x, Mu, y)
    tao = Tao(x, Mu, y)
    np.save("model/RJSA/Mu{}.npy".format(order), Mu)
    np.save("model/RJSA/Alpha{}.npy".format(order), alpha)
    np.save("model/RJSA/Tao{}.npy".format(order), tao)
    plt.figure()
    plt.plot(np.arange(i + 1), loss)
    plt.savefig("model/RJSA/loss{}.png".format(order))
    print("[ Iteration %d ] [ time = %.4f ] [ k = %d ] [ loss = %.5f ]" % (i, t, k, loss_))
    print(">> Training is over. Do testing ...")

    # test
    ytest = Predict(xtest, Mu, alpha, tao, c)
    np.save("model/RJSA/v{}.npy".format(order), ytest)
    print("*** model-RJSA-v{} is saved ***".format(order))
