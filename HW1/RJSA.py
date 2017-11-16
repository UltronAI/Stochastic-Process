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

for order in [2]: # [1, 2]:

    print("*** Loading data from {} ***".format("data/data{}.mat".format(order)))
    data = sio.loadmat("data/data{}.mat".format(order))
    phi = "Gauss" if order == 1 else "Gauss"
    use_val = False

    x = data["x"]
    y = data["y"]
    xtest = data["xtest"]

    index_max = y[:, 1].argsort()[-3:][::-1]
    x = np.delete(x, index_max, 0)
    y = np.delete(y, index_max, 0)
    index_min = y[:, 1].argsort()[:2][::-1]
    x = np.delete(x, index_min, 0)
    y = np.delete(y, index_min, 0)

    print("x:", x.shape)
    print("y:", y.shape)
    print("xtest:", xtest.shape)

    n_split = 5
    x_split = np.split(x, n_split)
    y_split = np.split(y, n_split)

    print("******** Initializing parameters ********")
    if order == 1:
        k = 0
        Mu = np.array([])
    else:
        k = 200
        # Mu = np.random.randn(k, x.shape[1])
        Mu = InitMu(k, x)
    """
    k = 0
    Mu = np.array([])
    """
    iter = 2000
    s = 1
    k_max = 500
    loss = np.array([])
    
    print("[k_max, s, iter] = [{0}, {1}, {2}]".format(k_max, s, iter))

    b_ = lambda k, k_max: 0 if k == k_max else 1 if k == 0 else 0.2
    d_ = lambda k: 0 if k == 0 else 0.2
    s_ = lambda k, k_max: 0 if k == 0 or k == k_max else 0.2
    m_ = lambda k: 0 if k == 0 or k ==1 else 0.2

    iter_val = n_split if use_val == True else 1
    for val in range(iter_val):
        if use_val == True:
            print("************ Do training #{} ************".format(val + 1))
            x_val = x_split[val]
            y_val = y_split[val]
            x_train = np.concatenate([f for j, f in enumerate(x_split) if j != val])
            y_train = np.concatenate([f for j, f in enumerate(y_split) if j != val])
        else:
            x_train = x
            x_val = x
            y_train = y
            y_val = y

        [N, d] = x_train.shape
        c = y_train.shape[1]

        print("[N, d, c] = [{0}, {1}, {2}]".format(N, d, c))

        print(">> Starting iteration ...")
        t0 = time.time()
        for i in range(iter):
            k = Mu.shape[0]
            Mu_old = Mu
            alpha = Alpha(x_train, Mu, y_train, phi)
            tao = Tao(x_train, Mu, y_train, phi)
            loss_ = Loss(x_val, Mu, y_val, alpha, tao, phi)
            loss = np.append(loss, loss_)
            if np.mod(i, 100) == 0: # save model per 50 iter
                if i != 0:
                    plt.figure()
                    plt.plot(np.arange(i + 1 + val * iter), loss)
                    if use_val == True:
                        plt.savefig("model/RJSA/loss{0}_{1}_{2}.png".format(order, val, phi))
                    else:
                        plt.savefig("model/RJSA/loss{0}_{1}.png".format(order, phi))
                t1 = time.time()
                t = t1 - t0
                np.save("model/RJSA/Loss{0}_{1}.npy".format(order, phi), loss)
                np.save("model/RJSA/Mu{0}_{1}.npy".format(order, phi), Mu)
                np.save("model/RJSA/Alpha{0}_{1}.npy".format(order, phi), alpha)
                np.save("model/RJSA/Tao{0}_{1}.npy".format(order, phi), tao)
                if use_val == True:
                    print("[ %d ] [ kernel = %s ] [ val = %d ] [ Iteration %d ] [ time = %.4f ] [ k = %d ] [ Mu.row_num = %d ] [ loss = %.5f ]" % (order, phi, val + 1, i, t, k, Mu.shape[0], loss_))
                else:
                    print("[ %d ] [ kernel = %s ] [ Iteration %d ] [ time = %.4f ] [ k = %d ] [ Mu.row_num = %d ] [ loss = %.5f ]" % (order, phi, i, t, k, Mu.shape[0], loss_))
                t0 = time.time()
            [bi, di, si, mi] = [b_(k, k_max), d_(k), s_(k, k_max), m_(k)]
            u = np.random.rand()
            if u <= bi:
                # birth move
                # print("birth move")
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
                # death move
                # print("death move")
                u_ = np.random.rand()
                j = np.random.randint(0, k)
                if u_ <= A(Death(x_train, Mu, y_train, j, phi)):
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
                if u_ <= A(Split(x_train, Mu, y_train, s, j, mu1, mu2, phi)):
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
                if u_ <= A(Merge(x_train, Mu, y_train, s, j1, j2, mu, phi)):
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
                    Mu = Update(x_train, Mu, y_train, phi)

            # perform a MH step with the annealed acceptance ratio
            # SA method 1
            """
            alpha = Alpha(x, Mu, y, phi)
            tao = Tao(x, Mu, y, phi)
            Mu = SA1(x, Mu, y, alpha, tao, i, phi)
            """
            # SA method 2
            """
            alpha = Alpha(x_train, Mu, y_train, phi)
            tao = Tao(x_train, Mu, y_train, phi)
            alpha_old = Alpha(x_train, Mu_old, y_train, phi)
            tao_old = Tao(x_train, Mu_old, y_train, phi)
            Mu = SA2(x_val, y_val, i + val * iter, Mu, Mu_old, alpha, alpha_old, tao, tao_old, phi)
            """
            # SA method 3
            Mu = SA3(x_train, y_train, i, Mu, Mu_old, phi)
            

        # save model
        alpha = Alpha(x_train, Mu, y_train, phi)
        tao = Tao(x_train, Mu, y_train, phi)
        # loss_ = Loss(x_val, Mu, y_val, alpha, tao)
        # loss = np.append(loss, loss_)
        np.save("model/RJSA/Loss{0}_{1}.npy".format(order, phi), loss)
        np.save("model/RJSA/Mu{0}_{1}.npy".format(order, phi), Mu)
        np.save("model/RJSA/Alpha{0}_{1}.npy".format(order, phi), alpha)
        np.save("model/RJSA/Tao{0}_{1}.npy".format(order, phi), tao)
        plt.figure()
        plt.plot(np.arange((val + 1) * iter), loss)
        if use_val == True:
            plt.savefig("model/RJSA/loss{0}_{1}_{2}.png".format(order, val, phi))
            print("[ %d ] [ kernel = %s ] [ val = %d ] [ Iteration %d ] [ time = %.4f ] [ k = %d ] [ Mu.row_num = %d ] [ loss = %.5f ]" % (order, phi, val + 1, i, t, k, Mu.shape[0], loss_))

            print("************ Training #{} is done ************".format(val + 1))
        else:
            plt.savefig("model/RJSA/loss{0}_{1}.png".format(order, phi))
            print("[ %d ] [ kernel = %s ] [ Iteration %d ] [ time = %.4f ] [ k = %d ] [ Mu.row_num = %d ] [ loss = %.5f ]" % (order, phi, i, t, k, Mu.shape[0], loss_))

    alpha = Alpha(x, Mu, y, phi)
    tao = Tao(x, Mu, y, phi)
    loss_ = Loss(x, Mu, y, alpha, tao, phi)
    print("total loss = %.5f, k = %d" % (loss_, k))

    print("************ Do testing ************")

    # test
    ytest = Predict(xtest, Mu, alpha, tao, c, phi)
    np.save("model/RJSA/v{0}_{1}.npy".format(order, phi), ytest)
    print("******** model-RJSA-v{} is saved ********".format(order))
