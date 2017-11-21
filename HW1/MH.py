import numpy as np
import numpy.linalg as la
import warnings
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

gauss = lambda x, mu, sigma: 1/(2*np.pi)/la.det(sigma)**0.5*np.exp(-1/2*(x-mu).T.dot(la.inv(sigma)).dot(x-mu))

n_ignore = 300
x_mu = np.array([[5], [10]])
lll = 1
x_sigma = np.array([[1, -1], [-1, 4]])
# lll = 3
# x_sigma = np.array([[1, -lll], [-lll, 4]])

def Q(x_, x):
    mu = x
    sigma = x_sigma
    return gauss(x_, mu, sigma)

def p(x):
    mu = x_mu
    sigma = x_sigma
    return gauss(x, mu, sigma)

alpha = lambda x_, x: min(p(x_)*Q(x, x_)/(p(x)*Q(x_, x)), 1)

if __name__ == "__main__":
    x = np.random.multivariate_normal(x_mu.reshape((2, )), x_sigma).reshape((2, 1))
    iter = 50000
    loss = np.array([])
    X = np.array([])
    RHO = -0.5
    Rho = np.array([])

    X = np.array([[],[]])

    for i in range(iter):
        u = np.random.rand()
        x_ = np.random.multivariate_normal(x.reshape((2, )), 4 * x_sigma).reshape((2, 1))
        #"""
        try:
            a = alpha(x_, x)
        except RuntimeWarning:
            i -= 1
            continue
        else:
            pass   
        #"""
        #a = alpha(x_, x)                         

        if u < a:
            x = x_
            X = np.hstack((X, x))

        if X.shape[0] == 0 or X.shape[1] <= n_ignore:
            continue
        cov = np.cov(X[:, n_ignore:])
        rho = cov[0, 1]/cov[0, 0]**0.5/cov[1, 1]**0.5
        Rho = np.append(Rho, rho)
        loss_ = abs(rho-RHO)/abs(RHO)
        loss = np.append(loss, loss_)
        if Rho.shape[0] >= 10 and loss[-5:].mean() <= 1e-4:
            break
        if np.mod(i, 100) == 0:
            if len(loss) <= 20:
                continue
            print("Iteration %d, rho = %.4f, loss = %.4f" % (i, rho, loss_))
            plt.close()
            plt.figure()
            plt.plot(range(len(loss[n_ignore:])), loss[n_ignore:])
            plt.savefig("model/MH/loss_mh_{}.png".format(lll))
            plt.close()
            plt.figure()
            plt.plot(range(len(Rho[n_ignore:])), Rho[n_ignore:])
            plt.savefig("model/MH/Rho_{}.png".format(lll))

    print("Iteration %d, rho = %.4f, loss = %.4f" % (i, rho, loss_))
    plt.close()
    plt.figure()
    plt.plot(range(len(loss[n_ignore:])), loss[n_ignore:])
    plt.savefig("model/MH/loss_mh_{}.png".format(lll))
    plt.close()
    plt.figure()
    plt.plot(range(len(Rho[n_ignore:])), Rho[n_ignore:])
    plt.savefig("model/MH/Rho_{}.png".format(lll))

# iter = 5e5: -0.500496736343
# iter = 5e4: -0.49755816426
# iter = 5e3: -0.47690968431
# iter = 5e2: -0.608209759235
