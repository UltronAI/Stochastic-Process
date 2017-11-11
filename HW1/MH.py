import numpy as np
import numpy.linalg as la

import warnings
warnings.filterwarnings("ignore")

gauss = lambda x, mu, sigma: 1/(2*np.pi)/la.det(sigma)**0.5*np.exp(-1/2*(x-mu).T.dot(la.inv(sigma)).dot(x-mu))

x_mu = np.array([[5], [10]])
x_sigma = np.array([[1, -1], [-1, 4]])

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

    X = np.array([[],[]])

    for i in range(iter):
        u = np.random.rand()
        x_ = np.random.multivariate_normal(x.reshape((2, )), x_sigma).reshape((2, 1))
        try:
            a = alpha(x_, x)
        except RuntimeWarning:
            i -= 1
            continue
        else:
            pass                            

        if u < a:
            x = x_
            X = np.hstack((X, x))

    cov = np.cov(X)
    rho = cov[0, 1]/cov[0, 0]**0.5/cov[1, 1]**0.5

    print("correlation coefficient:", rho)

# iter = 5e5: -0.500496736343
# iter = 5e4: -0.49755816426
# iter = 5e3: -0.47690968431
# iter = 5e2: -0.608209759235