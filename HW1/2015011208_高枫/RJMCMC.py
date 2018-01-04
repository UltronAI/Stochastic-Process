import scipy.io as sio
import numpy as np
import random
from utils_RJMCMC import *
import warnings, time, os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# British coal mine disaster data by year (1851-1962)
x = np.array([1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961])
y = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

plt.figure()
plt.scatter(x, y, c = "r")
# plt.savefig('coal_data.png')

L = len(x) # L = 111
k = 100 # number of steps
k_max = 200

h = np.array([random.uniform(0, 20) for i in range(k)])
s = np.array([x[0]] + sorted([random.uniform(min(x), max(x)) for i in range(k - 1)]) + [x[-1]])

if k != h.shape[0]:
    raise ValueError("H_error : k = {0}, while h_len = {1}".format(k, h.shape[0]))
elif k != s.shape[0] - 1:
    raise ValueError("S_error : k = {0}, while s_len = {1}".format(k, s.shape[0]))
else:
    pass

#h, s, k = Birth(h, s, k, y)

loss = []
iter = 5000
for i in range(iter):
    loss_ = Loss(h, s, y)
    loss.append(loss_)
    if k != h.shape[0]:
        raise ValueError("H_error : k = {0}, while h_len = {1}".format(k, h.shape[0]))
    elif k != s.shape[0] - 1:
        raise ValueError("S_error : k = {0}, while s_len = {1}".format(k, s.shape[0]))
    else:
        pass
    if np.mod(i, 100) == 0:
        plt.figure()
        plt.plot(range(i + 1), loss)
        plt.savefig("model/RJMCMC/loss.png")
        print("[ iteration %d ] [ k = %d ] [ loss = %.5f ]" % (i, k, loss_))
    u = np.random.rand()
    if u <= b(k):
        # print('birth')
        h, s, k = Birth(h, s, k, y)
    elif u <= b(k) + d(k):
        # print('death')
        h, s, k = Death(h, s, k, y)
    elif u <= b(k) + d(k) + pi(k):
        # print('position')
        h, s, k = SMove(h, s, k, y)
    else:
        # print('height')
        h, s, k = HMove(h, s, k, y)

np.save("model/RJMCMC/loss.npy", loss)
np.save("model/RJMCMC/h.npy", h)
np.save("model/RJMCMC/s.npy", s)
