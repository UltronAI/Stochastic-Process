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

order = 1

print(">> Loading data from {} ...".format("data/data{}.mat".format(order)))
data = sio.loadmat("data/data{}.mat".format(order))

x = data["x"]
y = data["y"]
xtest = data["xtest"]

print("x:", x.shape)
print("y:", y.shape)
print("xtest:", xtest.shape)
