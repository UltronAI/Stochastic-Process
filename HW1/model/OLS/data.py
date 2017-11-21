import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

loss_val = np.load("valLoss2_Gauss.npy")[20:]
loss_train = np.load("trainLoss2_Gauss.npy")[20:]

"""
L = len(loss_val)

plt.figure()
plt.plot(range(L), loss_val, "r", label = "loss_val")
plt.plot(range(L), loss_train, "b", label = "loss_train")
plt.legend()
plt.savefig("newloss2.png")
"""

v1 = np.load("v1_Gauss.npy")
v2 = np.load("v2_Gauss.npy")

sio.savemat("OLS2015011208.mat", {"v1": v1, "v2": v2})