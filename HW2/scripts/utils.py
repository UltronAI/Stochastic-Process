import numpy as np

def mse(y_pred, y):
    return np.mean((y-y_pred)**2)
