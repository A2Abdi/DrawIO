import numpy as np
def one_hot(Y):
    one_hot_y = np.zeros((Y.size, Y.max() + 1))
    one_hot_y[np.arange(Y.size), Y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def ReLu(Z):
    return np.maximum(0,Z)
def deriv_ReLu(Z):
    return Z > 0
def softmax(Z):
    return np.exp(Z)/sum(np.exp(Z))