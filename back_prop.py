import encoding
import numpy as np
def back_prop(Z1, A1, Z2 , A2, W1, W2 , X , Y, row):
    one_hot_y = encoding.one_hot(Y)
    dZ2 = A2 - one_hot_y
    dW2 = 1/row *dZ2.dot(A1.T)
    db2 = 1/row * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * encoding.deriv_ReLu(Z1)
    dW1 = 1/row * dZ1.dot(X.T)
    db1 = 1/row * np.sum(dZ1)
    return dW1, db1, dW2, db2