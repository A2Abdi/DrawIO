import begin, forward_prop, back_prop
import numpy as np

def gradient_descent(X,Y , iterations, alpha, row):
    W1, b1 , W2 , b2 = begin._init_params()
    for i in range(iterations):
        Z1, A1, Z2 , A2 = forward_prop.forward_prop(W1,b1,W2,b2,X)
        dW1, db1, dW2, db2 = back_prop.back_prop(Z1, A1, Z2 , A2, W1, W2, X, Y, row)
        W1, b1 , W2 , b2 = update_param(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration : ", i)
            print("Accuracy : ", get_Accuracy(get_prediction(A2),Y))
        
    return W1, b1 , W2 , b2

def update_param(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1 , W2 , b2


def get_prediction(A2):
    return np.argmax(A2,0)

def get_Accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y)/ y.size