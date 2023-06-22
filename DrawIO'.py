import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import grad_descent, predictions

data = pd.read_csv('train.csv')
data = np.array(data)

row , col = data.shape
np.random.shuffle(data)

data_train = data[1000:row].T
y_train = data_train[0]
x_train = data_train[1:col]
x_train = x_train/ 255


data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:col]
x_dev = x_dev / 255

W1, b1 , W2 , b2 = grad_descent.gradient_descent(x_train, y_train , 1000, 0.25, row)