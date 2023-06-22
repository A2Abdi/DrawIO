import matplotlib.pyplot as plt
import forward_prop, grad_descent


def make_prediction(X, W1, b1, W2, b2 ):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = grad_descent.get_prediction(A2)
    return predictions
def test_predictions(index, W1, b1, W2, b2, x_train, y_train):
    current_image = x_train[:, index, None]
    predictions = make_prediction(x_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", predictions)
    print("Label: ", label)
    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation= 'nearest')
    plt.show()