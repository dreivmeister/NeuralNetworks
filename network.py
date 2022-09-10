import pickle
import numpy as np


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

def evaluate(network, x_test, y_test):
    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))


def save_network(network, filename):
    pickle.dump(network, open(filename, "wb"))

def load_network(filename):
    return pickle.load(open(filename, "rb"))
