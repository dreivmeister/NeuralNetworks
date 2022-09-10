import numpy as np
from layers.layer import Layer
from optimizer.optimizer import SGD, Adam


class Dense(Layer):
    def __init__(self, input_size, output_size, optimizer='SGD', learning_rate=0.1):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        
        if optimizer == 'SGD':
            self.opt = SGD(learning_rate)
        elif optimizer == 'Adam':
            self.opt = Adam(learning_rate)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient):
        #Backprop
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        #Optimizer
        self.weights, self.bias = self.opt.update(self.weights, self.bias, weights_gradient, output_gradient)

        return input_gradient