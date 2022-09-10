from optimizer.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, weights, biases, weights_gradient, output_gradient):
        weights -= self.learning_rate*weights_gradient
        biases -= self.learning_rate*output_gradient

        return weights, biases