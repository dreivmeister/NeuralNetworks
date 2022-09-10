from activations.activation import Activation
import numpy as np



class ReLU(Activation):
    def __init__(self):

        def relu(x):
            return max(0,x)
        
        def relu_prime(x):
            if x > 0:
                return 1
            return 0
        
        super().__init__(relu, relu_prime)