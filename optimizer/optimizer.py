import numpy as np


class Optimizer:
    def __init__(self):
        pass
    
    def update(self):
        #probably receives weights and biases and returns the updated versions
        pass
    

class SGD(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, weights, biases, weights_gradient, output_gradient):
        weights -= self.learning_rate*weights_gradient
        biases -= self.learning_rate*output_gradient

        return weights, biases

class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.num_iter = 1

        self.m_weights_grad = self.v_weights_grad = 0
        self.m_bias_grad = self.v_bias_grad = 0

    def update(self, weights, biases, weights_gradient, output_gradient):
        #Look at other Adam Python implementations
        self.m_weights_grad = self.beta1*self.m_weights_grad + (1-self.beta1)*weights_gradient
        self.m_bias_grad = self.beta1*self.m_bias_grad + (1-self.beta1)*output_gradient

        self.v_weights_grad = self.beta2*self.v_weights_grad + (1-self.beta2)*(weights_gradient**2)
        self.v_bias_grad = self.beta2*self.v_bias_grad + (1-self.beta2)*(output_gradient)

        m_weights_grad_c = self.m_weights_grad/(1-self.beta1**self.num_iter)
        m_bias_grad_c = self.m_bias_grad/(1-self.beta1**self.num_iter)

        v_weights_grad_c = self.v_weights_grad/(1-self.beta2**self.num_iter)
        v_bias_grad_c = self.v_bias_grad/(1-self.beta2**self.num_iter)

        weights -= self.learning_rate*(m_weights_grad_c/(np.sqrt(v_weights_grad_c)+self.epsilon))
        biases -= self.learning_rate*(m_bias_grad_c/(np.sqrt(v_bias_grad_c)+self.epsilon))

        self.num_iter += 1

        return weights, biases