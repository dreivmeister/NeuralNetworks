import numpy as np
from layers.layer import Layer
from activations.Sigmoid import Sigmoid

class Recurrent(Layer):
    def __init__(self, num_steps, hidden_dim, output_dim, learning_rate):
        self.num_steps = num_steps
        self.bptt_truncate = 5
        self.min_clip_value = -10
        self.max_clip_value = 10
        self.learning_rate = learning_rate

        self.U = np.random.uniform(0, 1, (hidden_dim, num_steps))
        self.W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
        self.V = np.random.uniform(0, 1, (output_dim, hidden_dim))

        self.dU = np.zeros(self.U.shape)
        self.dV = np.zeros(self.V.shape)
        self.dW = np.zeros(self.W.shape)
        
        self.dU_t = np.zeros(self.U.shape)
        self.dV_t = np.zeros(self.V.shape)
        self.dW_t = np.zeros(self.W.shape)
        
        self.dU_i = np.zeros(self.U.shape)
        self.dW_i = np.zeros(self.W.shape)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, y, prev_s):
        layers = []
        for t in range(self.num_steps):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mul_u = np.dot(self.U, new_input)
            mul_w = np.dot(self.W, prev_s)
            add = mul_u + mul_w
            s = self.sigmoid(add)
            mul_v = np.dot(self.V, s)
            layers.append({'s':s, 'prev_s':prev_s})
            prev_s = s
        return layers, (mul_v - y), add, mul_w, mul_u


    def backward(self, x, layers, dmul_v, add, mul_w, mul_u):
        for t in range(self.num_steps):
            dV_t = np.dot(dmul_v, layers[t]['s'].T)
            dsv = np.dot(self.V.T, dmul_v)

            ds = dsv
            d_add = add * (1 - add) * ds

            dmul_w = d_add * np.ones_like(mul_w)

            dprev_s = np.dot(self.W.T, dmul_w)

            for i in range(t-1, max(-1, t - self.bptt_truncate-1), -1):
                ds = dsv + dprev_s
                d_add = add * (1 - add) * ds

                dmul_w = d_add * np.ones_like(mul_w)
                dmul_u = d_add * np.ones_like(mul_u)

                dW_i = np.dot(self.W, layers[t]['prev_s'])
                dprev_s = np.dot(self.W.T, dmul_w)

                new_input = np.zeros(x.shape)
                new_input[t] = x[t]
                dU_i = np.dot(self.U, new_input)
                dx = np.dot(self.U.T, dmul_u)

                self.dU_t += dU_i
                self.dW_t += dW_i
            
            self.dV += dV_t
            self.dU += self.dU_t
            self.dW = self.dW_t


            if self.dU.max() > self.max_clip_value:
                self.dU[self.dU > self.max_clip_value] = self.max_clip_value
            if self.dV.max() > self.max_clip_value:
                self.dV[self.dV > self.max_clip_value] = self.max_clip_value
            if self.dW.max() > self.max_clip_value:
                self.dW[self.dW > self.max_clip_value] = self.max_clip_value
            
            if self.dU.min() < self.min_clip_value:
                self.dU[self.dU < self.min_clip_value] = self.min_clip_value
            if self.dV.min() < self.min_clip_value:
                self.dV[self.dV < self.min_clip_value] = self.min_clip_value
            if self.dW.min() < self.min_clip_value:
                self.dW[self.dW < self.min_clip_value] = self.min_clip_value
                # update
        self.U -= self.learning_rate * self.dU
        self.V -= self.learning_rate * self.dV
        self.W -= self.learning_rate * self.dW