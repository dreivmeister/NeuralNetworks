import numpy as np
from layers.layer import Layer



class MaxPooling1D(Layer):
    def __init__(self, pool_size, stride, padding = 'valid'):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        #experimental
        self.max_idxs = []

    def forward(self, input):
        self.input = input
        output_size = np.floor((len(self.input)-self.pool_size)/self.stride+1)
        self.output = np.zeros(output_size)

        for i in range(0, output_size, step=self.stride):
            max_idx = np.argmax(self.input[i:self.pool_size])
            self.max_idxs.append(max_idx)
            self.output[i] = self.input[max_idx]

        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros(len(self.input))

        for i, idx in enumerate(self.max_idxs):
            input_gradient[idx] = output_gradient[i]
        
        return input_gradient


class MaxPooling2D(Layer):
    def __init__(self, input_shape, pool_size, stride, padding = 'valid'):
        #(num_channels, input_height, input_width)
        self.input_shape = input_shape
        #(pool_height, pool_width)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, input):
        self.input = input
        self.output_height = np.floor((self.input_shape[1]-self.pool_size[0])/self.stride+1) 
        self.output_width = np.floor((self.input_shape[2]-self.pool_size[1])/self.stride+1) 
        self.output = np.zeros((self.input_shape[0], self.output_height, self.output_width))

        for c in range(self.input_shape[0]):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output[c, i, j] = np.max(self.input[c, i * self.stride : i * self.stride + self.pool_size[0], j * self.stride : j * self.stride + self.pool_size[1]])
        
        return self.output
    
    def backward(self, output_gradient):
        input_gradient = np.zeros_like(self.input)


        for c in range(self.input_shape[0]):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    i_t, j_t = np.where(np.max(self.input[c, i * self.stride : i * self.stride + self.pool_size[0], j * self.stride : j * self.stride + self.pool_size[1]]) == self.input[c, i * self.stride : i * self.stride + self.pool_size[0], j * self.stride : j * self.stride + self.pool_size[1]])
                    i_t, j_t = i_t[0], j_t[0]
                    input_gradient[c, i * self.stride : i * self.stride + self.pool_size[0], j * self.stride : j * self.stride + self.pool_size[1]][i_t, j_t] = output_gradient[c, i, j]

        return input_gradient