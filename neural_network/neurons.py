import numpy as np


class Neuron:
    def __init__(self, input_size, activation_function):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.activation_function = activation_function

    def activate(self, inputs):
        self.input_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(self.input_sum)
