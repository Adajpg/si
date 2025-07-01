import numpy as np


class Neuron:
    def __init__(self, input_size, activation_function):
        # Initialize neuron with random weights and bias.
        self.weights = np.random.uniform(-1, 1, size=input_size)
        self.bias = np.random.uniform(-1, 1)
        self.activation_function = activation_function  # Activation function for the neuron.

    def activate(self, inputs):
        # Compute the weighted sum of inputs plus bias.
        self.input_sum = np.dot(inputs, self.weights) + self.bias
        # Activate the neuron using the specified activation function.
        return self.activation_function(self.input_sum)
