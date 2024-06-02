import numpy as np
from .neurons import Neuron

class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.neurons = [Neuron(input_size, activation_function) for _ in range(output_size)]
        self.output = np.zeros(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.array([neuron.activate(inputs) for neuron in self.neurons])
        return self.output

    def backward(self, error, learning_rate):
        input_error = np.zeros(len(self.inputs))
        for i, neuron in enumerate(self.neurons):
            delta = error[i] * neuron.activation_function.derivative(neuron.input_sum)
            for j in range(len(self.inputs)):
                input_error[j] += delta * neuron.weights[j]
                neuron.weights[j] += learning_rate * delta * self.inputs[j]
            neuron.bias += learning_rate * delta
        return input_error
