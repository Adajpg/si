import numpy as np
from .layers import Layer
from .activation_functions import SigmoidUnipolar, SigmoidBipolar


class NeuralNetwork:
    def __init__(self, layers_config, learning_rate=0.1, activation='unipolar', max_iterations=10000, max_error=0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.max_error = max_error
        self.layers = []

        if activation == 'unipolar':
            activation_function = SigmoidUnipolar()
        else:
            activation_function = SigmoidBipolar()

        input_size = layers_config[0]
        for output_size in layers_config[1:]:
            self.layers.append(Layer(input_size, output_size, activation_function))
            input_size = output_size

    def train(self, X, y):
        for iteration in range(self.max_iterations):
            total_error = 0
            for xi, target in zip(X, y):
                output = self.forward(xi)
                total_error += self.calculate_total_error(output, target)
                self.backward(target)
            if total_error < self.max_error:
                print(f'Training stopped after {iteration} iterations with error: {total_error}')
                break

    def forward(self, x):
        input = x
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, target):
        error = target - self.layers[-1].output
        for i in reversed(range(len(self.layers))):
            error = self.layers[i].backward(error, self.learning_rate)

    def calculate_total_error(self, output, target):
        return 0.5 * np.sum((target - output) ** 2)
