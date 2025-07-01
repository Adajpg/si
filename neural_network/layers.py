import numpy as np
from .neurons import Neuron  # Importing the Neuron class from neurons module


class Layer:
    def __init__(self, input_size, output_size, activation_function):
        # Initialize the layer with neurons and output array
        self.neurons = [Neuron(input_size, activation_function) for _ in range(output_size)]
        self.output = np.empty(output_size)

    def forward(self, inputs):
        # Forward pass through the layer
        self.inputs = inputs  # Store inputs for later use in backward pass
        # Activate each neuron in the layer and store the outputs
        self.output = np.array([neuron.activate(inputs) for neuron in self.neurons])
        return self.output  # Return the output of the layer

    def backward(self, error, learning_rate):
        input_error = np.zeros(len(self.inputs))  # Initialize input error array
        # Backward pass through the layer to update weights and biases
        for i, neuron in enumerate(self.neurons):
            # Calculate delta for the neuron based on the error and derivative of activation function
            delta = error[i] * neuron.activation_function.derivative(neuron.input_sum)
            # Update weights and calculate input error for previous layer
            for j in range(len(self.inputs)):
                input_error[j] += delta * neuron.weights[j]
                neuron.weights[j] += learning_rate * delta * self.inputs[j]
            neuron.bias += learning_rate * delta  # Update bias for the neuron
        return input_error  # Return the input error for the previous layer
