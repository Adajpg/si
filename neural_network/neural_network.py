import numpy as np
from .layers import Layer  # Import Layer class from layers module
from .activation_functions import SigmoidUnipolar, SigmoidBipolar  # Import activation functions


class NeuralNetwork:
    def __init__(self, layers_config, learning_rate=0.1, activation='unipolar', max_iterations=10000, max_error=0.01):
        """
        Initialize a neural network with specified configurations.

        Parameters:
        - layers_config (list): List of integers specifying the number of neurons in each layer.
        - learning_rate (float): Learning rate for gradient descent (default: 0.1).
        - activation (str): Type of activation function ('unipolar' or 'bipolar', default: 'unipolar').
        - max_iterations (int): Maximum number of training iterations (default: 10000).
        - max_error (float): Maximum error threshold for training termination (default: 0.01).
        """
        self.learning_rate = learning_rate  # Initialize learning rate
        self.max_iterations = max_iterations  # Initialize maximum iterations for training
        self.max_error = max_error  # Initialize maximum error threshold
        self.layers = []  # Initialize empty list for layers

        if activation == 'unipolar':
            activation_function = SigmoidUnipolar()  # Use unipolar sigmoid activation function
        else:
            activation_function = SigmoidBipolar()  # Use bipolar sigmoid activation function

        input_size = layers_config[0]  # Number of inputs is specified in the first element of layers_config
        # Create layers based on layers_config
        for output_size in layers_config[1:]:
            self.layers.append(Layer(input_size, output_size, activation_function))
            input_size = output_size  # Update input_size for the next layer

    def train(self, X, y):
        """
        Train the neural network using backpropagation.

        Parameters:
        - X (np.ndarray): Input data of shape (n_samples, n_features).
        - y (np.ndarray): Target values of shape (n_samples,).

        """
        errors = []
        for iteration in range(self.max_iterations):
            total_error = 0
            for xi, target in zip(X, y):
                output = self.forward(xi)  # Perform forward pass
                total_error += self.calculate_total_error(output, target)  # Calculate total error
                self.backward(target)  # Perform backward pass
            errors.append(total_error)
            if total_error < self.max_error:
                print(f'Training stopped after {iteration} iterations with error: {total_error}')
                break
        return errors

    def forward(self, x):
        """
        Perform forward propagation through the neural network.

        Parameters:
        - x (np.ndarray): Input data of shape (n_features,).

        Returns:
        - np.ndarray: Output of the neural network.
        """
        input = x
        for layer in self.layers:
            input = layer.forward(input)  # Perform forward pass through each layer
        return input  # Return final output of the neural network

    def backward(self, target):
        """
        Perform backward propagation (backpropagation) through the neural network.

        Parameters:
        - target (np.ndarray): Target output for the current input.

        """
        error = target - self.layers[-1].output  # Compute error for output layer
        for i in reversed(range(len(self.layers))):
            error = self.layers[i].backward(error, self.learning_rate)  # Perform backward pass through each layer

    def calculate_total_error(self, output, target):
        """
        Calculate the total error (loss) between predicted output and target.

        Parameters:
        - output (np.ndarray): Predicted output of the neural network.
        - target (np.ndarray): Target output for the current input.

        Returns:
        - float: Total error between output and target.
        """
        return 0.5 * np.sum((target - output) ** 2)  # Calculate total error using mean squared error
