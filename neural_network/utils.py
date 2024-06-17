import numpy as np


def initialize_weights(input_size, output_size):
    """
    Initialize weights for a neural network layer.

    Parameters:
    - input_size (int): Number of input units.
    - output_size (int): Number of output units.

    Returns:
    - np.ndarray: Initialized weights with shape (input_size, output_size).
    """
    return np.random.rand(input_size, output_size)  # Initialize weights with random values


def initialize_bias(output_size):
    """
    Initialize bias for a neural network layer.

    Parameters:
    - output_size (int): Number of output units.

    Returns:
    - np.ndarray: Initialized bias with shape (output_size,).
    """
    return np.random.rand(output_size)  # Initialize bias with random values
