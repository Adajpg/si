import numpy as np


def initialize_weights(input_size, output_size):
    return np.random.rand(input_size, output_size)


def initialize_bias(output_size):
    return np.random.rand(output_size)
