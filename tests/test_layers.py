import unittest
import numpy as np
from neural_network.layers import Layer
from neural_network.activation_functions import SigmoidUnipolar


class TestLayer(unittest.TestCase):

    def test_forward_output_shape(self):
        layer = Layer(4, 2, SigmoidUnipolar())
        input_data = np.array([0.1, 0.2, 0.3, 0.4])
        output = layer.forward(input_data)
        self.assertEqual(output.shape, (2,))

    def test_backward_updates_weights(self):
        layer = Layer(2, 1, SigmoidUnipolar())
        inputs = np.array([1.0, 1.0])
        layer.forward(inputs)
        original_weights = layer.neurons[0].weights.copy()

        error = np.array([0.5])
        layer.backward(error, learning_rate=0.1)

        updated_weights = layer.neurons[0].weights
        self.assertFalse(np.allclose(original_weights, updated_weights))
