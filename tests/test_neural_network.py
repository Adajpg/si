import unittest
import numpy as np
from neural_network.neural_network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):

    def test_forward_output_shape(self):
        nn = NeuralNetwork([3, 4, 2], activation='unipolar')
        x = np.array([0.5, 0.5, 0.5])
        output = nn.forward(x)
        self.assertEqual(output.shape, (2,))

    def test_training_error_decreases(self):
        X = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([[0], [1], [1], [0]])  # XOR
        nn = NeuralNetwork([2, 3, 1], learning_rate=0.5, max_iterations=2000, max_error=0.01)
        errors = nn.train(X, y)
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[-1], 0.1)

    def test_total_error_calculation(self):
        nn = NeuralNetwork([2, 2, 1])
        output = np.array([0.5])
        target = np.array([1.0])
        expected = 0.5 * (1.0 - 0.5)**2
        actual = nn.calculate_total_error(output, target)
        self.assertAlmostEqual(actual, expected)

    def test_early_stopping(self):
        X = np.array([[0], [1]])
        y = np.array([[0], [1]])
        nn = NeuralNetwork([1, 2, 1], learning_rate=0.5, max_iterations=1000, max_error=0.9)
        errors = nn.train(X, y)
        self.assertLess(len(errors), 1000)
