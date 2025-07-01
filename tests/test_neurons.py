import unittest
import numpy as np
from neural_network.neurons import Neuron
from neural_network.activation_functions import SigmoidUnipolar


class TestNeuron(unittest.TestCase):

    def test_activate_output_range(self):
        neuron = Neuron(3, SigmoidUnipolar())
        x = np.array([0.0, 0.0, 0.0])
        output = neuron.activate(x)
        self.assertGreaterEqual(output, 0.0)
        self.assertLessEqual(output, 1.0)

    def test_neuron_ideal_output(self):
        neuron = Neuron(1, SigmoidUnipolar())
        neuron.weights = np.array([0])
        neuron.bias = 0
        output = neuron.activate(np.array([0]))
        self.assertAlmostEqual(output, 0.5)
