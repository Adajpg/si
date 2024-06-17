import numpy as np


class ActivationFunction:
    def __call__(self, x):
        # Placeholder for activation function calculation.
        raise NotImplementedError

    def derivative(self, x):
        # Placeholder for derivative calculation.
        raise NotImplementedError


class SigmoidUnipolar(ActivationFunction):
    def __call__(self, x):
        # Compute the unipolar sigmoid activation function.
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        # Compute the derivative of the unipolar sigmoid activation function.
        fx = self.__call__(x)
        return fx * (1 - fx)


class SigmoidBipolar(ActivationFunction):
    def __call__(self, x):
        # Compute the bipolar sigmoid activation function.
        return (1 - np.exp(-x)) / (1 + np.exp(-x))

    def derivative(self, x):
        # Compute the derivative of the bipolar sigmoid activation function.
        fx = self.__call__(x)
        return 1 - fx ** 2
