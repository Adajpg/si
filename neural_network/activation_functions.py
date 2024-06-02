import numpy as np


class ActivationFunction:
    def __call__(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class SigmoidUnipolar(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        fx = self.__call__(x)
        return fx * (1 - fx)


class SigmoidBipolar(ActivationFunction):
    def __call__(self, x):
        return (1 - np.exp(-x)) / (1 + np.exp(-x))

    def derivative(self, x):
        fx = self.__call__(x)
        return 1 - fx ** 2
