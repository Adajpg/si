from si.neural_network import NeuralNetwork
import numpy as np

# Dane treningowe
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

# Konfiguracja sieci neuronowej
layers_config = [2, 2, 1]  # 2 neurony wejściowe, 1 warstwa ukryta z 2 neuronami, 1 neuron wyjściowy
learning_rate = 0.5
max_iterations = 10000
max_error = 0.01

# Tworzenie i trenowanie sieci
nn = NeuralNetwork(layers_config, learning_rate, 'unipolar', max_iterations, max_error)
nn.train(X, y)

# Testowanie sieci
for xi in X:
    print(f'Input: {xi}, Output: {nn.forward(xi)}')
