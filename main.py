import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from si.neural_network import NeuralNetwork  # Importuj własną implementację sieci neuronowej

class NeuralNetworkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neural Network Configuration")
        self.geometry("600x400")

        self.create_widgets()

    def create_widgets(self):
        # Dane wejściowe
        self.create_input_frame()

        # Parametry sieci neuronowej
        self.create_network_frame()

        # Parametry uczenia
        self.create_training_frame()

        # Wyniki
        self.create_result_frame()

    def create_input_frame(self):
        frame = ttk.LabelFrame(self, text="Training Data")
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(frame, text="Input Data (X)").grid(row=0, column=0, padx=5, pady=5)
        self.input_x = tk.Entry(frame)
        self.input_x.grid(row=0, column=1, padx=5, pady=5)
        self.input_x.insert(0, "0,0; 0,1; 1,0; 1,1")

        ttk.Label(frame, text="Output Data (y)").grid(row=1, column=0, padx=5, pady=5)
        self.input_y = tk.Entry(frame)
        self.input_y.grid(row=1, column=1, padx=5, pady=5)
        self.input_y.insert(0, "0; 1; 1; 0")

    def create_network_frame(self):
        frame = ttk.LabelFrame(self, text="Network Configuration")
        frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(frame, text="Layers Configuration").grid(row=0, column=0, padx=5, pady=5)
        self.layers_config = tk.Entry(frame)
        self.layers_config.grid(row=0, column=1, padx=5, pady=5)
        self.layers_config.insert(0, "2,2,1")

    def create_training_frame(self):
        frame = ttk.LabelFrame(self, text="Training Parameters")
        frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(frame, text="Learning Rate").grid(row=0, column=0, padx=5, pady=5)
        self.learning_rate = tk.Entry(frame)
        self.learning_rate.grid(row=0, column=1, padx=5, pady=5)
        self.learning_rate.insert(0, "0.5")

        ttk.Label(frame, text="Max Iterations").grid(row=1, column=0, padx=5, pady=5)
        self.max_iterations = tk.Entry(frame)
        self.max_iterations.grid(row=1, column=1, padx=5, pady=5)
        self.max_iterations.insert(0, "10000")

        ttk.Label(frame, text="Max Error").grid(row=2, column=0, padx=5, pady=5)
        self.max_error = tk.Entry(frame)
        self.max_error.grid(row=2, column=1, padx=5, pady=5)
        self.max_error.insert(0, "0.01")

        ttk.Label(frame, text="Activation Function").grid(row=3, column=0, padx=5, pady=5)
        self.activation_function = ttk.Combobox(frame, values=["unipolar", "bipolar"])
        self.activation_function.grid(row=3, column=1, padx=5, pady=5)
        self.activation_function.set("unipolar")

        train_button = ttk.Button(frame, text="Train Network", command=self.train_network)
        train_button.grid(row=4, column=0, columnspan=2, pady=10)

    def create_result_frame(self):
        frame = ttk.LabelFrame(self, text="Results")
        frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.result_text = tk.Text(frame, height=10)
        self.result_text.grid(row=0, column=0, padx=5, pady=5)

    def train_network(self):
        # Pobierz dane z interfejsu
        X = np.array([list(map(int, xi.split(','))) for xi in self.input_x.get().split(';')])
        y = np.array([list(map(int, yi.split())) for yi in self.input_y.get().split(';')])

        # Pobierz konfigurację sieci
        layers_config = list(map(int, self.layers_config.get().split(',')))
        learning_rate = float(self.learning_rate.get())
        max_iterations = int(self.max_iterations.get())
        max_error = float(self.max_error.get())
        activation_function = self.activation_function.get()

        # Stwórz i trenuj sieć neuronową
        nn = NeuralNetwork(layers_config, learning_rate, activation_function, max_iterations, max_error)
        nn.train(X, y)

        # Wyświetl wyniki
        self.result_text.delete(1.0, tk.END)
        for xi in X:
            output = nn.forward(xi)
            self.result_text.insert(tk.END, f'Input: {xi}, Output: {output}\n')

if __name__ == "__main__":
    app = NeuralNetworkApp()
    app.mainloop()
