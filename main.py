import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from neural_network import NeuralNetwork  # Import your own neural network implementation
from neural_network.loaders.rses_loader import RSESLoader  # Import the RSESLoader


class NeuralNetworkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neural Network Configuration")
        self.geometry("800x600")  # Adjust the geometry to better fit the display

        self.create_widgets()

    def create_widgets(self):
        # Input Data
        self.create_input_frame()

        # Network Parameters
        self.create_network_frame()

        # Training Parameters
        self.create_training_frame()

        # Error Plot
        self.create_error_plot_frame()

        # Results
        self.create_result_frame()

    def create_input_frame(self):
        frame = ttk.LabelFrame(self, text="Training Data")
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        load_button = ttk.Button(frame, text="Load Data", command=self.load_data)
        load_button.grid(row=0, column=0, padx=5, pady=5)

        ttk.Label(frame, text="Input Data (X)").grid(row=1, column=0, padx=5, pady=5)
        self.input_x = tk.Entry(frame)
        self.input_x.grid(row=1, column=1, padx=5, pady=5)
        self.input_x.insert(0, "0,0; 0,1; 1,0; 1,1")

        ttk.Label(frame, text="Output Data (y)").grid(row=2, column=0, padx=5, pady=5)
        self.input_y = tk.Entry(frame)
        self.input_y.grid(row=2, column=1, padx=5, pady=5)
        self.input_y.insert(0, "0; 1; 1; 0")

    def create_network_frame(self):
        frame = ttk.LabelFrame(self, text="Network Configuration")
        frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(frame, text="Layers Configuration").grid(row=0, column=0, padx=5, pady=5)
        self.layers_config = tk.Entry(frame)
        self.layers_config.grid(row=0, column=1, padx=5, pady=5)

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

    def create_error_plot_frame(self):
        frame = ttk.LabelFrame(self, text="Error Plot")
        frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")

        self.error_plot = Figure(figsize=(5, 4), dpi=100)
        self.error_plot_ax = self.error_plot.add_subplot(111)
        self.error_plot_canvas = FigureCanvasTkAgg(self.error_plot, master=frame)
        self.error_plot_canvas.draw()
        self.error_plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def create_result_frame(self):
        frame = ttk.LabelFrame(self, text="Results")
        frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.result_text = tk.Text(frame, height=10)
        self.result_text.grid(row=0, column=0, padx=5, pady=5)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("RSES Files", "*.tab"), ("All Files", "*.*")])
        if file_path:
            X, y = RSESLoader.load(file_path)
            if X is not None and y is not None:
                self.input_x.delete(0, tk.END)
                self.input_y.delete(0, tk.END)
                self.input_x.insert(0, "; ".join([", ".join(map(str, map(int, xi))) for xi in X]))
                self.input_y.insert(0, "; ".join(map(str, map(int, y))))

                # Automatically set layers configuration based on input dimension
                input_dim = X.shape[1]
                output_dim = 1 if len(y.shape) == 1 else y.shape[1]
                self.layers_config.delete(0, tk.END)
                self.layers_config.insert(0, f"{input_dim},{input_dim + 2},{output_dim}")

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

        errors = []
        for iteration in range(max_iterations):
            total_error = 0
            for xi, target in zip(X, y):
                output = nn.forward(xi)
                total_error += nn.calculate_total_error(output, target)
                nn.backward(target)
            errors.append(total_error)
            if total_error < max_error:
                print(f'Training stopped after {iteration} iterations with error: {total_error}')
                break

        # Aktualizuj wykres błędów
        self.error_plot_ax.clear()
        self.error_plot_ax.plot(errors)
        self.error_plot_ax.set_xlabel('Iterations')
        self.error_plot_ax.set_ylabel('Error')
        self.error_plot_canvas.draw()

        # Wyświetl wyniki
        self.result_text.delete(1.0, tk.END)
        for xi in X:
            output = nn.forward(xi)
            self.result_text.insert(tk.END, f'Input: {xi}, Output: {output}\n')


if __name__ == "__main__":
    app = NeuralNetworkApp()
    app.mainloop()
