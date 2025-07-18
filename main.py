import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from neural_network import NeuralNetwork
from neural_network.loaders.rses_loader import RSESLoader


class NeuralNetworkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neural Network Configuration")
        self.geometry("800x600")  # Set initial window size

        self.create_widgets()  # Create all GUI components

    def create_widgets(self):
        # Input Data Frame
        self.create_input_frame()

        # Network Parameters Frame
        self.create_network_frame()

        # Training Parameters Frame
        self.create_training_frame()

        # Error Plot Frame
        self.create_error_plot_frame()

        # Results Frame
        self.create_result_frame()

    def create_input_frame(self):
        """
        Create the frame for loading and displaying input data.
        """
        frame = ttk.LabelFrame(self, text="Training Data")
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        load_button = ttk.Button(frame, text="Load Data", command=self.load_data)
        load_button.grid(row=0, column=0, padx=5, pady=5)

        ttk.Label(frame, text="Input Data (X)").grid(row=1, column=0, padx=5, pady=5)
        self.input_x = tk.Entry(frame)
        self.input_x.grid(row=1, column=1, padx=5, pady=5)
        self.input_x.insert(0, "0.0,0.0,0.0; 0.0,0.0,1.0; 0.0,1.0,0.0; 0.0,1.0,1.0; 1.0,0.0,0.0; 1.0,0.0,1.0; 1.0,"
                               "1.0,0.0; 1.0,1.0,1.0")

        ttk.Label(frame, text="Output Data (y)").grid(row=2, column=0, padx=5, pady=5)
        self.input_y = tk.Entry(frame)
        self.input_y.grid(row=2, column=1, padx=5, pady=5)
        self.input_y.insert(0, "0.0; 0.0; 0.0; 1.0; 0.0; 1.0; 1.0; 0.0")

    def create_network_frame(self):
        """
        Create the frame for configuring the neural network layers.
        """
        frame = ttk.LabelFrame(self, text="Network Configuration")
        frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(frame, text="Layers Configuration").grid(row=0, column=0, padx=5, pady=5)
        self.layers_config = tk.Entry(frame)
        self.layers_config.grid(row=0, column=1, padx=5, pady=5)

    def create_training_frame(self):
        """
        Create the frame for setting training parameters.
        """
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
        """
        Create the frame for displaying the training error plot.
        """
        frame = ttk.LabelFrame(self, text="Error Plot")
        frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")

        self.error_plot = Figure(figsize=(5, 4), dpi=100)
        self.error_plot_ax = self.error_plot.add_subplot(111)
        self.error_plot_canvas = FigureCanvasTkAgg(self.error_plot, master=frame)
        self.error_plot_canvas.draw()
        self.error_plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def create_result_frame(self):
        """
        Create the frame for displaying the results of the neural network.
        """
        frame = ttk.LabelFrame(self, text="Results")
        frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.result_text = tk.Text(frame, height=10)
        self.result_text.grid(row=0, column=0, padx=5, pady=5)

    def load_data(self):
        """
        Load data from a selected file using the RSESLoader.
        """
        file_path = filedialog.askopenfilename(filetypes=[("RSES files", "*.tab"), ("All files", "*.*")])
        if file_path:
            X, y = RSESLoader.load(file_path)
            if X is not None and y is not None:
                # Update input fields with loaded data
                self.input_x.delete(0, tk.END)
                self.input_y.delete(0, tk.END)
                self.input_x.insert(0, '; '.join(','.join(map(str, xi)) for xi in X))
                self.input_y.insert(0, '; '.join(map(str, y)))

                # Update layers configuration based on loaded data dimensions
                input_size = X.shape[1]
                output_size = 1 if y.ndim == 1 else y.shape[1]
                layers_config = [input_size] + [int(neurons) for neurons in self.layers_config.get().split(',')[1:-1]] + [output_size]
                self.layers_config.delete(0, tk.END)
                self.layers_config.insert(0, ','.join(map(str, layers_config)))

    def train_network(self):
        """
        Train the neural network using parameters specified in the GUI.
        """
        try:
            # Get data from GUI inputs
            X = np.array([list(map(float, xi.split(','))) for xi in self.input_x.get().split(';')])
            y = np.array([[float(yi)] for yi in self.input_y.get().split(';')])

            if len(X) != len(y):
                messagebox.showerror("Data Mismatch", "Number of input samples and output targets must match.")
                return
            
            # Get network configuration
            try:
                layers_config = list(map(int, self.layers_config.get().split(',')))
            except ValueError:
                messagebox.showerror("Invalid input",
                                     "Layers configuration must be a comma-separated list of integers.")
                return
            learning_rate = float(self.learning_rate.get())
            max_iterations = int(self.max_iterations.get())
            max_error = float(self.max_error.get())
            activation_function = self.activation_function.get()

            # Create and train the neural network
            nn = NeuralNetwork(layers_config, learning_rate, activation_function, max_iterations, max_error)

            errors = nn.train(X, y)

            # Display results
            self.result_text.delete(1.0, tk.END)
            for xi in X:
                output = nn.forward(xi)
                self.result_text.insert(tk.END, f'Input: {xi}, Output: {output}\n')

            # Update error plot
            self.error_plot_ax.clear()
            self.error_plot_ax.plot(errors)
            self.error_plot_ax.set_title("Training Error")
            self.error_plot_ax.set_xlabel("Iteration")
            self.error_plot_ax.set_ylabel("Total Error")
            self.error_plot_canvas.draw()

        except Exception as e:
            messagebox.showerror("Training Error", str(e))


if __name__ == "__main__":
    app = NeuralNetworkApp()
    app.mainloop()
