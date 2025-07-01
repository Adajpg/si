# Custom Neural Network GUI

This project provides an interactive GUI for building, training, and evaluating fully connected feedforward neural networks from scratch — without relying on high-level ML frameworks like TensorFlow or PyTorch. Users can define network architecture, configure training parameters, load datasets, and visualize training performance.

---

## Features

* Fully custom implementation of:

  * Neurons
  * Layers
  * Activation functions (unipolar & bipolar sigmoids)
  * Backpropagation
* Tkinter GUI for:

  * Network architecture configuration
  * Data input or file loading (RSES `.tab` format)
  * Training control and progress
  * Real-time error plot
  * Output predictions display
* Data can be entered manually or loaded from file
* Easy extensibility for custom activation functions or layers

---

## Project Structure

```
project/
│
├── main.py                     # Tkinter GUI app
├── neural_network/
│   ├── __init__.py             # NeuralNetwork class import
│   ├── activation_functions.py # Sigmoid functions (unipolar/bipolar)
│   ├── layers.py               # Layer class handling neurons
│   ├── neurons.py              # Neuron class with activation
│   ├── neural_network.py       # Core backprop-based NeuralNetwork class
│   ├── utils.py                # Weight/bias initializers
│   └── loaders/
│       └── rses_loader.py      # (Assumed) Loader for .tab files
```

---

## Getting Started

### Requirements

* Python 3.8+
* numpy
* matplotlib
* tkinter (comes preinstalled with Python)

Install dependencies:

```bash
pip install numpy matplotlib
```

> No deep learning libraries required — this is a pure Python implementation!

---

### Running the App

```bash
python main.py
```

A window will open where you can:

1. Load dataset (or manually input data).
2. Configure layers (e.g., `3,5,1` for a 3-input, 1-hidden-layer-5-neurons, 1-output network).
3. Set training parameters (learning rate, max iterations, error threshold).
4. Select activation function (`unipolar` or `bipolar`).
5. Click **"Train Network"** to start training and view predictions and error plots.

---

## Input Format

### Manual Input (default values shown in GUI)

**Input X (semicolon-separated vectors):**

```
0.0,0.0,0.0; 0.0,0.0,1.0; 1.0,1.0,1.0
```

**Output y (semicolon-separated values):**

```
0.0; 1.0; 0.0
```

> Data must match the input/output sizes of your network.

---

### File Input

Use the **"Load Data"** button to load `.tab` files (expected format compatible with `RSESLoader`).

---

## Neural Network Details

* Forward pass with per-layer activation
* Manual backpropagation using activation derivatives
* Mean squared error (MSE) loss
* Fully vectorized using NumPy
* Sigmoid activation:

  * `unipolar`: range \[0, 1]
  * `bipolar`: range \[-1, 1]

---

## Visualization

A live plot displays training error per iteration, allowing users to monitor convergence in real time.

---

## Example

* Architecture: `3,5,1`
* Activation: `unipolar`
* Learning rate: `0.5`
* Max iterations: `10000`
* Max error: `0.01`

Train on XOR-like patterns or load from real `.tab` files.

---

## Extendability

You can extend the project with:

* Additional activation functions
* New types of layers (e.g., dropout, batch norm)
* Dataset parsers beyond `.tab`
* Exporting trained weights

---

## License

This is an educational / experimental implementation. Use and adapt freely for learning or research purposes.

---