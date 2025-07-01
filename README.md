# Custom Neural Network GUI

[![Tests](https://github.com/Adajpg/si/actions/workflows/python-tests.yml/badge.svg)](https://github.com/Adajpg/si/actions/workflows/python-tests.yml)

This project provides an interactive GUI for building, training, and evaluating fully connected feedforward neural networks — written from scratch, with **no TensorFlow or PyTorch**. Ideal for learning how backpropagation works under the hood.

---

## 🚀 Features

- Fully custom implementation of:
  - Neurons
  - Layers
  - Sigmoid activation functions (unipolar & bipolar)
  - Backpropagation (online learning)
- Tkinter GUI with:
  - Layer configuration
  - Manual or file-based data entry
  - Live training plot
  - Output preview
- File loader for `.tab` format (RSES-style datasets)
- Customizable parameters: architecture, learning rate, stopping criteria
- **Early stopping** when total error drops below threshold
- Unit-tested core logic (`unittest`)

---

## 🧱 Project Structure

```

project/
│
├── main.py                     # Tkinter GUI application
├── neural\_network/
│   ├── **init**.py             # NeuralNetwork import
│   ├── activation\_functions.py # Sigmoid activations
│   ├── layers.py               # Layer class (list of neurons)
│   ├── neurons.py              # Neuron class with weights/bias
│   ├── neural\_network.py       # Training logic and model definition
│   ├── utils.py                # \[optional] weight/bias utilities
│   └── loaders/
│       └── rses\_loader.py      # .tab file parser
├── tests/
│   ├── test\_neurons.py
│   ├── test\_layers.py
│   ├── test\_network.py
│   └── test\_rses\_loader.py
├── .gitignore                  # Ignores **pycache**, \*.pyc, etc.
└── README.md

````

---

Dobrze — skoro masz już pełny `requirements.txt`, zaktualizuję sekcję instalacji w `README.md`, aby odwoływała się bezpośrednio do niego.

---

## 📦 Installation

### Requirements

- Python 3.8+
- tkinter (usually comes with Python)
- All other dependencies listed in `requirements.txt`

### Setup

1. (Optional) Create virtual environment:
 ```bash
 python -m venv venv
 source venv/bin/activate  # or venv\Scripts\activate on Windows
```

2. Install all dependencies:

   ```bash
   pip install -r requirements.txt
   ```

No TensorFlow, PyTorch or Keras — this is a pure Python neural network engine.

---

## ▶️ Running the App

```bash
python main.py
```

Then in the GUI:

1. Load or manually enter data
2. Define layer layout, e.g., `3,5,1`
3. Set training parameters (rate, iterations, error)
4. Choose activation: `unipolar` or `bipolar`
5. Click **Train Network** — results and error plot will appear

---

## 🧠 Input Format

### Manual Input

**Input X** (semicolon-separated row vectors):

```
0.0,0.0,0.0; 0.0,0.0,1.0; 1.0,1.0,1.0
```

**Output y** (semicolon-separated scalar values):

```
0.0; 1.0; 0.0
```

The number of input/output entries must match the architecture.

---

### File Input (.tab)

You can load `.tab` files with format similar to:

```
x x x d
0 0 0 0
1 1 1 1
```

* First row is header
* Each subsequent row contains input values followed by target
* Parsed using `RSESLoader` (robust to bad rows)

---

## 📉 Training & Visualization

* Real-time plot of **total error per iteration**
* Early stopping once error drops below threshold
* Fully online learning (one sample per step)

---

## ✅ Tests

Unit tests cover:

* Neuron activation
* Layer forward/backward propagation
* Neural network training and output shape
* Error convergence
* `.tab` loader behavior and error handling

Run all tests:

```bash
python -m unittest discover tests
```

Sample output:

```
Ran 10 tests in 0.29s
OK
```

---

## 🧩 Extendability

Add your own:

* Activation functions (`ReLU`, `tanh`, `softmax`)
* Layer types (dropout, batchnorm, residual)
* Model saving/loading with `pickle`
* CLI or Flask/Gradio interface

---

## 📷 Example Configuration

* Architecture: `3,5,1`
* Activation: `unipolar`
* Learning rate: `0.5`
* Iterations: `10000`
* Max error: `0.01`

Trained on XOR or `.tab` dataset:

* Outputs: 0.95, 0.03, 0.99, ...

---

## 📄 License

Educational / experimental — use freely for learning or prototyping.