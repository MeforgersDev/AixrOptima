# AixrOptima

AixrOptima is an advanced optimization algorithm designed to outperform other algorithms by leveraging quantum functions for faster learning rates and better convergence. This project includes a custom quantum circuit simulation, dynamic learning rate adjustments, quantum memory, and Monte Carlo simulations for global optimization.

## Features

- **Quantum Gradient Calculation**: Simulates quantum circuits to enhance gradient descent.
- **Dynamic Learning Rate**: Learning rate adjusts dynamically based on the current training step.
- **Quantum Monte Carlo Simulation**: Uses quantum-inspired Monte Carlo methods to improve global optimization.
- **Quantum Memory**: Utilizes past gradients to optimize updates, improving convergence stability.
- **Loss and F1 Score Visualization**: Compares multiple optimizers (Adam, AdamW, and AixrOptima) based on both loss and F1 score.
  
## Installation

You can install the required dependencies using the following command:

```bash
pip install numpy matplotlib scipy
```

# Usage

## Basic Usage
```python
import numpy as np
from aixroptima_quantum import AixrOptimaQuantum

# Example model parameters
params = np.random.randn(3)

# Initialize the AixrOptimaQuantum optimizer
optimizer = AixrOptimaQuantum(params)

# Simulate optimization process over 100 epochs
for epoch in range(100):
    grads = 2 * params  # Example gradient (can be replaced by actual gradients)
    params = optimizer.update(grads)
    print(f"Epoch {epoch + 1}, Updated Params: {params}")
```

# File Structure

- **aixroptima_quantum.py:** Contains the AixrOptimaQuantum optimizer class and its methods.
- **quantum_circuit.py:** Implements the quantum circuit simulation and operations on qubits.
- **tests/:** Directory containing unit tests for the project.

# Future Improvements

- Extend quantum circuit simulation with real quantum gate operations.
- Add support for different optimization techniques, such as RMSprop.
- Improve the Monte Carlo simulation to explore more robust global optimization strategies.

# Test Results

![Only Loss Low Better](https://github.com/MeforgersDev/AixrOptima/blob/main/tests/only_loss.png?raw=true)

![F1 Score And Loss](https://github.com/MeforgersDev/AixrOptima/blob/main/tests/F1_LOSS.png?raw=true)


# License

**This project is licensed under the MIT License. See the LICENSE file for more details.**
