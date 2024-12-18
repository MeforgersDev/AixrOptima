# AixrOptima

AixrOptima is a modular, extensible toolkit designed to optimize Large Language Model (LLM) fine-tuning through cutting-edge parameter-efficient adaptation, quantization, and quantum-inspired optimization techniques. Its goal is to significantly reduce memory and computational overhead while maintaining or improving model performance.

By integrating methods like LoRA, QLoRA, DoRA, QDoRA, and quantum-inspired optimization strategies, AixrOptima offers a new dimension of flexibility and efficiency in refining pre-trained language models. Whether you are working on GPT-like architectures, LLaMA, or custom transformer models, AixrOptima helps you achieve faster, more cost-effective, and privacy-friendly model adaptation.

## Key Features

### Parameter-Efficient Adaptation (LoRA/DoRA/QDoRA):
Inject low-rank matrices into frozen weights to adapt large models without fully retraining all parameters. This reduces memory and improves training speed.

### Quantization (QLoRA):
Leverage low-bit (e.g., 4-bit) quantization to decrease memory footprint and computational cost, making fine-tuning feasible on resource-constrained devices.

### Quantum-Inspired Optimization:
Employ techniques analogous to simulated annealing or variational quantum algorithms to escape local minima and improve optimization outcomes without requiring actual quantum hardware.

### Modular Integration:
Easily integrate AixrOptima into your existing PyTorch models with minimal code changes. Automatically discover linear layers and apply LoRA and quantization hooks.

### Flexible and Extensible:
Configure parameters such as rank, bit depth, quantization schemes, and quantum-inspired optimization hyperparameters to suit your use case.

## Installation
You can install AixrOptima directly from GitHub:

```bash
pip install git+https://github.com/MeforgersDev/aixroptima.git
```

### Requirements:
*   Python 3.7+
*   PyTorch >= 1.8.0
*   fairscale >= 0.4.0 (for certain parallelization functionalities)
For a full list of requirements, see `requirements.txt`.

## Quick Start
A simple usage example is shown below. Assume you have a PyTorch-based LLM model:

```python
import torch
import torch.nn.functional as F
from AixrOptima import integrate_aixroptima, QuantumInspiredOptimizer, regularization_loss

# Your custom model or a pre-existing transformer model
model = ...  # A pre-trained LLM or any transformer-based model
model = model.cuda()

# Integrate AixrOptima
# This will inject low-rank adaptation modules and quantization hooks into linear layers
model = integrate_aixroptima(
    model, 
    rank=8,
    bits=4,
    use_quantization=True,
    per_channel_quant=True
)

# Sample data (for demonstration)
X = torch.randint(0, 32000, (2, 128)).cuda()  # tokens
Y = torch.randint(0, 32000, (2, 128)).cuda()  # targets

# Forward pass
output = model(X)
loss = F.cross_entropy(output.view(-1, model.vocab_size), Y.view(-1))

# Add regularization from LoRA modules
reg_loss = regularization_loss(model.named_modules(), lambda_reg=1e-4)
total_loss = loss + reg_loss

# Gather LoRA parameters for optimization
lora_params = []
for n, m in model.named_modules():
    if hasattr(m, 'lora_module'):
        lora_params.extend([m.lora_module.A, m.lora_module.B])

# Use a quantum-inspired optimizer for parameter updates
optimizer = QuantumInspiredOptimizer(lora_params, lr=1e-3, initial_temp=1.0, cooling_rate=0.95)
total_loss.backward(retain_graph=True)
optimizer.step(total_loss)

print(f"Loss: {loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}")
```

## Tutorials & Examples
### LLaMA Integration:
Check out `examples/llama3_integration_example.py` for a step-by-step guide on integrating AixrOptima with a LLaMA 3 model.

### GPT-like Models:
Similar steps can be followed to integrate with GPT-2, GPT-Neo, or other transformer variants by adapting the `integrate_aixroptima` function calls.

## Advanced Configuration
AixrOptima is highly configurable. You can specify:
*   Rank for low-rank adaptation (e.g., `rank=8`)
*   Quantization Bits (e.g., `bits=4` for 4-bit quantization)
*   Per-channel Quantization for finer control over quantization distribution
*   Quantum-Inspired Optimization Parameters such as `initial_temp` (initial temperature) and `cooling_rate` to control the simulated annealing process.

These settings can be passed directly to `integrate_aixroptima` and `QuantumInspiredOptimizer`.

## Architecture
*   `lora.py`: Implementations of low-rank adaptation modules, including A and B matrices and their integration into existing weight matrices.
*   `quantization.py`: Utilities for per-tensor and per-channel quantization.
*   `qoptimizer.py`: Quantum-inspired optimizers that blend standard gradient-based methods with annealing and probabilistic acceptance criteria.
*   `integrate.py`: A convenience function to automatically find linear layers in a model and apply LoRA and quantization hooks.
*   `utils.py`: Helper functions for regularization, parameter counting, and other utility operations.
*   `config.py`: Default configuration parameters and hyperparameters.

## Tests
We provide a set of unit tests in the `tests/` directory:

```bash
pytest tests/
```

This ensures that LoRA, quantization, and quantum-inspired optimization components work as expected. We recommend running tests before any production use.

## Contributing
Contributions are welcome! If you would like to add new features, improve documentation, or fix bugs:

1.  Fork the repository.
2.  Create a new branch for your feature or bugfix.
3.  Write tests and ensure all existing tests pass with pytest.
4.  Open a Pull Request and describe your changes.

See `CONTRIBUTING.md` for more details.

## License
AixrOptima is released under the MIT License. You are free to use, modify, and distribute this software for both commercial and non-commercial purposes.