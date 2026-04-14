# AGENTS.md - AI Agent Guidelines for mnist-demo

## Project Overview
This is a PyTorch-based MNIST digit classification project. It implements a simple neural network with two hidden layers for handwritten digit recognition.

## Build/Run Commands

### Prerequisites
- Python 3.11+
- uv package manager (version 0.11.3+)

### Installation
```bash
uv sync  # Install dependencies from uv.lock
```

### Running the Model
```bash
uv run python main.py  # Train and evaluate the model
```

### Running Single Tests
There is no formal test suite. To test specific functionality:
```bash
uv run python -c "from layers import SimpleNN; print(SimpleNN())"  # Test model architecture
uv run python -c "import torch; print(torch.cuda.is_available())"  # Test CUDA availability
```

### Useful Commands
```bash
uv run python --version      # Check Python version
uv pip list                  # List installed packages
uv add <package>             # Add new dependency
uv remove <package>          # Remove dependency
```

## Code Style Guidelines

### Imports
- Standard library imports first (e.g., `import torch`)
- Third-party library imports second (e.g., `from torchvision import datasets`)
- Local imports last (e.g., `from layers import SimpleNN`)
- Use absolute imports over relative imports
- Group related imports together

### Formatting
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (PEP 8 compliant)
- Use blank lines to separate functions and logical sections
- Spaces around operators: `x = a + b`

### Types
- Use type hints for function parameters and return values when adding new functions
- Example: `def evaluate(model: nn.Module, loader: DataLoader) -> float:`

### Naming Conventions
- **Files**: lowercase with underscores (e.g., `layers.py`, `main.py`)
- **Classes**: PascalCase (e.g., `SimpleNN`)
- **Functions/Variables**: snake_case (e.g., `train_loader`, `show_sample_images`)
- **Constants**: UPPERCASE (e.g., `BATCH_SIZE`, `LEARNING_RATE`)
- **Private methods**: Leading underscore (e.g., `_internal_helper`)

### Error Handling
- Use try-except blocks for file I/O and external operations
- Validate inputs in public functions
- Use assertions for debugging, not runtime validation
- Log errors with meaningful messages

### PyTorch-Specific Conventions
- Always call `model.train()` before training loops
- Always call `model.eval()` before evaluation/inference
- Use `torch.no_grad()` context manager during evaluation
- Move tensors to device immediately: `data.to(DEVICE)`
- Use `CrossEntropyLoss` for classification (includes softmax)
- Save/load models using `state_dict()`: `torch.save(model.state_dict(), path)`

### Documentation
- Add docstrings to all public functions
- Format: `"""Brief description of function purpose."""`
- Include parameter descriptions for complex functions
- Comment non-obvious code logic (especially ML-specific operations)

### Git Practices
- Write clear, concise commit messages
- Keep commits focused on single concerns
- Do not commit `.venv/`, `__pycache__/`, `data/`, or `*.pth` files

## Architecture Notes
- `layers.py`: Neural network model definitions
- `main.py`: Training loop, data loading, evaluation, and visualization
- Model: SimpleNN with 784 -> 128 -> 64 -> 10 architecture
- Uses Adam optimizer with learning rate 0.001
- Batch size: 512, Epochs: 15

## Cursor/Copilot Rules
No Cursor rules (.cursor/rules/ or .cursorrules) or Copilot rules (.github/copilot-instructions.md) exist in this repository.
