# Chisel CLI

Chisel CLI - Accelerate your Python functions with cloud GPUs.

## Installation

### From GitHub (dev branch)

```bash
pip install git+https://github.com/Herdora/chisel.git@dev
```

### From PyPI (when published)

```bash
pip install chisel-cli
```

## Quick Start

Create a Python script with Chisel:

```python
from chisel import ChiselApp

app = ChiselApp("my-app")

@app.capture_trace(trace_name="my_function", record_shapes=True)
def my_function(x):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(x, device=device)
    return (tensor * 2).cpu().numpy()

result = my_function([1, 2, 3, 4])
```

Run with Chisel to enable cloud GPU execution:

```bash
chisel python my_script.py
```

## Usage

- **Normal execution**: `python my_script.py` - No overhead, decorators are pass-through
- **Chisel execution**: `chisel python my_script.py` - Full cloud GPU functionality with authentication and job submission

The `chisel` command works with any Python command:
```bash
chisel python -m pip list
chisel pytest tests/
chisel jupyter notebook
```

## Examples

Check out the [`examples/`](examples/) directory for comprehensive usage examples:

- **[Basic Usage](examples/basic_usage.py)** - Fundamental Chisel CLI usage

Run examples:
```bash
chisel python examples/basic_usage.py
```

## Authentication

Chisel handles authentication automatically when using the `chisel` command:

1. First run opens browser for authentication
2. Credentials stored securely in `~/.chisel`
3. Subsequent runs authenticate instantly

## Development

```bash
git clone https://github.com/Herdora/chisel.git
cd chisel
pip install -e .

# Run examples
chisel python examples/basic_usage.py
```

## Project Structure

```
chisel/
├── src/chisel/               # Main package
│   ├── __init__.py          # Public API and CLI entry point
│   ├── core.py              # ChiselApp and core functionality  
│   ├── auth.py              # Authentication service
│   ├── spinner.py           # Loading spinner utility
│   └── constants.py         # Configuration constants
├── examples/                # Usage examples
│   └── basic_usage.py       # Basic functionality
├── pyproject.toml          # Package configuration
└── README.md               # This file
```