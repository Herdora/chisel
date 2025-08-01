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

```python
from chisel import ChiselApp

# Create a Chisel app - authentication happens automatically
app = ChiselApp("my-app")

# Use tracing decorator for GPU profiling
@app.capture_trace(trace_name="my_function", record_shapes=True)
def my_function(x):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(x, device=device)
    return (tensor * 2).cpu().numpy()

# Call your function - it will run on cloud GPUs with profiling
result = my_function([1, 2, 3, 4])
```

## Examples

Check out the [`examples/`](examples/) directory for comprehensive usage examples:

- **[Basic Usage](examples/basic_usage.py)** - Fundamental Chisel CLI usage

Run examples:
```bash
python examples/basic_usage.py
```

## Authentication

Chisel CLI handles authentication automatically on first use:

1. Creates a `ChiselApp` → opens browser for auth
2. Stores credentials securely in `~/.chisel`
3. Subsequent uses authenticate instantly

Manual control:
```python
from chisel import authenticate, clear_credentials, is_authenticated

# Check auth status
if not is_authenticated():
    authenticate()  # Opens browser if needed
    
# Clear stored credentials
clear_credentials()
```

## Development

```bash
git clone https://github.com/Herdora/chisel.git
cd chisel
pip install -e .

# Run examples
python examples/basic_usage.py
```

## Project Structure

```
chisel/
├── src/chisel/               # Main package
│   ├── __init__.py          # Public API
│   ├── core.py              # ChiselApp and core functionality  
│   └── constants.py         # Configuration constants
├── examples/                # Usage examples
│   └── basic_usage.py       # Basic functionality
├── pyproject.toml          # Package configuration
└── README.md               # This file
```