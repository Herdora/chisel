# Chisel CLI

Accelerate your Python functions with cloud GPUs using a simple decorator.

## Quick Start

**1. Install Chisel CLI:**
```bash
pip install chisel-cli
```

**2. Create your script:**
```python
from chisel import capture_trace

@capture_trace(trace_name="matrix_ops")
def matrix_multiply(size=1000):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    result = torch.mm(a, b)
    
    return result.cpu().numpy()

if __name__ == "__main__":
    result = matrix_multiply(2000)
    print(f"Result shape: {result.shape}")
```

**3. Run on cloud GPU:**
```bash
# Local execution
python my_script.py

# Cloud GPU execution
chisel python my_script.py
```

**4. Interactive setup:**
- Chisel CLI prompts for app name, upload directory, requirements file, and GPU configuration
- First-time authentication opens browser automatically
- Real-time job status and output streaming

## Features

- **Simple decorator**: Just add `@capture_trace()` to your functions
- **Local & cloud**: Same code runs locally or on cloud GPUs
- **Interactive CLI**: Guided setup for job configuration
- **Real-time streaming**: Live output and job status
- **GPU profiling**: Automatic performance traces and memory analysis
- **Multi-GPU support**: Scale from 1 to 8x A100-80GB GPUs

## Usage

### Basic Example

```python
from chisel import capture_trace

@capture_trace(trace_name="my_function")
def my_gpu_function():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.randn(1000, 1000, device=device)

# Runs locally with: python script.py
# Runs on GPU with: chisel python script.py
```

### Multiple Functions

```python
@capture_trace(trace_name="preprocess")
def preprocess(data): 
    # Data preprocessing
    pass

@capture_trace(trace_name="train")  
def train(data): 
    # Model training
    pass

@capture_trace(trace_name="evaluate")
def evaluate(data): 
    # Model evaluation
    pass
```

### Command Line Arguments

```bash
# Pass arguments to your script
chisel python train.py --epochs 100 --batch-size 32

# Or configure the job directly
chisel python train.py --app-name "training-job" --gpu 4
```

## GPU Options

When prompted by the CLI, choose from:

| Option | GPU Configuration | Memory | Use Case               |
| ------ | ----------------- | ------ | ---------------------- |
| 1      | 1x A100-80GB      | 80GB   | Development, inference |
| 2      | 2x A100-80GB      | 160GB  | Medium training        |
| 4      | 4x A100-80GB      | 320GB  | Large models           |
| 8      | 8x A100-80GB      | 640GB  | Massive models         |

## Documentation

- **[Getting Started](docs/getting-started.md)** - Installation and first steps
- **[API Reference](docs/api-reference.md)** - Complete function reference
- **[Examples](docs/examples.md)** - Working code examples
- **[Configuration](docs/configuration.md)** - Setup and optimization
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

## Examples

See the [examples](examples/) directory for working code:

- [Basic usage](examples/simple_example.py) - Matrix operations
- [Command line args](examples/args_example.py) - Script with arguments
- [Deep learning](docs/examples.md#deep-learning) - PyTorch training
- [Multi-GPU](docs/examples.md#multi-gpu) - Parallel processing

## Authentication

Chisel CLI handles authentication automatically:

```bash
# First time: Browser opens for authentication
chisel python my_script.py

# Logout to clear credentials
chisel --logout
```

Credentials are stored securely in `~/.chisel/credentials.json`.

## Development

```bash
# Clone repository
git clone https://github.com/Herdora/chisel.git
cd chisel

# Install in development mode
pip install -e .

# Run tests
pytest

# Check code style
ruff check src/ examples/
```

## Contributing

We welcome contributions! Please see [Development Guide](docs/development.md) for details.

## Support

- **📧 Email**: [contact@herdora.com](mailto:contact@herdora.com)
- **🐛 Issues**: [GitHub Issues](https://github.com/Herdora/chisel/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/Herdora/chisel/discussions)

## License

MIT License - see [LICENSE](LICENSE) for details.