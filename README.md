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

# Cloud GPU execution (interactive format)
chisel python my_script.py

# Cloud GPU execution (separator format)
chisel --app-name "matrix-ops" --gpu 2 -- python my_script.py
```

**4. Two execution formats:**
- **Interactive format**: `chisel python script.py [script-args]` - CLI prompts for chisel configuration
- **Separator format**: `chisel [chisel-flags] -- python script.py [script-args]` - All configuration upfront
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

Chisel CLI supports **two clean argument formats** for handling chisel configuration and script arguments:

```bash
# 1. Separator format (RECOMMENDED) - chisel flags first, then -- separator
chisel --app-name "training-job" --gpu 4 -- python train.py --epochs 100 --batch-size 32

# 2. Interactive format - script args only (prompts for chisel config)
chisel python train.py --epochs 100 --batch-size 32
```

**Key Features:**
- **Clean Separation**: Use `--` for explicit separation between chisel and script arguments
- **Interactive Fallback**: Script args only triggers interactive mode for chisel configuration
- **Error Prevention**: Mixing chisel flags with script args is not allowed - keeps things simple
- **Helpful Errors**: Clear messages guide you to the correct format

**Available Chisel Flags:**
- `--app-name, -a` - Job name for tracking
- `--gpu, -g` - GPU count (1, 2, 4, 8)
- `--upload-dir, -d` - Directory to upload
- `--requirements, -r` - Requirements file
- `--interactive, -i` - Force interactive mode
- `--preview, -p` - Preview upload contents

**The `--` separator is what determines chisel flags vs script arguments:**
- **Before `--`** = chisel flags  
- **After `--`** = script arguments (regardless of flag names)
- **No `--`** = interactive mode (ALL args after `python script.py` are script args)

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

### Basic Usage

```bash
# Simple script execution
chisel python my_script.py

# With custom configuration (separator format)
chisel --app-name "matrix-ops" --gpu 2 -- python my_script.py
```

### Script Arguments

```bash
# Interactive format - script args only (prompts for chisel config)
chisel python train.py --model-size large --epochs 100 --batch-size 32

# Separator format - clean separation with --
chisel --app-name "training-job" --gpu 4 -- python train.py --model-size large --epochs 100 --batch-size 32
```

### Advanced Configuration

```bash
# Custom requirements and upload directory
chisel --requirements custom_requirements.txt --upload-dir src/ --gpu 2 -- python models/train.py --config config.yaml

# Preview upload contents before submission
chisel --preview -- python my_script.py

# Force interactive mode even with flags
chisel --interactive python my_script.py
```

### Working Examples

See the [examples](examples/) directory for comprehensive working code:

- **[Basic Models](examples/basic_models/)** - Simple CNN and linear regression
- **[NLP Models](examples/nlp_models/)** - Transformers and HuggingFace integration  
- **[Vision Models](examples/vision_models/)** - ResNet and computer vision
- **[Edge Cases](examples/edge_cases/)** - Command line arguments and long-running demos
- **[Requirements Examples](examples/requirements_examples/)** - Custom dependency configurations

## Authentication

Chisel CLI handles authentication automatically:

```bash
# First time: Browser opens for authentication
chisel python my_script.py

# Logout to clear credentials
chisel --logout
```

Credentials are stored securely in `~/.chisel/credentials.json`.

## Backend Configuration

Chisel CLI connects to the production API by default. For development, you can switch to localhost:

```bash
# Production (default)
chisel python my_script.py
# → Connects to https://api.keysandcaches.com

# Development mode
export CHISEL_DEV=1
chisel python my_script.py
# → Connects to http://localhost:8000

# Custom backend
export CHISEL_BACKEND_URL="https://custom-api.example.com"
chisel python my_script.py
```

**Priority order:**
1. `CHISEL_BACKEND_URL` environment variable (highest priority)
2. `CHISEL_DEV=1` → uses localhost:8000
3. Default → uses production API

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