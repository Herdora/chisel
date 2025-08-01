# Chisel CLI

**Accelerate your Python functions with cloud GPUs.**

Chisel CLI automatically offloads your Python functions to powerful cloud GPUs with zero code changes. Simply add a decorator and run with the `chisel` command.

## ⚡ Quick Start

**1. Install Chisel:**
```bash
pip install chisel-cli
```

**2. Add Chisel to your code:**
```python
from chisel import ChiselApp, GPUType

app = ChiselApp("my-app", gpu=GPUType.A100_80GB_1)

@app.capture_trace(trace_name="matrix_ops", record_shapes=True)
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
# Local execution (normal Python)
python my_script.py

# Cloud GPU execution (powered by Chisel)
chisel python my_script.py
```

That's it! Your function now runs on a cloud A100 GPU. 🚀

## 🎯 Key Features

- **Zero overhead locally** - Decorators are pass-through when not using `chisel`
- **Automatic GPU detection** - Code works on both CPU and GPU seamlessly  
- **Multiple GPU types** - From single A100 to 8x A100 configurations
- **Argument passing** - Command-line arguments work transparently
- **Secure authentication** - Browser-based auth with secure credential storage
- **Real-time streaming** - Live output and status updates during job execution
- **Job tracking** - Web interface to monitor your cloud GPU jobs

## 📖 Documentation

| Document                                         | Description                   |
| ------------------------------------------------ | ----------------------------- |
| **[📚 Complete Documentation](docs/)**            | Full documentation hub        |
| **[🚀 Getting Started](docs/getting-started.md)** | Installation and first steps  |
| **[📋 API Reference](docs/api-reference.md)**     | Complete API documentation    |
| **[💡 Examples](docs/examples.md)**               | Working examples and patterns |
| **[⚙️ Configuration](docs/configuration.md)**     | GPU types and optimization    |
| **[🔧 Troubleshooting](docs/troubleshooting.md)** | Common issues and solutions   |

## 🔥 Examples

### Command Line Arguments
```bash
chisel python examples/args_example.py --iterations 10
```

### Deep Learning Training
```python
from chisel import ChiselApp, GPUType

app = ChiselApp("training", gpu=GPUType.A100_80GB_4)

@app.capture_trace(trace_name="model_training", profile_memory=True)
def train_model(epochs=100):
    # Your PyTorch training code here
    # Automatically runs on 4x A100-80GB GPUs
    pass
```

### Data Processing
```python
@app.capture_trace(trace_name="data_processing", record_shapes=True)
def process_large_dataset(data_size_gb=10):
    # Process large datasets on GPU
    pass
```

**[👀 See more examples](docs/examples.md)**

## 🖥️ GPU Types

| GPU Configuration     | Use Case               | Memory |
| --------------------- | ---------------------- | ------ |
| `GPUType.A100_80GB_1` | Development, inference | 80GB   |
| `GPUType.A100_80GB_2` | Medium training        | 160GB  |
| `GPUType.A100_80GB_4` | Large models           | 320GB  |
| `GPUType.A100_80GB_8` | Massive models         | 640GB  |

## 🛠️ Development

**Local development setup:**
```bash
git clone https://github.com/Herdora/chisel.git
cd chisel
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

**Run tests and formatting:**
```bash
ruff check src/ examples/
ruff format src/ examples/
pytest
```

## 📦 Project Structure

```
chisel/
├── src/chisel/              # Main package
│   ├── __init__.py         # Public API and CLI entry point  
│   ├── core.py             # ChiselApp and core functionality
│   ├── auth.py             # Authentication service
│   ├── constants.py        # GPU types and configuration
│   └── spinner.py          # UI utilities
├── examples/               # Working examples
├── docs/                   # Comprehensive documentation
└── pyproject.toml         # Package configuration
```

## 🚀 Why Chisel?

- **Effortless scaling** - Run the same code locally or on powerful cloud GPUs
- **Cost effective** - Pay only for GPU time used, automatic job cleanup
- **Developer friendly** - Minimal code changes, works with existing workflows
- **Production ready** - Secure authentication, job monitoring, error handling

## 📞 Support

- **📧 Email**: [contact@herdora.com](mailto:contact@herdora.com)
- **📖 Documentation**: [docs/](docs/)
- **🐛 Issues**: [GitHub Issues](https://github.com/Herdora/chisel/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/Herdora/chisel/discussions)

---

**Ready to accelerate your Python code?** Start with the [Getting Started Guide](docs/getting-started.md)!
