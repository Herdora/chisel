# Chisel CLI Documentation

Welcome to the comprehensive documentation for Chisel CLI - the simplified tool for accelerating Python functions with cloud GPUs.

## 📚 Documentation Overview

| Document | Description | Best For |
|----------|-------------|----------|
| **[Getting Started](getting-started.md)** | Installation and first steps | New users |
| **[API Reference](api-reference.md)** | Complete function and CLI reference | Developers |
| **[Examples](examples.md)** | Working code examples | Learning |
| **[Configuration](configuration.md)** | CLI flags and optimization | Setup |
| **[Troubleshooting](troubleshooting.md)** | Common issues and solutions | Problem solving |
| **[Development](development.md)** | Contributing guide | Contributors |

## 🚀 Quick Navigation

### New to Chisel?
Start with **[Getting Started](getting-started.md)** for:
- Simple installation with pip
- Your first GPU-accelerated function
- Understanding the `@capture_trace` decorator
- Using the `chisel` command

### Ready to Code?
Check out **[Examples](examples.md)** for:
- Working examples from the `examples/` directory
- Command line argument handling
- Deep learning with PyTorch
- Multi-GPU usage patterns
- Best practices and performance tips

### Need Reference?
See **[API Reference](api-reference.md)** for:
- Complete `capture_trace` decorator API
- CLI command options and flags
- GPU type configurations
- Environment variables
- Authentication functions

### Having Issues?
Visit **[Troubleshooting](troubleshooting.md)** for:
- Installation and setup problems
- CLI and authentication issues
- Runtime errors and memory problems
- Performance optimization
- Network connectivity issues

### Want to Contribute?
Read **[Development](development.md)** for:
- Development environment setup
- Project structure and architecture
- Code style guidelines and testing
- Contribution workflow

## 📖 What's New in the Simplified API

Chisel CLI has been simplified significantly:

### Before (Old API)
```python
from chisel import ChiselApp, GPUType

app = ChiselApp("my-app", gpu=GPUType.A100_80GB_2)

@app.capture_trace()
def my_function():
    pass

app.run()  # Complex setup
```

### Now (New API)
```python
from chisel import capture_trace

@capture_trace(trace_name="my_function")
def my_function():
    pass

# Run locally: python script.py
# Run on GPU: chisel python script.py
```

**Key improvements:**
- **Single import**: Just `from chisel import capture_trace`
- **No app setup**: Direct decorator usage
- **CLI handles everything**: Authentication, configuration, job submission
- **Same code everywhere**: Works locally and on cloud

## 🔗 External Resources

- **[Main Repository](https://github.com/Herdora/chisel)** - Source code and issues
- **[Examples Directory](../examples/)** - Working example scripts
- **[PyPI Package](https://pypi.org/project/chisel-cli/)** - Package distribution

## 📋 Common Use Cases

**Find documentation for your use case:**

| Use Case | Start Here | Then See |
|----------|------------|----------|
| **First time setup** | [Getting Started](getting-started.md) | [Examples](examples.md) |
| **Basic GPU computation** | [Examples - Basic Usage](examples.md#basic-usage) | [Configuration - GPU Types](configuration.md#gpu-types) |
| **Command line scripts** | [Examples - Command Line Args](examples.md#command-line-arguments) | [API Reference - CLI](api-reference.md#command-line-interface) |
| **PyTorch training** | [Examples - Deep Learning](examples.md#deep-learning) | [Configuration - Performance](configuration.md#performance-configuration) |
| **Multi-GPU usage** | [Examples - Multi-GPU](examples.md#multi-gpu) | [Configuration - Multi-GPU](configuration.md#multi-gpu-configuration) |
| **Custom requirements** | [Examples - Requirements](examples.md#requirements-file) | [Configuration - Requirements](configuration.md#requirements-configuration) |
| **Authentication issues** | [Troubleshooting - Auth](troubleshooting.md#authentication-issues) | [API Reference - Auth](api-reference.md#authentication-functions) |
| **Performance optimization** | [Configuration - Performance](configuration.md#performance-configuration) | [Troubleshooting - Performance](troubleshooting.md#performance-issues) |
| **Contributing code** | [Development](development.md) | [API Reference](api-reference.md) |

## 🆘 Quick Help

**Most common questions:**

### How do I install Chisel?
```bash
pip install git+https://github.com/Herdora/chisel.git@dev
```
See [Getting Started - Installation](getting-started.md#installation)

### How do I run my script on GPU?
```bash
# Normal execution (local)
python my_script.py

# GPU execution (cloud)
chisel python my_script.py
```
See [Getting Started - Quick Start](getting-started.md#quick-start)

### How do I add the decorator?
```python
from chisel import capture_trace

@capture_trace(trace_name="my_operation")
def my_function():
    import torch
    # Your GPU code here
    pass
```
See [Getting Started - Key Concepts](getting-started.md#key-concepts)

### What GPU types are available?
```bash
# CLI prompts for GPU selection:
# 1. 1x A100-80GB (80GB)   - Development, inference
# 2. 2x A100-80GB (160GB)  - Medium training  
# 4. 4x A100-80GB (320GB)  - Large models
# 8. 8x A100-80GB (640GB)  - Massive models

# Or specify directly:
chisel python script.py --gpu 4
```
See [Configuration - GPU Types](configuration.md#gpu-types)

### How do I pass command line arguments?
```bash
# Script arguments work normally
chisel python train.py --epochs 100 --batch-size 32

# Mix with chisel configuration
chisel python train.py --app-name "training" --gpu 2 --epochs 100
```
See [Examples - Command Line Arguments](examples.md#command-line-arguments)

### How do I handle authentication?
```bash
# First run opens browser automatically
chisel python script.py

# To logout and re-authenticate
chisel --logout
```
See [Getting Started - Authentication](getting-started.md#authentication)

### Something not working?
Check [Troubleshooting](troubleshooting.md) for solutions to common issues.

## 📝 Documentation Structure

```
docs/
├── README.md              # This overview (start here)
├── getting-started.md     # Installation and first steps  
├── api-reference.md       # Complete API documentation
├── examples.md           # Code examples and patterns
├── configuration.md      # Setup and optimization guide
├── troubleshooting.md    # Problem solving guide
└── development.md        # Contributor guide
```

## 🎯 Quick Start Paths

### Path 1: Complete Beginner
1. [Getting Started](getting-started.md) - Install and understand basics
2. [Examples - Basic Usage](examples.md#basic-usage) - Try first example
3. [Configuration](configuration.md) - Learn CLI options
4. [Examples](examples.md) - Explore more examples

### Path 2: Experienced Developer  
1. [API Reference](api-reference.md) - Complete function reference
2. [Examples](examples.md) - Working code patterns
3. [Configuration](configuration.md) - Advanced optimization
4. [Development](development.md) - Contributing

### Path 3: Troubleshooting
1. [Troubleshooting](troubleshooting.md) - Common issues
2. [Getting Started](getting-started.md) - Verify setup
3. [Examples - Basic Usage](examples.md#basic-usage) - Test minimal case
4. Contact support if needed

## 📞 Contact & Support

- **📧 Email**: [contact@herdora.com](mailto:contact@herdora.com) - Direct support
- **🐛 Issues**: [GitHub Issues](https://github.com/Herdora/chisel/issues) - Bug reports
- **💬 Discussions**: [GitHub Discussions](https://github.com/Herdora/chisel/discussions) - Questions and ideas

## 📈 Documentation Updates

This documentation reflects the **simplified API** introduced in recent versions:

- **Removed**: ChiselApp class, complex setup
- **Added**: Direct `capture_trace` decorator usage
- **Simplified**: CLI handles all configuration
- **Enhanced**: Better examples and troubleshooting

If you find outdated information or have suggestions:
- **Report issues**: [GitHub Issues](https://github.com/Herdora/chisel/issues)
- **Suggest improvements**: [GitHub Discussions](https://github.com/Herdora/chisel/discussions)
- **Contribute**: See [Development - Contributing](development.md#contributing)

---

**Ready to get started?** Begin with [Getting Started](getting-started.md) or jump to [Examples](examples.md) to see Chisel in action!