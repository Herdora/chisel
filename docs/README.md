# Chisel CLI Documentation

Welcome to the comprehensive documentation for Chisel CLI - the tool for accelerating Python functions with cloud GPUs.

## 📚 Documentation Overview

| Document                                  | Description                           | Best For        |
| ----------------------------------------- | ------------------------------------- | --------------- |
| **[Getting Started](getting-started.md)** | Installation and first steps          | New users       |
| **[API Reference](api-reference.md)**     | Complete function and class reference | Developers      |
| **[Examples](examples.md)**               | Working code examples                 | Learning        |
| **[Configuration](configuration.md)**     | GPU types and optimization            | Setup           |
| **[Troubleshooting](troubleshooting.md)** | Common issues and solutions           | Problem solving |
| **[Development](development.md)**         | Contributing guide                    | Contributors    |

## 🚀 Quick Navigation

### New to Chisel?
Start with **[Getting Started](getting-started.md)** for:
- Installation instructions
- Your first GPU-accelerated function
- Basic concepts and workflow

### Ready to Code?
Check out **[Examples](examples.md)** for:
- Working code examples
- Deep learning with PyTorch
- Data processing patterns
- Multi-GPU usage

### Need Reference?
See **[API Reference](api-reference.md)** for:
- Complete ChiselApp API
- GPU type configurations
- Authentication functions
- CLI command options

### Having Issues?
Visit **[Troubleshooting](troubleshooting.md)** for:
- Common error solutions
- Performance optimization
- Network and connectivity issues
- Memory management tips

### Want to Contribute?
Read **[Development](development.md)** for:
- Development environment setup
- Code style guidelines
- Testing procedures
- Contribution workflow

## 📖 Documentation Structure

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

## 🔗 External Resources

- **[Main Repository](https://github.com/Herdora/chisel)** - Source code and issues
- **[Examples Directory](../examples/)** - Working example scripts
- **[PyPI Package](https://pypi.org/project/chisel-cli/)** - Package distribution (when published)

## 📋 Common Use Cases

**Find documentation for your use case:**

| Use Case                     | Start Here                                                         | Then See                                                                |
| ---------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| **First time setup**         | [Getting Started](getting-started.md)                              | [Examples](examples.md)                                                 |
| **PyTorch model training**   | [Examples - Deep Learning](examples.md#deep-learning-example)      | [Configuration - GPU Types](configuration.md#gpu-types)                 |
| **Data processing**          | [Examples - Data Processing](examples.md#data-processing-example)  | [Configuration - Performance](configuration.md#performance-tuning)      |
| **Command line args**        | [Examples - CLI Args](examples.md#command-line-arguments)          | [API Reference - CLI](api-reference.md#command-line-interface)          |
| **Multi-GPU usage**          | [Examples - Multi-GPU](examples.md#multi-gpu-example)              | [Configuration - Multi-GPU](configuration.md#performance-configuration) |
| **Authentication issues**    | [Troubleshooting - Auth](troubleshooting.md#authentication-issues) | [API Reference - Auth](api-reference.md#authentication-functions)       |
| **Performance optimization** | [Configuration - Performance](configuration.md#performance-tuning) | [Troubleshooting - Performance](troubleshooting.md#performance-issues)  |
| **Contributing code**        | [Development](development.md)                                      | [API Reference](api-reference.md)                                       |

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
See [Getting Started - Your First Application](getting-started.md#your-first-chisel-application)

### What GPU types are available?
```python
from chisel import GPUType

# Available options:
GPUType.A100_80GB_1  # Single GPU
GPUType.A100_80GB_2  # 2 GPUs
GPUType.A100_80GB_4  # 4 GPUs
GPUType.A100_80GB_8  # 8 GPUs
```
See [Configuration - GPU Types](configuration.md#gpu-types)

### How do I pass command line arguments?
```bash
chisel python my_script.py --arg1 value1 --arg2 value2
```
See [Examples - Command Line Arguments](examples.md#command-line-arguments)

### Something not working?
Check [Troubleshooting](troubleshooting.md) for solutions to common issues.

## 📝 Documentation Updates

This documentation is actively maintained. If you find:
- **Outdated information** - Please open an issue
- **Missing examples** - Suggest new examples
- **Unclear explanations** - Help us improve them
- **Errors** - Report them on GitHub

**Contributing to docs:** See [Development - Contributing](development.md#contributing)

---

## 📞 Contact & Support

- **📧 Email**: [contact@herdora.com](mailto:contact@herdora.com)
- **🐛 Issues**: [GitHub Issues](https://github.com/Herdora/chisel/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/Herdora/chisel/discussions)

---

**Ready to get started?** Begin with [Getting Started](getting-started.md) or jump to [Examples](examples.md) to see Chisel in action!