# Chisel CLI Examples

This directory contains examples showing how to use Chisel CLI in different scenarios.

## Examples

### Basic Usage (`basic_usage.py`)
Shows fundamental Chisel CLI usage:
- Creating a ChiselApp
- Using the `@capture_trace` decorator
- GPU matrix operations with profiling

```bash
python examples/basic_usage.py
```

## Installation

Make sure you have Chisel CLI installed:

```bash
# From GitHub (recommended for development)
pip install git+https://github.com/Herdora/chisel.git@dev

# Or if published to PyPI
pip install chisel-cli
```

## Prerequisites

- Python 3.8+
- PyTorch (for GPU examples)

## Running Examples

Run directly with Python:
```bash
python examples/basic_usage.py
```

## Authentication

On first run, Chisel CLI will automatically open your browser for authentication. Your credentials are securely stored in `~/.chisel` for future use.

## Notes

- GPU examples will automatically detect and use CUDA if available
- CPU fallback is provided for all GPU examples
- Trace files are saved to the backend and can be downloaded for analysis
- All examples include comprehensive error handling and user feedback