# Keys & Caches Library

![Keys & Caches Banner](assets/banner.png)

A Python library for GPU profiling and tracing PyTorch models, inspired by Weights & Biases.

---

## What is Keys & Caches?

Keys & Caches is a Python library that provides automatic profiling and performance insights for PyTorch models. With simple decorators and a familiar API, you can:

* 📊 **Get automatic profiling** — Detailed performance traces for every model forward pass
* 🔍 **Debug performance bottlenecks** — Chrome trace format for visual analysis
* ⏱️ **Time critical operations** — Built-in timing decorators
* 🎯 **Zero-overhead when disabled** — Profiling only activates when initialized

---

## Installation

```bash
pip install kandc
```

---

## Quick Start

```python
import kandc
import torch
import torch.nn as nn

# Initialize Keys & Caches
run = kandc.init(project="my-project", name="experiment-1")

# Define a model with automatic profiling
@kandc.capture_model_class(model_name="MyModel")
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# Use the model - forward passes are automatically profiled
model = MyModel()
x = torch.randn(32, 10)
y = model(x)

# Log metrics
loss = nn.functional.mse_loss(y, torch.randn(32, 1))
kandc.log({"loss": loss.item(), "epoch": 1})

# Time other operations
@kandc.timed("training_step")
def train_step(model, data):
    # Your training logic here
    pass

# Finish the run
kandc.finish()
```

---

## Key Features

### 🎯 Simple Initialization

```python
kandc.init(
    project="my-ml-project",
    name="experiment-1",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "resnet18",
    }
)
```

### 📈 Automatic Model Profiling

```python
# Class decorator
@kandc.capture_model_class(model_name="CustomModel")
class CustomModel(nn.Module):
    # Your model definition
    pass

# Instance wrapper (great for HuggingFace models)
model = AutoModel.from_pretrained("bert-base-uncased")
model = kandc.capture_model_instance(model, model_name="BERT")
```

### ⏱️ Timing Utilities

```python
# Function decorator
@kandc.timed("data_preprocessing")
def preprocess_data(data):
    # Your preprocessing logic
    return processed_data

# One-off timing
result = kandc.timed_call("expensive_operation", my_function, arg1, arg2)
```

---

## Examples

See the `examples/` directory for detailed examples:
- `quickstart.py` - Minimal getting started example
- `library_usage_example.py` - Comprehensive usage examples

---

## API Reference

### Core Functions
- `kandc.init()` - Initialize a new run with configuration
- `kandc.finish()` - Finish the current run and save all data
- `kandc.log()` - Log metrics to the current run
- `kandc.get_current_run()` - Get the active run object
- `kandc.is_initialized()` - Check if kandc is initialized

### Decorators
- `@kandc.capture_model_class()` - Profile all forward passes of a model class
- `@kandc.capture_model_instance()` - Profile all forward passes of a model instance
- `@kandc.capture_trace()` - Capture detailed PyTorch profiler traces
- `@kandc.timed()` - Time function execution
- `kandc.timed_call()` - Time a single function call

### Run Modes
- `"online"` - Default mode, full functionality
- `"offline"` - Save everything locally, no server sync
- `"disabled"` - No-op mode, useful for production

---

## 🎓 Students & Educators

Email us at **[founders@herdora.com](mailto:founders@herdora.com)** for support and collaboration opportunities!

---

# 📦 Publishing to PyPI

## 🚀 Publish Stable Release

1. **Bump the version** in `pyproject.toml` (e.g., `0.0.15`).

2. **Run the following commands:**
   ```bash
   rm -rf dist build *.egg-info
   python -m pip install --upgrade build twine
   python -m build
   export TWINE_USERNAME=__token__
   twine upload dist/*
   ```

## 🧪 Publish Dev Release

1. **Bump the dev version** in `pyproject.dev.toml` (e.g., `0.0.15.dev1`).

2. **Run the following commands:**
   ```bash
   rm -rf dist build *.egg-info
   cp pyproject.dev.toml pyproject.toml
   python -m pip install --upgrade build twine
   python -m build
   export TWINE_USERNAME=__token__
   twine upload dist/*
   git checkout -- pyproject.toml   # Restore the original pyproject.toml
   ```