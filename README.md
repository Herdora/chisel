# Keys & Caches

![Keys & Caches Banner](assets/banner.png)

A Python library for experiment tracking and machine learning workflow management.

---

## What is Keys & Caches?

Keys & Caches is a Python library that provides experiment tracking and workflow management for machine learning projects. With a simple API, you can:

* 📊 **Track experiments** — Automatically log metrics and hyperparameters
* 🌐 **Cloud dashboard** — Real-time visualization of your experiments
* 🏷️ **Organize projects** — Group related experiments together
* 🎯 **Zero-overhead when disabled** — Tracking only activates when initialized

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

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)

def main():
    # Initialize experiment tracking
    kandc.init(
        project="my-project",
        name="experiment-1",
        config={"batch_size": 32, "learning_rate": 0.01}
    )

    # Your training/inference code
    model = SimpleNet()
    data = torch.randn(32, 784)
    output = model(data)
    loss = output.mean()

    # Log metrics
    kandc.log({"loss": loss.item(), "accuracy": 0.85})

    # Finish the run
    kandc.finish()

if __name__ == "__main__":
    main()
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

### 📊 Metrics Logging

```python
# Log single or multiple metrics
kandc.log({"loss": 0.25, "accuracy": 0.92})

# Log with step numbers for training loops
for epoch in range(100):
    loss = train_epoch()
    kandc.log({"epoch_loss": loss}, step=epoch)
```

### 🌐 Multiple Modes

```python
# Online mode (default) - full cloud experience
kandc.init(project="my-project")

# Offline mode - local development
kandc.init(project="my-project", mode="offline")

# Disabled mode - zero overhead
kandc.init(project="my-project", mode="disabled")
```

### 🔮 Inference Tracking

```python
import kandc
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

def run_inference():
    # Initialize inference tracking
    kandc.init(
        project="inference-demo",
        name="simple-inference",
        config={"batch_size": 32}
    )
    
    # Create model and wrap with profiler
    model = SimpleNet()
    model = kandc.capture_model_instance(model, model_name="SimpleNet")
    model.eval()
    
    # Run inference
    data = torch.randn(32, 784)
    with torch.no_grad():
        predictions = model(data)
        confidence = torch.softmax(predictions, dim=1).max(dim=1)[0].mean()
    
    # Log results
    kandc.log({
        "avg_confidence": confidence.item(),
        "batch_size": 32
    })
    
    kandc.finish()

if __name__ == "__main__":
    run_inference()
```

---

## Examples

See the `examples/` directory for detailed examples:
- `complete_example.py` - Simple getting started example
- `offline_example.py` - Offline mode usage

---

## API Reference

### Core Functions
- `kandc.init()` - Initialize a new run with configuration
- `kandc.finish()` - Finish the current run and save all data
- `kandc.log()` - Log metrics to the current run
- `kandc.get_current_run()` - Get the active run object
- `kandc.is_initialized()` - Check if kandc is initialized

### Run Modes
- `"online"` - Default mode, full cloud functionality
- `"offline"` - Save everything locally, no server sync
- `"disabled"` - No-op mode, zero overhead

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