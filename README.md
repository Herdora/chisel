# Keys & Caches

![Keys & Caches Banner](assets/banner.png)

Welcome to **Keys & Caches** — the fastest way to run PyTorch models on cloud GPUs with automatic profiling and performance insights.

---

## 📚 Documentation Overview

This documentation will help you get started with Keys & Caches and make the most of its powerful features for GPU-accelerated machine learning.

---

## What is Keys & Caches?

Keys & Caches is a command-line tool that makes it effortless to run PyTorch models on high-performance cloud GPUs. With just one command, you can:

* 🚀 **Submit jobs to cloud GPUs** — Access A100, H100, and L4 GPUs instantly
* 📊 **Get automatic profiling** — Detailed performance traces for every model forward pass
* 🔍 **Debug performance bottlenecks** — Chrome trace format for visual analysis
* ⚡ **Stream real-time logs** — Watch your training progress live
* 💰 **Pay only for what you use** — No idle time charges

---

## Key Features

### 🎯 One-Command Deployment

```bash
cd examples
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run any PyTorch script on cloud GPUs
kandc run python basic_models/simple_cnn.py

# Or capture locally with profiling
kandc capture python basic_models/simple_cnn.py
```

---

### 📈 Automatic Model Profiling

```python
from kandc import capture_model_class

@capture_model_class(model_name="MyModel")
class MyModel(nn.Module):
    # Your model automatically gets profiled!
```

### 🎓 Students & Educators

* Email us at **[founders@herdora.com](mailto:founders@herdora.com)** for free credits!
* Access high-end GPUs for coursework and research
* Learn about model optimization with built-in profiling tools
* Focus on ML concepts, not DevOps complexity

### 🚀 Startups & Small Teams

* Get enterprise-grade GPU access without upfront costs
* Scale compute resources based on actual needs
* Streamline ML workflows from development to production

---


## Ready to Get Started?

👉 Jump to the **[Getting Started Guide](https://www.keysandcaches.com/docs)** to install Keys & Caches and run your first GPU job in under 5 minutes!


# 📦 Publishing to PyPI

## 🚀 Publish Stable Release (`kandc`)

1. **Bump the version** in `pyproject.toml` (e.g., `0.0.4`).

2. **Run the following commands:**
   ```bash
   rm -rf dist build *.egg-info
   python -m pip install --upgrade build twine
   python -m build
   export TWINE_USERNAME=__token__
   twine upload dist/*
   ```

## 🧪 Publish Dev Release (`kandc-dev`)

1. **Bump the dev version** in `pyproject.dev.toml` (e.g., `0.0.4.dev1`).

2. **Run the following commands:**
   ```bash
   rm -rf dist build *.egg-info
   cp pyproject.dev.toml pyproject.toml
   python -m pip install --upgrade build twine
   python -m build
   export TWINE_USERNAME=__token__
   twine upload dist/*
   git checkout -- pyproject.toml   # Restore the original pyproject.toml after publishing (undo the cp above)
   ```
   ```
