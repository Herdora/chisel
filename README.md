# Keys & Caches

![Keys & Caches Banner](assets/banner.png)

**The fastest way to run PyTorch models on cloud GPUs with automatic profiling and performance insights.**

---

## 🚀 What is Keys & Caches?

Keys & Caches is a command-line tool that makes it effortless to run PyTorch models on high-performance cloud GPUs. With just one command, you can:

- **🚀 Submit jobs to cloud GPUs** - Access A100, H100, and L4 GPUs instantly
- **📊 Get automatic profiling** - Detailed performance traces for every model forward pass
- **🔍 Debug performance bottlenecks** - Chrome trace format for visual analysis
- **⚡ Stream real-time logs** - Watch your training progress live
- **💰 Pay only for what you use** - No idle time charges

## 🎯 Key Features

### 🎮 **One-Command Deployment**
```bash
# Run any PyTorch script on cloud GPUs
kandc python train.py --model-size large --epochs 100
```

### 📈 **Automatic Model Profiling**
```python
from kandc import capture_model_class

@capture_model_class(model_name="MyModel")
class MyModel(nn.Module):
    # Your model automatically gets profiled!
```

### 🎮 **Flexible GPU Configurations**
- **A100 GPUs** (40GB/80GB) - Proven performance for training and inference
- **H100 GPUs** (80GB) - Latest architecture with enhanced performance
- **L4 GPUs** (24GB) - Cost-effective option for efficient workloads
- **Scale 1-8 GPUs** - From development to massive scale training

## 🔧 Installation

### Prerequisites
- **Python 3.8+** (Python 3.9+ recommended)
- **PyTorch** installed in your environment

### Install Keys & Caches
```bash
pip install kandc
```

### Verify Installation
```bash
kandc --version
```

## 🎯 Quick Start (5 Minutes)

### 1. Create a Simple Model
Create `my_first_model.py`:

```python
import torch
import torch.nn as nn
from kandc import capture_model_class

@capture_model_class(model_name="FirstModel")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

def main():
    print("🚀 Running my first Keys & Caches job!")
    
    model = SimpleModel()
    x = torch.randn(32, 784)
    
    # Forward pass (automatically profiled!)
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"✅ Output shape: {output.shape}")
    
    print("✅ Job completed successfully!")

if __name__ == "__main__":
    main()
```

### 2. Test Locally First
```bash
python my_first_model.py
```

### 3. Run on Cloud GPUs
```bash
kandc python my_first_model.py
```

That's it! Your model runs on high-performance GPUs with automatic profiling. 🎉

## 🎮 Command Formats

### Interactive Format (Beginner-Friendly)
```bash
kandc python my_model.py --epochs 10 --batch-size 32
```
- Prompts you for job configuration (app name, GPU count, etc.)
- Great for getting started and experiments

### Separator Format (Automation-Ready)
```bash
kandc --app-name "my-experiment" --gpu A100-80GB:2 -- python my_model.py --epochs 10
```
- Fully specified with `--` separator
- Ideal for scripts and automation

## 🎯 GPU Options

| GPU Type  | Count | Memory    | Use Case                | Example Flag        |
| --------- | ----- | --------- | ----------------------- | ------------------- |
| A100-40GB | 1-8   | 40GB each | Cost-effective training | `--gpu A100:4`      |
| A100-80GB | 1-8   | 80GB each | High-memory models      | `--gpu A100-80GB:2` |
| H100      | 1-8   | 80GB each | Latest architecture     | `--gpu H100:8`      |
| L4        | 1-8   | 24GB each | Efficient inference     | `--gpu L4:1`        |

## 📊 Automatic Model Profiling

Keys & Caches automatically profiles your models:

### Class Decorator (Most Common)
```python
from kandc import capture_model_class

@capture_model_class(model_name="MyModel")
class MyModel(nn.Module):
    # Your model definition
```

### Instance Wrapper (For Pre-built Models)
```python
from kandc import capture_model_instance

# For HuggingFace models, etc.
model = AutoModel.from_pretrained("bert-base-uncased")
model = capture_model_instance(model, model_name="BERT")
```

### Profiling Features
- **⏱️ Layer-level timing** - See which layers are bottlenecks
- **💾 Memory tracking** - Monitor GPU memory usage
- **🔍 Shape recording** - Debug tensor dimension issues
- **📈 Chrome traces** - Visual timeline in chrome://tracing

## 💡 Examples

### Computer Vision
```bash
# ResNet training
kandc python examples/vision_models/resnet_example.py
```

### NLP & Transformers
```bash
# HuggingFace BERT
kandc --requirements requirements_examples/nlp_requirements.txt -- python examples/nlp_models/pretrained_models.py
```

### Vision-Language Models
```bash
# OpenAI CLIP
kandc --requirements requirements_examples/vlm_requirements.txt -- python examples/vlm_models/clip_example.py
```

### Generative Models
```bash
# GANs and VAEs
kandc python examples/generative_models/gan_example.py
```
 
## 📚 Documentation

- **[🚀 Getting Started](docs/getting-started.md)** - Installation, setup, and your first GPU job
- **[💡 Examples](docs/examples.md)** - Comprehensive examples and use cases  
- **[📞 Contact & Support](docs/contact.md)** - Get help and connect with the community

## 💰 Publishing to PyPI

### Prerequisites
```bash
# Install build and publishing tools
pip install build twine

# Ensure you have PyPI credentials configured
# Create ~/.pypirc or use environment variables
```

### Build and Upload Process
```bash
# 1. Update version in pyproject.toml
# Edit the version field: version = "0.1.1"

# 2. Clean previous builds
rm -rf dist/ build/

# 3. Build the package
python -m build

# 4. Check the built package
twine check dist/*


# 6. Upload to PyPI
twine upload dist/*
```

## 🆘 Support

- **📧 Email**: [support@herdora.com](mailto:support@herdora.com)
- **🐛 Issues**: [GitHub Issues](https://github.com/Herdora/kandc/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/Herdora/kandc/discussions)

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

*Ready to accelerate your ML workflows? Install Keys & Caches and run your first GPU job in under 5 minutes! 🚀*