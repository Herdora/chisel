# 🚀 Getting Started with Chisel

Get up and running with Chisel in under 5 minutes! This guide will walk you through installation, your first GPU job, and essential concepts.

## 📋 Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **PyTorch** installed in your environment
- **Basic familiarity** with PyTorch and command line

## 🔧 Installation

### Option 1: Install from GitHub (Recommended)
```bash
pip install git+https://github.com/Herdora/chisel.git@dev
```

### Option 2: Clone and Install
```bash
git clone https://github.com/Herdora/chisel.git
cd chisel
pip install -e .
```

### Verify Installation
```bash
chisel --version
```

You should see something like: `Chisel CLI v1.0.0`

---

## 🎯 Your First GPU Job (5-Minute Quickstart)

Let's run a simple PyTorch model on cloud GPUs!

### Step 1: Create a Simple Model

Create a file called `my_first_model.py`:

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
from chisel import capture_model_class

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
    print("🚀 Running my first Chisel job!")
    
    # Create model and sample data
    model = SimpleModel()
    batch_size = 32
    x = torch.randn(batch_size, 784)
    
    print(f"📊 Input shape: {x.shape}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass (automatically profiled!)
    model.eval()
    with torch.no_grad():
        for i in range(3):
            print(f"🔄 Forward pass {i+1}...")
            output = model(x)
            print(f"   Output shape: {output.shape}")
    
    print("✅ Job completed successfully!")

if __name__ == "__main__":
    main()
```

### Step 2: Test Locally First
```bash
python my_first_model.py
```

You should see output like:
```
🚀 Running my first Chisel job!
📊 Input shape: torch.Size([32, 784])
📊 Model parameters: 269,322
🔄 Forward pass 1...
   Output shape: torch.Size([32, 10])
🔄 Forward pass 2...
   Output shape: torch.Size([32, 10])
🔄 Forward pass 3...
   Output shape: torch.Size([32, 10])
✅ Job completed successfully!
```

### Step 3: Run on Cloud GPUs

Now let's run it on cloud GPUs with automatic profiling:

```bash
chisel python my_first_model.py
```

**What happens next:**
1. **Interactive Setup**: Chisel will prompt you for job configuration
2. **Authentication**: You'll be guided through a simple auth flow
3. **File Upload**: Your code gets uploaded to the cloud
4. **GPU Execution**: Your model runs on high-performance GPUs
5. **Real-time Logs**: Watch your job progress live
6. **Performance Traces**: Get detailed profiling data

### Step 4: View Results

After your job completes:
- **📊 Performance traces** are saved as JSON files (Chrome trace format)
- **📝 Complete logs** are available in the web interface
- **🔍 Profiling data** shows layer-level timing and memory usage

---

## 🎮 Command Formats

Chisel supports two simple command formats:

### 1. Interactive Format (Beginner-Friendly)
```bash
chisel python my_model.py --epochs 10 --batch-size 32
```
- Prompts you for all job configuration (app name, GPU count, etc.)
- Great for getting started and one-off experiments
- No chisel flags allowed - everything configured interactively

### 2. Separator Format (Automation-Ready)
```bash
chisel --app-name "my-experiment" --gpu A100-80GB:2 -- python my_model.py --epochs 10
```
- Fully specified with `--` separator, no interactive prompts
- All configuration via command line flags
- Ideal for scripts and automation

---

## 🔧 Essential Configuration

### GPU Options
| GPU Type  | Count | Memory    | Use Case                | Example Flag        |
| --------- | ----- | --------- | ----------------------- | ------------------- |
| A100-40GB | 1-8   | 40GB each | Cost-effective training | `--gpu A100:4`      |
| A100-80GB | 1-8   | 80GB each | High-memory models      | `--gpu A100-80GB:2` |
| H100      | 1-8   | 80GB each | Latest architecture     | `--gpu H100:8`      |
| L4        | 1-8   | 24GB each | Efficient inference     | `--gpu L4:1`        |

### Common Flags
- `--app-name "my-job"` - Job name for tracking
- `--gpu <type>` - GPU configuration (e.g., `A100-80GB:1`, `H100:4`, `L4:2`)
- `--upload-dir .` - Directory to upload (default: current)
- `--requirements requirements.txt` - Python dependencies

### Requirements Files
Create a `requirements.txt` for your dependencies:
```
torch>=2.0.0
torchvision
numpy
matplotlib
transformers  # if using HuggingFace models
```

---

## 📊 Model Profiling Basics

Chisel automatically profiles your models when you use the decorators:

### Class Decorator (Most Common)
```python
from chisel import capture_model_class

@capture_model_class(model_name="MyModel")
class MyModel(nn.Module):
    # Your model definition
```

### Instance Wrapper (For Pre-built Models)
```python
from chisel import capture_model_instance

# For HuggingFace models, etc.
model = AutoModel.from_pretrained("bert-base-uncased")
model = capture_model_instance(model, model_name="BERT")
```

### Profiling Features
- **⏱️ Layer-level timing** - See which layers are bottlenecks
- **💾 Memory tracking** - Monitor GPU memory usage
- **🔍 Shape recording** - Debug tensor dimension issues
- **📈 Chrome traces** - Visual timeline in chrome://tracing

---

## 🎯 Best Practices

### 1. Always Test Locally First
```bash
# Test your code works before cloud submission
python my_model.py

# Then run on cloud GPUs
chisel python my_model.py
```

### 2. Start Small, Scale Up
- Begin with 1 GPU for development
- Scale to 2-4 GPUs for training
- Use 8 GPUs only for very large models

### 3. Use Descriptive Job Names
```bash
chisel --app-name "resnet50-imagenet-experiment" --gpu H100:2 python train.py
```

### 4. Organize Your Code
```
my_project/
├── requirements.txt      # Dependencies
├── train.py             # Main training script
├── model.py             # Model definitions
├── data.py              # Data loading
└── .chiselignore        # Files to exclude from upload
```

### 5. Exclude Unnecessary Files
Create a `.chiselignore` file:
```
# Large files that can be downloaded in the script
*.bin
*.safetensors
data/
checkpoints/
wandb/
.git/
__pycache__/
```

---

## 🐛 Troubleshooting

### Common Issues

**❌ "No module named chisel"**
```bash
pip install git+https://github.com/Herdora/chisel.git@dev
```

**❌ "Authentication failed"**
- Follow the auth flow prompts carefully
- Check your internet connection
- Try `chisel --logout` then retry

**❌ "Script not found"**
- Ensure your script path is correct
- Check that the script is in your upload directory

**❌ "Import errors in cloud"**
- Add missing packages to `requirements.txt`
- Test locally with a clean environment first

### Getting Help

1. **Check Examples**: See our [Examples Guide](examples.md)
2. **Community Support**: Join our Discord/Slack
3. **Bug Reports**: Open issues on GitHub
4. **Documentation**: This guide and inline help (`chisel --help`)

---

## 🎉 Next Steps

Congratulations! You've successfully run your first Chisel job. Here's what to explore next:

### 📚 Learn More
- **[Examples Guide](examples.md)** - See real-world use cases
- **Advanced Features** - Multi-GPU training, custom requirements
- **Performance Optimization** - Using profiling data effectively

### 🚀 Try More Examples
```bash
# Computer vision
chisel python examples/vision_models/resnet_example.py

# NLP with transformers
chisel --gpu A100-80GB:2 --requirements requirements_examples/nlp_requirements.txt -- python examples/nlp_models/pretrained_models.py

# Generative models
chisel python examples/generative_models/gan_example.py
```

### 🔧 Advanced Usage
- **Custom Docker environments**
- **Multi-node training**
- **Integration with MLOps tools**
- **Automated hyperparameter sweeps**

---

## 💡 Tips for Success

1. **Start Simple**: Begin with basic models and gradually increase complexity
2. **Profile Everything**: Use `@capture_model_class` on all your models
3. **Monitor Costs**: Check GPU usage in the web interface
4. **Iterate Fast**: Use 1 GPU for development, scale for final training
5. **Share Results**: Export traces and share insights with your team

---

*Ready to dive deeper? Check out our [Examples Guide](examples.md) for real-world use cases and advanced techniques!*
