# Getting Started

Chisel CLI accelerates your Python functions by running them on cloud GPUs with a simple decorator.

## Installation

```bash
pip install git+https://github.com/Herdora/chisel.git@dev
```

## Quick Start

**1. Create a Python script:**

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

**2. Run on cloud GPU:**

```bash
# Local execution
python my_script.py

# Cloud GPU execution  
chisel python my_script.py
```

**3. Interactive setup:**

When you run `chisel python my_script.py`, the CLI will guide you through:

- **Authentication** (first time only): Browser opens automatically for login
- **App name**: Enter a name for job tracking (e.g., "matrix-test")
- **Upload directory**: Which folder to upload (default: current directory)
- **Requirements file**: Python dependencies (default: requirements.txt)
- **GPU configuration**: Choose 1, 2, 4, or 8x A100-80GB GPUs

**4. Real-time execution:**
- Upload progress with spinner
- Job ID and URL when ready
- Live stdout/stderr streaming
- Performance traces saved automatically

## Command Line Options

You can skip the interactive prompts by providing flags:

```bash
# Basic job submission
chisel python my_script.py --app-name "my-job" --gpu 2

# Full configuration
chisel python train.py \
  --app-name "training-job" \
  --upload-dir "./src" \
  --requirements "requirements.txt" \
  --gpu 4

# With script arguments
chisel python train.py --epochs 100 --batch-size 32
```

**Available flags:**
- `--app-name` (`-a`): Job name for tracking
- `--upload-dir` (`-d`): Directory to upload (default: current directory)
- `--requirements` (`-r`): Requirements file (default: requirements.txt)
- `--gpu` (`-g`): GPU count - 1, 2, 4, or 8 (default: 1)
- `--interactive` (`-i`): Force interactive mode even with flags

## Key Concepts

### @capture_trace Decorator

```python
from chisel import capture_trace

@capture_trace(trace_name="operation", record_shapes=True, profile_memory=True)
def my_function():
    # Runs locally with 'python script.py'
    # Runs on GPU with 'chisel python script.py'
    pass
```

**Parameters:**
- `trace_name`: Operation identifier for trace files
- `record_shapes`: Record tensor shapes for debugging
- `profile_memory`: Profile memory usage

**Behavior:**
- **Local execution** (`python script.py`): Decorator is pass-through, function runs normally
- **Cloud execution** (`chisel python script.py`): Function runs on GPU with profiling

### GPU Options

| Option | GPU Configuration | Memory | Use Case               |
| ------ | ----------------- | ------ | ---------------------- |
| 1      | 1x A100-80GB      | 80GB   | Development, inference |
| 2      | 2x A100-80GB      | 160GB  | Medium training        |
| 4      | 4x A100-80GB      | 320GB  | Large models           |
| 8      | 8x A100-80GB      | 640GB  | Massive models         |

## Authentication

### First Time Setup

```bash
chisel python my_script.py
```

1. CLI detects no authentication
2. Browser opens automatically
3. Sign in or create Herdora account
4. Credentials stored securely in `~/.chisel/credentials.json`

### Managing Authentication

```bash
# Check version
chisel --version

# Clear credentials (logout)
chisel --logout

# Next run will prompt for re-authentication
chisel python my_script.py
```

## Common Patterns

### Multiple Functions

```python
@capture_trace(trace_name="preprocess")
def preprocess(data):
    # Data preprocessing
    return processed_data

@capture_trace(trace_name="train")  
def train(data):
    # Model training
    return trained_model

@capture_trace(trace_name="evaluate")
def evaluate(model, data):
    # Model evaluation
    return metrics

if __name__ == "__main__":
    data = preprocess(raw_data)
    model = train(data)
    results = evaluate(model, test_data)
```

### Error Handling

```python
@capture_trace(trace_name="robust_function")
def robust_function(data):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Your GPU code here
        tensor = torch.tensor(data, device=device)
        return tensor.cpu().numpy()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
```

### Command Line Arguments

```python
import argparse
from chisel import capture_trace

@capture_trace(trace_name="training")
def train_model(epochs, batch_size):
    # Training logic here
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    train_model(args.epochs, args.batch_size)

if __name__ == "__main__":
    main()
```

Run with:
```bash
chisel python train.py --epochs 100 --batch-size 64
```

## Local Development

Your code works identically locally and on cloud:

```python
@capture_trace(trace_name="my_function")
def my_function():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Same code works everywhere
    return torch.randn(100, device=device)

# Test locally first
if __name__ == "__main__":
    result = my_function()
    print(f"✅ Local test passed: {result.shape}")
```

```bash
# Develop and test locally
python my_script.py

# Run on cloud GPU when ready
chisel python my_script.py
```

## Next Steps

- **[Examples](examples.md)** - Working code examples
- **[API Reference](api-reference.md)** - Complete function reference  
- **[Configuration](configuration.md)** - Advanced setup options
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions