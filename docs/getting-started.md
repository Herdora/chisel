# Getting Started

Chisel CLI accelerates your Python functions by automatically running them on cloud GPUs.

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

**3. Authentication (first time only):**
- Chisel CLI checks authentication first
- Browser opens automatically for login
- Sign in or create Herdora account
- Credentials stored securely in `~/.chisel/credentials.json`

**4. Interactive setup:**
- After authentication, CLI prompts for:
  - App name (for job tracking)
  - Upload directory (default: current directory)
  - Requirements file (default: requirements.txt)
  - GPU configuration (1, 2, 4, or 8x A100-80GB)

**5. Real-time streaming:**
- Live upload progress with spinner
- Instant job ID and URL when ready
- Real-time stdout/stderr output
- Cost information when job completes

## Logout

To clear your stored credentials:

```bash
chisel --logout
```

This removes your authentication token from `~/.chisel/credentials.json` and requires re-authentication on next use.

## Version Information

Check your Chisel CLI version:

```bash
chisel --version    # or chisel -v or chisel version
```

## Key Concepts

### @capture_trace Decorator

```python
from chisel import capture_trace

@capture_trace(trace_name="operation", record_shapes=True)
def my_function():
    # Runs on cloud GPU when using 'chisel' command
    # Runs locally when using 'python' command
    pass
```

**Parameters:**
- `trace_name`: Operation identifier for trace files
- `record_shapes`: Record tensor shapes for debugging
- `profile_memory`: Profile memory usage

### GPU Types

When prompted by the CLI, choose from:

| Option | GPU Configuration | Memory | Use Case               |
| ------ | ----------------- | ------ | ---------------------- |
| 1      | 1x A100-80GB      | 80GB   | Development, inference |
| 2      | 2x A100-80GB      | 160GB  | Medium training        |
| 4      | 4x A100-80GB      | 320GB  | Large models           |
| 8      | 8x A100-80GB      | 640GB  | Massive models         |

## Command Line Usage

```bash
# Basic usage
chisel python script.py

# With arguments
chisel python script.py --arg1 value --arg2 value

# Any Python command
chisel python -m pytest
chisel jupyter notebook
```

## Common Patterns

**Multiple functions:**
```python
@capture_trace(trace_name="preprocess")
def preprocess(data): pass

@capture_trace(trace_name="train")  
def train(data): pass
```

**Error handling:**
```python
@capture_trace(trace_name="robust")
def robust_function(data):
    try:
        # GPU code here
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # ...
    except Exception as e:
        print(f"Error: {e}")
        raise
```

**Local development:**
```python
# This works both locally and on cloud
@capture_trace(trace_name="my_function")
def my_function():
    # Runs locally with 'python script.py'
    # Runs on GPU with 'chisel python script.py'
    pass
```

**Next:** [API Reference](api-reference.md) | [Examples](examples.md) | [Configuration](configuration.md)