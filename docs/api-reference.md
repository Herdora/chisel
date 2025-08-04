# API Reference

Complete reference for Chisel CLI functions and classes.

## capture_trace

Main decorator for GPU execution and tracing.

```python
from chisel import capture_trace

@capture_trace(
    trace_name=None,
    record_shapes=False,
    profile_memory=False
)
def my_function():
    pass
```

**Parameters:**
- `trace_name` (str): Operation identifier for trace files
- `record_shapes` (bool): Record tensor shapes for debugging
- `profile_memory` (bool): Profile memory usage

**Example:**
```python
@capture_trace(trace_name="matrix_ops", record_shapes=True)
def matrix_multiply(a, b):
    import torch
    return torch.mm(a, b)
```

**Behavior:**
- **Local execution** (`python script.py`): Decorator is pass-through, function runs normally
- **Cloud execution** (`chisel python script.py`): Function runs on GPU with profiling

## GPU Types

```python
from chisel import GPUType

GPUType.A100_80GB_1  # Single A100-80GB GPU
GPUType.A100_80GB_2  # 2x A100-80GB GPUs
GPUType.A100_80GB_4  # 4x A100-80GB GPUs  
GPUType.A100_80GB_8  # 8x A100-80GB GPUs
```

**Usage:**
```python
# Get string value
gpu_string = GPUType.A100_80GB_2.value  # "A100-80GB:2"

# List all types
for gpu in GPUType:
    print(f"{gpu.name} = {gpu.value}")
```

## Authentication

### Functions

```python
from chisel.auth import is_authenticated, authenticate, clear_credentials

# Check authentication status
if is_authenticated():
    print("✅ Authenticated")

# Manual authentication
api_key = authenticate("http://localhost:8000")

# Clear credentials
clear_credentials()
```

### Credential Storage

Credentials stored in `~/.chisel/credentials.json` with restricted permissions (directory: 0o700, file: 0o600).

## Command Line Interface

### chisel command

```bash
chisel <command> [arguments...]
chisel --logout
chisel --version
```

**How it works:**
1. **Interactive setup**: Prompts for app name, upload directory, requirements file, and GPU configuration
2. **Authentication**: Handles authentication automatically
3. **Job submission**: Uploads code and submits job to cloud GPU
4. **Real-time output**: Streams job status and output

**Examples:**
```bash
# Basic usage
chisel python my_script.py

# With arguments
chisel python train.py --epochs 10

# Any Python command
chisel python -m pytest
chisel jupyter notebook

# Logout (clear credentials)
chisel --logout

# Show version
chisel --version
chisel -v
chisel version
```

## Environment Variables

| Variable             | Purpose              | Set By          |
| -------------------- | -------------------- | --------------- |
| `CHISEL_BACKEND_RUN` | Running on backend   | Backend system  |
| `CHISEL_JOB_ID`      | Current job ID       | Backend system  |
| `CHISEL_BACKEND_URL` | Override backend URL | User (optional) |
| `CHISEL_API_KEY`     | Authentication token | Auth system     |

**Custom backend:**
```bash
export CHISEL_BACKEND_URL="https://api.herdora.com"
chisel python my_script.py
```

## Error Handling

### Common Exceptions

**RuntimeError:** Authentication failed
```python
RuntimeError("❌ Authentication failed. Unable to get valid API key.")
```

**AssertionError:** Script not in upload directory
```python
AssertionError("Script /path/to/script.py is not inside upload_dir /path/to/upload")
```

### Best Practices

```python
@capture_trace(trace_name="safe_ops")
def safe_function(data):
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.tensor(data, device=device)
        return tensor.cpu().numpy()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
```

## Advanced Usage

### Multiple Functions

```python
@capture_trace(trace_name="preprocess")
def preprocess(data): 
    # Data preprocessing
    pass

@capture_trace(trace_name="train")  
def train(data): 
    # Model training
    pass

@capture_trace(trace_name="evaluate")
def evaluate(data): 
    # Model evaluation
    pass
```

### Conditional Execution

```python
import os

# Check if running on Chisel backend
if os.environ.get("CHISEL_BACKEND_RUN") == "1":
    print("🚀 Running on cloud GPU!")
else:
    print("💻 Running locally")
```

### Local Development

```python
# This works both locally and on cloud
@capture_trace(trace_name="my_function")
def my_function():
    # Your GPU code here
    pass

# Test locally
if __name__ == "__main__":
    result = my_function()
    print(f"Result: {result}")
```

## Type Hints

```python
from typing import Optional
from chisel import capture_trace
import numpy as np

@capture_trace(trace_name="typed_ops")
def process(data: np.ndarray) -> np.ndarray:
    import torch
    tensor = torch.from_numpy(data)
    return (tensor * 2).numpy()
```

**Next:** [Examples](examples.md) | [Configuration](configuration.md) | [Troubleshooting](troubleshooting.md)