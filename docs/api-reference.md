# API Reference

Complete reference for Chisel CLI classes and functions.

## ChiselApp

Main class for GPU-accelerated applications.

```python
from chisel import ChiselApp, GPUType

app = ChiselApp(name, upload_dir=".", gpu=None)
```

**Parameters:**
- `name` (str): Application name for job tracking
- `upload_dir` (str): Directory to upload (default: current directory)
- `gpu` (GPUType): GPU configuration

**Examples:**
```python
# Recommended: Using GPUType enum
app = ChiselApp("my-app", gpu=GPUType.A100_80GB_2)

# Legacy: Using string
app = ChiselApp("my-app", gpu="A100-80GB:2")
```

### @capture_trace()

Decorator for GPU execution and tracing.

```python
@app.capture_trace(
    trace_name=None,
    record_shapes=False,
    profile_memory=False
)
def my_function():
    pass
```

**Parameters:**
- `trace_name` (str): Operation identifier
- `record_shapes` (bool): Record tensor shapes for debugging
- `profile_memory` (bool): Profile memory usage

**Example:**
```python
@app.capture_trace(trace_name="matrix_ops", record_shapes=True)
def matrix_multiply(a, b):
    import torch
    return torch.mm(a, b)
```

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
# Direct usage
app = ChiselApp("my-app", gpu=GPUType.A100_80GB_4)

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

Credentials stored in `~/.chisel/credentials.json` with restricted permissions.

## Command Line Interface

### chisel command

```bash
chisel <command> [arguments...]
```

**How it works:**
1. Sets `CHISEL_ACTIVATED=1` environment variable
2. Executes command with GPU functionality enabled
3. Streams real-time output and status updates during execution

**Examples:**
```bash
# Basic usage
chisel python my_script.py

# With arguments
chisel python train.py --epochs 10

# Any Python command
chisel python -m pytest tests/
chisel jupyter notebook
```

## Environment Variables

| Variable             | Purpose                     | Set By           |
| -------------------- | --------------------------- | ---------------- |
| `CHISEL_ACTIVATED`   | Activates GPU functionality | `chisel` command |
| `CHISEL_BACKEND_URL` | Override backend URL        | User (optional)  |
| `CHISEL_API_KEY`     | Authentication token        | Auth system      |
| `CHISEL_BACKEND_RUN` | Running on backend          | Backend system   |
| `CHISEL_JOB_ID`      | Current job ID              | Backend system   |

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
try:
    app = ChiselApp("my-app", gpu=GPUType.A100_80GB_1)
    
    @app.capture_trace(trace_name="safe_ops")
    def safe_function(data):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.tensor(data, device=device)
        return tensor.cpu().numpy()
        
except RuntimeError as e:
    print(f"❌ Chisel failed: {e}")
    # Fallback to local execution
```

## Advanced Usage

### Multiple Apps

```python
# Different GPU configs for different tasks
prep_app = ChiselApp("preprocess", gpu=GPUType.A100_80GB_1)
train_app = ChiselApp("training", gpu=GPUType.A100_80GB_4)

@prep_app.capture_trace(trace_name="clean")
def preprocess(data): pass

@train_app.capture_trace(trace_name="train")  
def train(data): pass
```

### Conditional Activation

```python
import os

if os.environ.get("CHISEL_ACTIVATED") == "1":
    print("🚀 Chisel is activated!")
```

### Custom Upload Directory

```python
# Upload specific directory
app = ChiselApp("my-app", upload_dir="./src")

# Upload parent directory  
app = ChiselApp("my-app", upload_dir="../")
```

## Type Hints

```python
from typing import Optional
from chisel import ChiselApp, GPUType
import numpy as np

def create_app(name: str, gpu: Optional[GPUType] = None) -> ChiselApp:
    return ChiselApp(name, gpu=gpu)

@app.capture_trace(trace_name="typed_ops")
def process(data: np.ndarray) -> np.ndarray:
    import torch
    tensor = torch.from_numpy(data)
    return (tensor * 2).numpy()
```

**Next:** [Examples](examples.md) | [Configuration](configuration.md) | [Troubleshooting](troubleshooting.md)