# API Reference

Complete reference for Chisel CLI functions and command-line interface.

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
- `trace_name` (str, optional): Operation identifier for trace files. Defaults to function name.
- `record_shapes` (bool): Record tensor shapes for debugging. Default: False.
- `profile_memory` (bool): Profile memory usage. Default: False.

**Example:**
```python
@capture_trace(trace_name="matrix_ops", record_shapes=True)
def matrix_multiply(a, b):
    import torch
    return torch.mm(a, b)
```

**Behavior:**
- **Local execution** (`python script.py`): Decorator is pass-through, function runs normally
- **Cloud execution** (`chisel python script.py`): Function runs on GPU with profiling and trace generation

**Trace Files:**
When running on cloud GPUs, trace files are automatically saved as `{trace_name}.json` in Chrome trace format for analysis.

## GPU Types

Available GPU configurations when using the CLI:

```python
from chisel import GPUType

GPUType.A100_80GB_1  # Single A100-80GB GPU
GPUType.A100_80GB_2  # 2x A100-80GB GPUs
GPUType.A100_80GB_4  # 4x A100-80GB GPUs  
GPUType.A100_80GB_8  # 8x A100-80GB GPUs
```

**Usage in CLI:**
```bash
chisel python script.py --gpu 1  # Single GPU
chisel python script.py --gpu 2  # Dual GPU
chisel python script.py --gpu 4  # Quad GPU
chisel python script.py --gpu 8  # Octa GPU
```

## Command Line Interface

### chisel command

Main command for cloud GPU execution:

```bash
chisel python <script.py> [script_args...] [chisel_options...]
```

**Examples:**
```bash
# Basic usage with interactive setup
chisel python my_script.py

# With chisel configuration flags
chisel python train.py --app-name "training-job" --gpu 4

# With script arguments
chisel python train.py --epochs 100 --batch-size 32

# Combined chisel and script arguments
chisel python train.py --app-name "training" --gpu 2 --epochs 50
```

### Chisel CLI Options

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--app-name` | `-a` | Job name for tracking | Interactive prompt |
| `--upload-dir` | `-d` | Directory to upload | Current directory (`.`) |
| `--requirements` | `-r` | Requirements file | `requirements.txt` |
| `--gpu` | `-g` | GPU count (1,2,4,8) | 1 |
| `--interactive` | `-i` | Force interactive mode | Auto-detect |

**Interactive Mode:**
When required options are missing, CLI automatically enters interactive mode with prompts for:
- App name for job tracking
- Upload directory selection  
- Requirements file path
- GPU configuration choice

### Utility Commands

```bash
# Show version
chisel --version
chisel -v
chisel version

# Clear authentication credentials
chisel --logout
```

## Environment Variables

| Variable | Purpose | Set By |
|----------|---------|--------|
| `CHISEL_BACKEND_RUN` | Indicates running on backend | Backend system |
| `CHISEL_JOB_ID` | Current job ID | Backend system |
| `CHISEL_BACKEND_URL` | Override backend URL | User (optional) |
| `CHISEL_API_KEY` | Authentication token | Auth system |

**Custom backend:**
```bash
export CHISEL_BACKEND_URL="https://api.herdora.com"
chisel python my_script.py
```

**Detecting execution environment:**
```python
import os

if os.environ.get("CHISEL_BACKEND_RUN") == "1":
    print("🚀 Running on cloud GPU!")
    job_id = os.environ.get("CHISEL_JOB_ID")
    print(f"Job ID: {job_id}")
else:
    print("💻 Running locally")
```

## Authentication Functions

```python
from chisel.auth import is_authenticated, authenticate, clear_credentials

# Check authentication status
if is_authenticated():
    print("✅ Authenticated")

# Manual authentication (rarely needed)
api_key = authenticate("http://localhost:8000")

# Clear credentials (equivalent to chisel --logout)
clear_credentials()
```

**Credential Storage:**
- Credentials stored in `~/.chisel/credentials.json`
- Directory permissions: 0o700 (user only)
- File permissions: 0o600 (user read/write only)

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
    # Cloud-specific logic
else:
    print("💻 Running locally")
    # Local-specific logic
```

### Local Development Pattern

```python
# This works both locally and on cloud
@capture_trace(trace_name="my_function")
def my_function():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Your GPU code here
    return torch.randn(100, device=device)

# Test locally first
if __name__ == "__main__":
    result = my_function()
    print(f"Result: {result}")
```

### Command Line Argument Parsing

```python
import argparse
from chisel import capture_trace

@capture_trace(trace_name="training")
def train_model(epochs, learning_rate):
    # Training logic
    pass

def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    train_model(args.epochs, args.lr)

if __name__ == "__main__":
    main()
```

## Type Hints

```python
from typing import Optional
from chisel import capture_trace
import numpy as np
import torch

@capture_trace(trace_name="typed_ops")
def process(data: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = tensor.to(device)
    result = tensor * 2
    return result.cpu().numpy()
```

## CLI Workflow

The `chisel` command follows this workflow:

1. **Parse arguments** - Separate chisel flags from script arguments
2. **Check authentication** - Authenticate if needed (opens browser)
3. **Interactive setup** - Prompt for missing configuration
4. **Package code** - Create tar.gz archive of upload directory
5. **Submit job** - Upload and submit to backend
6. **Stream output** - Real-time job status and output
7. **Save traces** - Performance traces available for download

**Next:** [Examples](examples.md) | [Configuration](configuration.md) | [Troubleshooting](troubleshooting.md)