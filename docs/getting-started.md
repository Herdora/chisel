# Getting Started

Chisel CLI accelerates your Python functions by automatically running them on cloud GPUs.

## Installation

```bash
pip install git+https://github.com/Herdora/chisel.git@dev
```

## Quick Start

**1. Create a Python script:**

```python
from chisel import ChiselApp, GPUType

app = ChiselApp("my-app", gpu=GPUType.A100_80GB_1)

@app.capture_trace(trace_name="matrix_ops")
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

**3. First-time authentication:**
- Browser opens automatically
- Sign in or create Herdora account
- Credentials stored securely in `~/.chisel/`

**4. Real-time streaming:**
- Live upload progress with spinner
- Instant job ID and URL when ready
- Real-time stdout/stderr output
- Cost information when job completes

## Key Concepts

### ChiselApp Configuration

```python
app = ChiselApp("app-name", gpu=GPUType.A100_80GB_2)
```

- `"app-name"`: Job tracking identifier
- `gpu`: GPU configuration (1, 2, 4, or 8x A100-80GB)

### @capture_trace Decorator

```python
@app.capture_trace(trace_name="operation", record_shapes=True)
def my_function():
    # Runs on cloud GPU when using 'chisel' command
    pass
```

### GPU Types

| Type                  | GPUs    | Memory | Use Case               |
| --------------------- | ------- | ------ | ---------------------- |
| `GPUType.A100_80GB_1` | 1x A100 | 80GB   | Development, inference |
| `GPUType.A100_80GB_2` | 2x A100 | 160GB  | Medium training        |
| `GPUType.A100_80GB_4` | 4x A100 | 320GB  | Large models           |
| `GPUType.A100_80GB_8` | 8x A100 | 640GB  | Massive models         |

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
@app.capture_trace(trace_name="preprocess")
def preprocess(data): pass

@app.capture_trace(trace_name="train")  
def train(data): pass
```

**Error handling:**
```python
@app.capture_trace(trace_name="robust")
def robust_function(data):
    try:
        # GPU code here
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # ...
    except Exception as e:
        print(f"Error: {e}")
        raise
```

**Next:** [API Reference](api-reference.md) | [Examples](examples.md) | [Configuration](configuration.md)