# Examples

Working examples for different Chisel CLI use cases. All examples are available in the [`examples/`](../examples/) directory.

## Quick Reference

| Example | Description | Command |
|---------|-------------|---------|
| [Basic Usage](#basic-usage) | Matrix operations | `chisel python examples/simple_example.py` |
| [Command Line Args](#command-line-arguments) | Script with arguments | `chisel python examples/args_example.py --iterations 5` |
| [Requirements](#requirements-file) | Custom dependencies | `chisel python examples/requirements_example.py` |
| [Inline Tracing](#inline-tracing) | Wrapping external functions | `chisel python examples/specific_call.py` |
| [Deep Learning](#deep-learning) | PyTorch training | See below |
| [Multi-GPU](#multi-gpu) | Parallel processing | See below |

## Basic Usage

**File:** [`examples/simple_example.py`](../examples/simple_example.py)

```python
from chisel import capture_trace

@capture_trace(trace_name="matrix_multiply", record_shapes=True)
def matrix_multiply(size: int = 1000):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    result = torch.mm(a, b)

    print(f"✅ Matrix multiplication completed! Shape: {result.shape}")
    return result.cpu().numpy()

@capture_trace(trace_name="simple_computation")
def simple_computation(n: int = 1000000):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(n, device=device)
    result = x.pow(2).sum()

    print(f"✅ Computation completed! Result: {result.item()}")
    return result.item()

if __name__ == "__main__":
    print("🚀 Starting Chisel example")

    matrix_result = matrix_multiply(500)
    computation_result = simple_computation(100000)

    print("✅ Example completed!")
```

**Run:**
```bash
# Test locally first
python examples/simple_example.py

# Run on cloud GPU
chisel python examples/simple_example.py
```

## Command Line Arguments

**File:** [`examples/args_example.py`](../examples/args_example.py)

```python
import argparse
from chisel import capture_trace

@capture_trace(trace_name="simple_ops", record_shapes=True)
def simple_operations(iterations: int):
    import torch

    print(f"🔥 Running simple operations for {iterations} iterations...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    for i in range(iterations):
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)

        result = torch.mm(a, b)
        result = torch.relu(result)
        result = result.sum()

        print(f"  Iteration {i + 1}/{iterations}: Result = {result.item():.4f}")

    print("✅ Operations completed!")
    return result.cpu().numpy() if hasattr(result, "cpu") else result

def main():
    parser = argparse.ArgumentParser(description="Chisel CLI Args Example")
    parser.add_argument("--iterations", type=int, help="Number of iterations (required)")

    args = parser.parse_args()

    assert args.iterations is not None, "❌ --iterations argument is required!"
    assert args.iterations > 0, "❌ --iterations must be greater than 0!"

    print("🚀 Starting Chisel CLI Args Example")
    print(f"Parameters: iterations={args.iterations}")

    result = simple_operations(args.iterations)
    print(f"Final result: {result}")

    print("✅ Example completed!")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
# With script arguments
chisel python examples/args_example.py --iterations 5

# With both chisel and script arguments
chisel python examples/args_example.py --app-name "args-test" --gpu 2 --iterations 10
```

## Requirements File

**File:** [`examples/requirements_example.py`](../examples/requirements_example.py)

This example demonstrates using custom requirements files. The CLI automatically detects and uses `requirements.txt`:

```python
from chisel import capture_trace

@capture_trace(trace_name="requirements_demo", record_shapes=True)
def requirements_demo():
    """Demonstrate that packages from requirements.txt are available."""
    import torch
    import numpy as np

    print("🔥 Testing packages from requirements.txt...")

    # Test PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 PyTorch device: {device}")

    x = torch.randn(10, 10, device=device)
    y = torch.randn(10, 10, device=device)
    result = torch.mm(x, y)

    print(f"✅ PyTorch matrix multiplication successful: {result.shape}")

    # Test NumPy
    np_array = np.random.rand(5, 5)
    np_result = np.dot(np_array, np_array.T)

    print(f"✅ NumPy matrix multiplication successful: {np_result.shape}")

    # Test matplotlib (import only, no plotting on backend)
    try:
        import matplotlib
        print(f"✅ Matplotlib available: {matplotlib.__version__}")
    except ImportError:
        print("⚠️ Matplotlib not available")

    return result.cpu().numpy()

if __name__ == "__main__":
    print("🚀 Starting Requirements Example")
    print("📦 This example demonstrates requirements.txt file usage with Chisel CLI")

    result = requirements_demo()
    print(f"Final result shape: {result.shape}")

    print("✅ Requirements example completed!")
```

**Run:**
```bash
# Uses requirements.txt by default
chisel python examples/requirements_example.py

# Use custom requirements file
chisel python examples/requirements_example.py --requirements "custom-requirements.txt"
```

**Requirements file example:**
```txt
# requirements.txt
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

## Inline Tracing

**File:** [`examples/specific_call.py`](../examples/specific_call.py)

Demonstrates wrapping external functions with `capture_trace` inline:

```python
from chisel import capture_trace

# External functions (not decorated)
def matrix_operations(size: int = 100):
    import torch
    
    print(f"🔢 Performing matrix operations with size {size}x{size}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    result = torch.mm(a, b)
    result = torch.relu(result).sum()
    
    print(f"✅ Matrix operations completed! Result: {result.item():.4f}")
    return result

def data_processing(num_samples: int = 1000):
    import torch
    
    print(f"📊 Processing {num_samples} data samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data = torch.randn(num_samples, 128, device=device)
    normalized = torch.nn.functional.normalize(data, dim=1)
    transformed = torch.sigmoid(normalized)
    final_result = transformed.mean(dim=0)
    
    print(f"✅ Data processing completed! Output shape: {final_result.shape}")
    return final_result.cpu().numpy()

def main():
    print("🚀 Starting Inline Tracing Example")
    
    # Inline tracing - wrap function calls directly
    print("\n🔧 Inline traced matrix operations:")
    result1 = capture_trace(
        trace_name="matrix_ops", 
        record_shapes=True, 
        profile_memory=True
    )(matrix_operations)(150)

    print("\n🔧 Inline traced data processing:")
    result2 = capture_trace(
        trace_name="data_processing", 
        record_shapes=True, 
        profile_memory=True
    )(data_processing)(2000)

    print("\n📊 All inline traces completed successfully!")
    print(f"Matrix result: {result1.item():.4f}")
    print(f"Data processing result shape: {result2.shape}")
    
    print("✅ Inline tracing example completed!")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
chisel python examples/specific_call.py
```

## Deep Learning

PyTorch model training example:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from chisel import capture_trace

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

@capture_trace(trace_name="data_generation")
def generate_data(n_samples=10000, n_features=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X = torch.randn(n_samples, n_features, device=device)
    weights = torch.randn(n_features, device=device)
    y = torch.mm(X, weights.unsqueeze(1)).squeeze()
    
    return X.cpu(), y.cpu()

@capture_trace(trace_name="training", profile_memory=True)
def train_model(X, y, epochs=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Training on: {device}")
    
    X, y = X.to(device), y.to(device)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = SimpleNN(X.shape[1], 256, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return model.cpu()

def main():
    print("🚀 Starting Deep Learning Example")
    
    X, y = generate_data(50000, 100)
    model = train_model(X, y, epochs=100)
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
chisel python deep_learning_example.py --app-name "training-job" --gpu 2
```

## Multi-GPU

Parallel processing with multiple GPUs:

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from chisel import capture_trace

@capture_trace(trace_name="multi_gpu_setup")
def setup_multi_gpu():
    if not torch.cuda.is_available():
        return False, 0
    
    n_gpus = torch.cuda.device_count()
    print(f"🎯 Found {n_gpus} GPU(s)")
    
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {name} ({memory:.1f}GB)")
    
    return True, n_gpus

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1000, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000)
        )
    
    def forward(self, x):
        return self.layers(x)

@capture_trace(trace_name="parallel_training", profile_memory=True)
def train_parallel(batch_size=1024, n_batches=50):
    has_cuda, n_gpus = setup_multi_gpu()
    if not has_cuda:
        print("❌ CUDA not available")
        return
    
    model = LargeModel()
    
    if n_gpus > 1:
        print(f"🚀 Using DataParallel with {n_gpus} GPUs")
        model = DataParallel(model)
    
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    for batch_idx in range(n_batches):
        inputs = torch.randn(batch_size, 1000, device='cuda')
        targets = torch.randn(batch_size, 1000, device='cuda')
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"   Batch [{batch_idx+1}/{n_batches}], Loss: {loss.item():.6f}")
    
    print("✅ Training completed!")
    return model

def main():
    print("🚀 Starting Multi-GPU Example")
    model = train_parallel()
    print("✅ Multi-GPU example completed!")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
# Use 4 GPUs for maximum parallelism
chisel python multi_gpu_example.py --app-name "multi-gpu-training" --gpu 4
```

## Best Practices

### Error Handling

```python
@capture_trace(trace_name="robust_function")
def robust_function(data):
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate input
        if data is None:
            raise ValueError("Input data cannot be None")
        
        # GPU computation
        tensor = torch.tensor(data, device=device)
        result = tensor * 2
        
        return result.cpu().numpy()
    except Exception as e:
        print(f"❌ Error in {robust_function.__name__}: {e}")
        raise
```

### Memory Management

```python
import torch
import gc

@capture_trace(trace_name="memory_efficient")
def memory_efficient_processing(data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Clear cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Process in chunks to avoid OOM
    chunk_size = 1000
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        
        # Process chunk
        chunk_tensor = torch.tensor(chunk, device=device)
        result = chunk_tensor.pow(2).sum()
        
        # Move result to CPU and cleanup
        results.append(result.cpu())
        del chunk_tensor
        
        # Periodic cleanup
        if i % (chunk_size * 10) == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return torch.stack(results)
```

### Environment Detection

```python
import os
from chisel import capture_trace

@capture_trace(trace_name="environment_aware")
def environment_aware_function():
    # Check execution environment
    if os.environ.get("CHISEL_BACKEND_RUN") == "1":
        print("🚀 Running on cloud GPU!")
        job_id = os.environ.get("CHISEL_JOB_ID")
        print(f"Job ID: {job_id}")
        
        # Cloud-specific optimizations
        torch.backends.cudnn.benchmark = True
    else:
        print("💻 Running locally")
        
        # Local-specific settings
        torch.set_num_threads(4)
    
    # Common code
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.randn(100, device=device)
```

## Performance Tips

1. **Use appropriate batch sizes** - Balance memory usage and throughput
2. **Keep data on GPU** - Minimize CPU-GPU transfers
3. **Enable memory profiling** - Use `profile_memory=True` to identify bottlenecks
4. **Clear GPU cache** - Use `torch.cuda.empty_cache()` for large workloads
5. **Use multiple GPUs** - Scale with `--gpu 2/4/8` for large models
6. **Test locally first** - Always validate with `python script.py` before cloud execution

**Next:** [Configuration](configuration.md) | [Troubleshooting](troubleshooting.md)