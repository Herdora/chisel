# Examples

Working examples for different Chisel CLI use cases.

## Quick Reference

| Example                                      | Description           | Command                                                 |
| -------------------------------------------- | --------------------- | ------------------------------------------------------- |
| [Basic Usage](#basic-usage)                  | Matrix operations     | `chisel python examples/simple_example.py`              |
| [Command Line Args](#command-line-arguments) | Script with arguments | `chisel python examples/args_example.py --iterations 5` |
| [Deep Learning](#deep-learning)              | PyTorch training      | See below                                               |
| [Multi-GPU](#multi-gpu)                      | Parallel processing   | See below                                               |

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
    result = simple_operations(args.iterations)
    print("✅ Example completed!")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
chisel python examples/args_example.py --iterations 5
```

## Deep Learning

PyTorch model training example.

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
chisel python deep_learning_example.py
```

## Multi-GPU

Parallel processing with multiple GPUs.

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
chisel python multi_gpu_example.py
```

## Best Practices

**Error handling:**
```python
@capture_trace(trace_name="robust")
def robust_function(data):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # GPU code here
    except Exception as e:
        print(f"Error: {e}")
        # Fallback or re-raise
        raise
```

**Memory management:**
```python
import torch
import gc

# Clear cache when needed
torch.cuda.empty_cache()
gc.collect()

# Process in batches for large data
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    result = process(batch)
    # Move to CPU to free GPU memory
    results.append(result.cpu())
```

**Multiple functions:**
```python
@capture_trace(trace_name="preprocess")
def preprocess(data): pass

@capture_trace(trace_name="train")
def train(data): pass

@capture_trace(trace_name="evaluate")
def evaluate(data): pass
```

## Performance Tips

1. **Batch processing** - Avoid memory issues with large datasets
2. **Keep data on GPU** - Minimize CPU-GPU transfers
3. **Use appropriate batch sizes** - Balance memory and speed
4. **Profile memory usage** - Use `profile_memory=True`
5. **Multiple GPUs** - Use `DataParallel` for large models

**Next:** [Configuration](configuration.md) | [Troubleshooting](troubleshooting.md)