# Configuration

Configuring Chisel CLI for optimal performance.

## GPU Types

When using the `chisel` command, you'll be prompted to select a GPU configuration:

| Option | GPU Configuration | Memory | Use Case               |
| ------ | ----------------- | ------ | ---------------------- |
| 1      | 1x A100-80GB      | 80GB   | Development, inference |
| 2      | 2x A100-80GB      | 160GB  | Medium training        |
| 4      | 4x A100-80GB      | 320GB  | Large models           |
| 8      | 8x A100-80GB      | 640GB  | Massive models         |

### Selection Guidelines

**Single GPU:** Small models, inference, development
```bash
chisel python my_script.py
# Select option 1 when prompted
```

**Dual GPU:** Medium models, balanced performance/cost
```bash
chisel python my_script.py
# Select option 2 when prompted
```

**Quad GPU:** Large models, high-throughput training
```bash
chisel python my_script.py
# Select option 4 when prompted
```

**Octa GPU:** Massive models, maximum throughput
```bash
chisel python my_script.py
# Select option 8 when prompted
```

## Environment Variables

| Variable             | Purpose              | Set By          |
| -------------------- | -------------------- | --------------- |
| `CHISEL_BACKEND_RUN` | Running on backend   | Backend system  |
| `CHISEL_JOB_ID`      | Current job ID       | Backend system  |
| `CHISEL_BACKEND_URL` | Override backend URL | User (optional) |
| `CHISEL_API_KEY`     | Authentication token | Auth system     |

### Custom Backend

```bash
export CHISEL_BACKEND_URL="https://api.herdora.com"
chisel python my_script.py
```

## Interactive Configuration

When you run `chisel python script.py`, the CLI will prompt you for:

### App Name
```bash
📝 App name (for job tracking): my-training-job
```
- Used for job tracking and identification
- Should be descriptive of your workload

### Upload Directory
```bash
📁 Upload directory (default: current directory): ./src
```
- Directory containing your code and data
- Should include your script and dependencies
- Keep under 100MB for optimal upload speed

### Requirements File
```bash
📋 Requirements file (default: requirements.txt): requirements.txt
```
- File listing Python dependencies
- Should include PyTorch and other required packages

### GPU Configuration
```bash
🎮 GPU Options:
  1. A100-80GB:1 - Single GPU - Development, inference
  2. A100-80GB:2 - 2x GPUs - Medium training
  4. A100-80GB:4 - 4x GPUs - Large models
  8. A100-80GB:8 - 8x GPUs - Massive models

🎮 Select GPU configuration (1-8, default: 1): 2
```

## Trace Configuration

```python
@capture_trace(
    trace_name="my_operation",
    record_shapes=True,
    profile_memory=True
)
def my_function():
    pass
```

**Trace Options:**

| Option                | Description              | Overhead | Best For     |
| --------------------- | ------------------------ | -------- | ------------ |
| `record_shapes=True`  | Record tensor dimensions | Low      | Debugging    |
| `profile_memory=True` | Track memory allocation  | Medium   | Optimization |
| `with_stack=True`     | Include call stack       | Medium   | Debugging    |

## Performance Optimization

### Memory Management

```python
import torch
import gc

@capture_trace(trace_name="memory_optimized")
def memory_efficient_processing(data):
    torch.cuda.empty_cache()  # Clear cache
    
    # Process in chunks
    chunk_size = 1000
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        result = process_chunk(chunk)
        results.append(result.cpu())  # Move to CPU
        
        del chunk, result
        if i % (chunk_size * 10) == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return torch.cat(results)
```

### Multi-GPU Setup

```python
from torch.nn.parallel import DataParallel

@capture_trace(trace_name="multi_gpu")
def setup_multi_gpu_model(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    return model.cuda()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

@capture_trace(trace_name="mixed_precision")
def train_with_mixed_precision(model, data_loader):
    scaler = GradScaler()
    
    for inputs, targets in data_loader:
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Batch Size Optimization

```python
def find_optimal_batch_size(model, sample_input, gpu_memory_gb=80):
    batch_size = 32
    
    while batch_size <= 1024:
        try:
            batch_input = sample_input.repeat(batch_size, 1, 1, 1)
            with torch.no_grad():
                output = model(batch_input)
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            if memory_used > gpu_memory_gb * 0.8:
                return batch_size // 2
                
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                return batch_size // 2
            raise e
    
    return batch_size
```

## Development Configuration

### Local Development

```python
# Your script works both locally and on cloud
@capture_trace(trace_name="my_function")
def my_function():
    # This runs locally with 'python script.py'
    # This runs on GPU with 'chisel python script.py'
    pass

# Test locally
if __name__ == "__main__":
    result = my_function()
    print(f"Result: {result}")
```

### Debug Mode

```python
import logging
import os

if os.getenv('CHISEL_DEBUG'):
    logging.basicConfig(level=logging.DEBUG)
    torch.autograd.set_detect_anomaly(True)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

### Testing

```python
def test_with_chisel():
    on_cloud = os.environ.get("CHISEL_BACKEND_RUN") == "1"
    
    if on_cloud:
        assert torch.cuda.is_available()
    else:
        print("Running local test")
    
    result = my_gpu_function()
    assert result is not None
```

## Authentication

### Credential Storage

Credentials stored in `~/.chisel/credentials.json` with restricted permissions (directory: 0o700, file: 0o600).

### Manual Authentication

```python
from chisel.auth import authenticate, clear_credentials, is_authenticated

# Check status
if not is_authenticated():
    print("Authentication required")

# Manual auth
api_key = authenticate("https://api.herdora.com")

# Clear credentials
clear_credentials()
```

## PyTorch Optimization

```python
import torch
import os

# A100 optimization
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Memory management
torch.cuda.set_per_process_memory_fraction(0.9)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Flash attention (PyTorch 2.0+)
torch.backends.cuda.enable_flash_sdp(True)
```

## Environment Files

```bash
# .env
CHISEL_BACKEND_URL=https://api.herdora.com
TORCH_CUDA_ARCH_LIST=8.0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

```python
from dotenv import load_dotenv
load_dotenv()

# Your script will use the configured backend
```

## Best Practices

1. **Start small** - Begin with single GPU, scale up as needed
2. **Profile first** - Use tracing to identify bottlenecks
3. **Manage memory** - Monitor usage, process in chunks
4. **Optimize batches** - Find optimal batch sizes
5. **Separate environments** - Different configs for dev/prod
6. **Test locally** - Ensure your script works with `python script.py`
7. **Use descriptive names** - App names help with job tracking

**Next:** [Troubleshooting](troubleshooting.md) | [Development](development.md)
