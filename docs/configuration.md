# Configuration

Configuring Chisel CLI for optimal performance.

## GPU Types

| GPU Type              | GPUs         | Memory | Use Case               |
| --------------------- | ------------ | ------ | ---------------------- |
| `GPUType.A100_80GB_1` | 1x A100-80GB | 80GB   | Development, inference |
| `GPUType.A100_80GB_2` | 2x A100-80GB | 160GB  | Medium training        |
| `GPUType.A100_80GB_4` | 4x A100-80GB | 320GB  | Large models           |
| `GPUType.A100_80GB_8` | 8x A100-80GB | 640GB  | Massive models         |

### Selection Guidelines

**Single GPU:** Small models, inference, development
```python
app = ChiselApp("inference-app", gpu=GPUType.A100_80GB_1)
```

**Dual GPU:** Medium models, balanced performance/cost
```python
app = ChiselApp("training-app", gpu=GPUType.A100_80GB_2)
```

**Quad GPU:** Large models, high-throughput training
```python
app = ChiselApp("large-training", gpu=GPUType.A100_80GB_4)
```

**Octa GPU:** Massive models, maximum throughput
```python
app = ChiselApp("massive-training", gpu=GPUType.A100_80GB_8)
```

## Environment Variables

| Variable             | Purpose                     | Set By           |
| -------------------- | --------------------------- | ---------------- |
| `CHISEL_ACTIVATED`   | Activates GPU functionality | `chisel` command |
| `CHISEL_BACKEND_URL` | Backend URL                 | User (optional)  |
| `CHISEL_API_KEY`     | Authentication token        | Auth system      |
| `CHISEL_BACKEND_RUN` | Running on backend          | Backend system   |
| `CHISEL_JOB_ID`      | Current job ID              | Backend system   |

### Custom Backend

```bash
export CHISEL_BACKEND_URL="https://api.herdora.com"
chisel python my_script.py
```

## Application Configuration

### ChiselApp Options

```python
app = ChiselApp(
    name="my-app",              # Required: App identifier
    upload_dir="./src",         # Optional: Upload directory (default: ".")
    gpu=GPUType.A100_80GB_2     # Optional: GPU configuration
)
```

### Upload Directory

```python
# Upload current directory (default)
app = ChiselApp("app", upload_dir=".")

# Upload specific directory
app = ChiselApp("app", upload_dir="./src")

# Upload parent directory
app = ChiselApp("app", upload_dir="../")
```

**Best practices:**
- Keep under 100MB
- Use `.gitignore` to exclude unnecessary files
- Include only required code and data

### Trace Configuration

```python
@app.capture_trace(
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

@app.capture_trace(trace_name="memory_optimized")
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

@app.capture_trace(trace_name="multi_gpu")
def setup_multi_gpu_model(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    return model.cuda()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

@app.capture_trace(trace_name="mixed_precision")
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
import os

# Disable Chisel for local development
if os.getenv('DEVELOPMENT'):
    pass  # ChiselApp will be in pass-through mode

app = ChiselApp("dev-app")

@app.capture_trace()  # No-op locally
def my_function():
    pass
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

Credentials stored in `~/.chisel/credentials.json` with restricted permissions.

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

app = ChiselApp("configured-app")
```

## Best Practices

1. **Start small** - Begin with single GPU, scale up as needed
2. **Profile first** - Use tracing to identify bottlenecks
3. **Manage memory** - Monitor usage, process in chunks
4. **Optimize batches** - Find optimal batch sizes
5. **Separate environments** - Different configs for dev/prod

**Next:** [Troubleshooting](troubleshooting.md) | [Development](development.md)