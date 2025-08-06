# Configuration

Configuring Chisel CLI for optimal performance with the new simplified API.

## CLI Configuration

### Interactive Mode (Default)

When you run `chisel python script.py` without flags, the CLI prompts for configuration:

```bash
chisel python my_script.py
```

**Interactive prompts:**
1. **App name**: Job identifier for tracking (e.g., "training-job")
2. **Upload directory**: Which folder to upload (default: current directory)
3. **Requirements file**: Python dependencies (default: requirements.txt)
4. **GPU configuration**: Choose 1, 2, 4, or 8x A100-80GB GPUs

### Command Line Flags

Skip interactive mode by providing configuration flags:

```bash
# Basic configuration
chisel python script.py --app-name "my-job" --gpu 2

# Full configuration
chisel python train.py \
  --app-name "training-job" \
  --upload-dir "./src" \
  --requirements "requirements.txt" \
  --gpu 4
```

**Available flags:**

| Flag             | Short | Description            | Default                 | Example                     |
| ---------------- | ----- | ---------------------- | ----------------------- | --------------------------- |
| `--app-name`     | `-a`  | Job name for tracking  | Interactive prompt      | `--app-name "training-job"` |
| `--upload-dir`   | `-d`  | Directory to upload    | Current directory (`.`) | `--upload-dir "./src"`      |
| `--requirements` | `-r`  | Requirements file      | `requirements.txt`      | `--requirements "dev.txt"`  |
| `--gpu`          | `-g`  | GPU count (1,2,4,8)    | 1                       | `--gpu 4`                   |
| `--interactive`  | `-i`  | Force interactive mode | Auto-detect             | `--interactive`             |

### Mixed Configuration

Combine flags with interactive prompts:

```bash
# Specify GPU count, prompt for other options
chisel python script.py --gpu 4

# Specify app name and GPU, prompt for upload directory
chisel python script.py --app-name "my-job" --gpu 2
```

## GPU Types

When prompted or using `--gpu` flag, choose from:

| Option | GPU Configuration | Memory | Use Case               | Cost    |
| ------ | ----------------- | ------ | ---------------------- | ------- |
| 1      | 1x A100-80GB      | 80GB   | Development, inference | Lowest  |
| 2      | 2x A100-80GB      | 160GB  | Medium training        | Low     |
| 4      | 4x A100-80GB      | 320GB  | Large models           | Medium  |
| 8      | 8x A100-80GB      | 640GB  | Massive models         | Highest |

### GPU Selection Guidelines

**Single GPU (--gpu 1):**
- Small models (< 10B parameters)
- Inference workloads
- Development and testing
- Cost-conscious training

**Dual GPU (--gpu 2):**
- Medium models (10B-50B parameters)
- Balanced performance/cost
- Multi-task training

**Quad GPU (--gpu 4):**
- Large models (50B-100B parameters)
- High-throughput training
- Complex multi-GPU algorithms

**Octa GPU (--gpu 8):**
- Massive models (100B+ parameters)
- Maximum parallelism
- Research-scale workloads

## capture_trace Configuration

Configure tracing behavior with decorator parameters:

```python
from chisel import capture_trace

@capture_trace(
    trace_name="my_operation",
    record_shapes=True,
    profile_memory=True
)
def my_function():
    pass
```

### Trace Parameters

| Parameter        | Type | Description              | Default       | Overhead |
| ---------------- | ---- | ------------------------ | ------------- | -------- |
| `trace_name`     | str  | Operation identifier     | Function name | None     |
| `record_shapes`  | bool | Record tensor dimensions | False         | Low      |
| `profile_memory` | bool | Track memory allocation  | False         | Medium   |

**Examples:**

```python
# Minimal tracing
@capture_trace()
def simple_function():
    pass

# Named trace
@capture_trace(trace_name="preprocessing")
def preprocess_data():
    pass

# Full profiling
@capture_trace(
    trace_name="training",
    record_shapes=True,
    profile_memory=True
)
def train_model():
    pass
```

## Environment Variables

| Variable             | Purpose              | Set By      | Example                   |
| -------------------- | -------------------- | ----------- | ------------------------- |
| `CHISEL_BACKEND_URL` | Override backend URL | User        | `https://api.herdora.com` |
| `CHISEL_BACKEND_RUN` | Running on backend   | Backend     | `1`                       |
| `CHISEL_JOB_ID`      | Current job ID       | Backend     | `job_123456`              |
| `CHISEL_API_KEY`     | Authentication token | Auth system | `sk_...`                  |

### Custom Backend

```bash
# Use custom backend
export CHISEL_BACKEND_URL="https://my-backend.com"
chisel python my_script.py

# Or inline
CHISEL_BACKEND_URL="https://my-backend.com" chisel python my_script.py
```

### Environment Detection

```python
import os

# Check execution environment
if os.environ.get("CHISEL_BACKEND_RUN") == "1":
    print("🚀 Running on cloud GPU!")
    job_id = os.environ.get("CHISEL_JOB_ID")
    print(f"Job ID: {job_id}")
else:
    print("💻 Running locally")
```

## Upload Directory Configuration

### Default Behavior

```bash
# Uploads current directory
chisel python script.py
```

### Custom Upload Directory

```bash
# Upload specific directory
chisel python script.py --upload-dir "./src"

# Upload parent directory
chisel python script.py --upload-dir "../project"
```

### Directory Structure

**Recommended structure:**
```
project/
├── src/                 # Code directory
│   ├── main.py         # Main script
│   └── utils.py        # Helper modules
├── requirements.txt    # Dependencies
├── data/              # Data files (may contain large files >1GB)
└── .gitignore         # Exclude patterns
```

**Upload optimization:**
```bash
# Upload only source code
chisel python src/main.py --upload-dir "./src"

# Upload entire project
chisel python main.py --upload-dir "."
```

**⚠️ Large File Upload Warning:**  
When uploading projects with large files (>1GB), **do not refresh the browser page** during the upload process. Chisel automatically caches large files, and refreshing can interrupt this process, causing uploads to fail or restart.

**Upload time estimates:**
- **1GB file**: First upload ~2-5 minutes, subsequent uploads ~10-30 seconds  
- **5GB file**: First upload ~10-25 minutes, subsequent uploads ~30-60 seconds
- **10GB+ files**: First upload ~20-50+ minutes, subsequent uploads ~1-2 minutes

*Times vary based on internet connection speed and server load.*

### Exclusion Patterns

Files automatically excluded from upload:
- `.venv/`, `venv/`, `__pycache__/`
- `.env`, `.git/`
- Files matching `.gitignore` patterns

## Requirements Configuration

### Default Requirements

```bash
# Uses requirements.txt automatically
chisel python script.py
```

### Custom Requirements File

```bash
# Use custom requirements file
chisel python script.py --requirements "dev.txt"

# Requirements in subdirectory
chisel python script.py --requirements "requirements/production.txt"
```

### Requirements File Examples

**Basic requirements.txt:**
```txt
torch>=2.0.0
numpy>=1.21.0
transformers>=4.20.0
```

**Development requirements (dev.txt):**
```txt
torch>=2.0.0
numpy>=1.21.0
transformers>=4.20.0
jupyter>=1.0.0
matplotlib>=3.5.0
pytest>=7.0.0
```

**Production requirements:**
```txt
torch==2.0.1
numpy==1.24.3
transformers==4.30.2
# Pinned versions for reproducibility
```

## Performance Configuration

### Memory Optimization

```python
import torch
import gc
from chisel import capture_trace

@capture_trace(trace_name="memory_optimized", profile_memory=True)
def memory_efficient_training():
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Set memory fraction
    torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Enable memory efficient attention
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Your training code here
    pass
```

### Multi-GPU Configuration

```python
import torch
from torch.nn.parallel import DataParallel
from chisel import capture_trace

@capture_trace(trace_name="multi_gpu_training")
def setup_multi_gpu_training():
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    return model.cuda()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler
from chisel import capture_trace

@capture_trace(trace_name="mixed_precision_training")
def train_with_mixed_precision():
    scaler = GradScaler()
    
    for batch in dataloader:
        with autocast():
            outputs = model(batch)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Development Configuration

### Local Testing

```python
from chisel import capture_trace

@capture_trace(trace_name="development")
def my_function():
    import torch
    
    # Works both locally and on cloud
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.randn(100, device=device)

# Test locally first
if __name__ == "__main__":
    result = my_function()
    print(f"✅ Local test: {result.shape}")
```

### Debug Configuration

```python
import os
import logging
from chisel import capture_trace

# Enable debug mode
if os.getenv('DEBUG'):
    logging.basicConfig(level=logging.DEBUG)
    torch.autograd.set_detect_anomaly(True)

@capture_trace(trace_name="debug_function", record_shapes=True)
def debug_function():
    # Debug-enabled function
    pass
```

### Testing Configuration

```python
import pytest
from chisel import capture_trace

@capture_trace(trace_name="test_function")
def compute_result():
    return 42

def test_local_execution():
    """Test function works locally."""
    result = compute_result()
    assert result == 42

def test_cloud_execution():
    """Test function works on cloud (when CHISEL_BACKEND_RUN=1)."""
    result = compute_result()
    assert result == 42
```

## Authentication Configuration

### Automatic Authentication

```bash
# First run opens browser automatically
chisel python script.py
```

### Manual Authentication Management

```bash
# Check authentication status
python -c "from chisel.auth import is_authenticated; print('Authenticated:', is_authenticated())"

# Clear credentials
chisel --logout

# Force re-authentication
chisel python script.py  # Will prompt for login
```

### Credential Storage

- Location: `~/.chisel/credentials.json`
- Directory permissions: 0o700 (user only)
- File permissions: 0o600 (user read/write only)

## Best Practices

### 1. Start Small, Scale Up

```bash
# Development: Single GPU
chisel python script.py --app-name "dev-test" --gpu 1

# Production: Scale as needed
chisel python script.py --app-name "production-run" --gpu 4
```

### 2. Use Descriptive Names

```bash
# Good app names
chisel python train.py --app-name "bert-fine-tuning-v2"
chisel python inference.py --app-name "batch-inference-prod"

# Poor app names
chisel python script.py --app-name "test"
chisel python script.py --app-name "run1"
```

### 3. Optimize Upload Size

```bash
# Upload only necessary files
chisel python src/main.py --upload-dir "./src"

# Use .gitignore to exclude large files
echo "*.pkl" >> .gitignore
echo "data/" >> .gitignore
```

### 4. Test Locally First

```bash
# Always test locally first
python script.py

# Then run on cloud
chisel python script.py
```

### 5. Use Appropriate Tracing

```python
# Minimal for production
@capture_trace(trace_name="production_inference")

# Full profiling for optimization
@capture_trace(
    trace_name="training_optimization",
    record_shapes=True,
    profile_memory=True
)
```

**Next:** [Troubleshooting](troubleshooting.md) | [Development](development.md)