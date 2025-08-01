# Troubleshooting

Common issues and solutions when using Chisel CLI.

## Installation Issues

### ModuleNotFoundError: No module named 'chisel'

**Solutions:**
```bash
# Install in development mode
pip install -e .

# Check virtual environment
source .venv/bin/activate
pip install -e .

# Check installation
pip list | grep chisel
```

### externally-managed-environment Error

**Solution:**
```bash
# Use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Permission Errors

**Solutions:**
```bash
# Virtual environment (best)
python -m venv .venv
source .venv/bin/activate
pip install -e .

# User installation
pip install --user -e .
```

## Authentication Issues

### Browser Doesn't Open

**Solutions:**
```bash
# SSH environments - use port forwarding
ssh -L 8000:localhost:8000 user@remote-host

# Manual authentication
python -c "from chisel.auth import authenticate; authenticate()"
```

### Authentication Fails

**Solutions:**
```python
# Clear credentials and retry
from chisel.auth import clear_credentials
clear_credentials()

# Check backend connectivity
curl -v http://localhost:8000/health

# Verify backend URL
import os
print(os.environ.get('CHISEL_BACKEND_URL', 'http://localhost:8000'))
```

### API Key Errors

**Solutions:**
```python
# Re-authenticate
from chisel.auth import clear_credentials, authenticate
clear_credentials()
authenticate()

# Check credentials file
ls -la ~/.chisel/credentials.json
```

## Runtime Issues

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# Reduce batch size
batch_size = 32  # Instead of 256

# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Process in chunks
for i in range(0, len(data), small_batch_size):
    chunk = data[i:i+small_batch_size]
    result = process(chunk)

# Use larger GPU
app = ChiselApp("my-app", gpu=GPUType.A100_80GB_4)
```

### Job Submission Fails

**Error:** `❌ Work upload failed`

**Solutions:**
```bash
# Check file size (keep under 100MB)
du -sh .

# Add .gitignore patterns
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".venv/" >> .gitignore

# Use smaller upload directory
app = ChiselApp("my-app", upload_dir="./src")
```

### Script Not Found

**Error:** `❌ Script not found in uploaded archive`

**Solutions:**
```python
# Ensure script is in upload directory
app = ChiselApp("my-app", upload_dir=".")

# Use relative paths
# Run from project root: chisel python src/script.py
```

### Argument Passing Issues

**Error:** `error: unrecognized arguments`

**Solutions:**
```bash
# Correct format
chisel python script.py --arg value

# Debug arguments
python -c "import sys; print(sys.argv)"

# Use quotes for complex args
chisel python script.py --text "hello world"
```

## Performance Issues

### Slow Execution

**Solutions:**
```python
import torch

# Check device placement
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name()}")

# Ensure tensors on GPU
tensor = tensor.cuda()

# Optimize data transfer - move to GPU once
data = data.cuda()

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    result = model(data)
```

### Memory Leaks

**Solutions:**
```python
import torch
import gc

# Explicit cleanup
del large_tensors
torch.cuda.empty_cache()
gc.collect()

# Use context managers
with torch.no_grad():
    result = model(data)

# Monitor memory
def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU Memory: {allocated:.2f}GB")
```

## Dependency Issues

### Missing Dependencies

**Solutions:**
```txt
# Add requirements.txt
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

```python
# Install in code (temporary)
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import my_package
except ImportError:
    install_package("my_package")
    import my_package
```

### PyTorch Not Found

**Note:** PyTorch is installed automatically on cloud GPUs. If missing, this is a backend issue - please report it.

## Network Issues

### Connection Timeouts

**Solutions:**
```bash
# Check backend status
curl http://localhost:8000/health

# Use alternative backend
export CHISEL_BACKEND_URL="https://backup-api.herdora.com"

# Configure proxy if needed
export HTTP_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1
```

### Streaming Issues

**Problem:** Real-time output not showing or connection drops during streaming.

**Solutions:**
```bash
# Check network stability
ping -c 5 your-backend-url

# Try with shorter timeout for testing
# (Note: This is handled automatically by Chisel)

# Check firewall settings for streaming connections
# Some corporate firewalls block streaming responses
```

## Debugging

### Enable Debug Mode

```python
import logging
import os

# Enable debug logging
if os.getenv('CHISEL_DEBUG'):
    logging.basicConfig(level=logging.DEBUG)
    torch.autograd.set_detect_anomaly(True)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

### Minimal Reproduction

```python
from chisel import ChiselApp, GPUType

app = ChiselApp("debug-test", gpu=GPUType.A100_80GB_1)

@app.capture_trace(trace_name="debug_test")
def minimal_test():
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Your minimal failing code here
    x = torch.tensor([1.0, 2.0], device='cuda')
    return x.cpu().numpy()

if __name__ == "__main__":
    result = minimal_test()
    print(f"Result: {result}")
```

### System Information

```python
def print_system_info():
    import torch, sys, os
    
    print("=== System Information ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"Chisel activated: {os.environ.get('CHISEL_ACTIVATED')}")
    print(f"Job ID: {os.environ.get('CHISEL_JOB_ID')}")

print_system_info()
```

## Common Error Patterns

| Error                   | Cause                   | Solution                            |
| ----------------------- | ----------------------- | ----------------------------------- |
| `ModuleNotFoundError`   | Missing dependency      | Add to requirements.txt             |
| `CUDA out of memory`    | Insufficient GPU memory | Reduce batch size or use larger GPU |
| `Script not found`      | Wrong upload directory  | Check upload_dir parameter          |
| `Authentication failed` | Invalid credentials     | Re-authenticate                     |
| `Connection timeout`    | Network issues          | Check connectivity                  |

## Getting Help

### Log Analysis
1. Note Job ID from command output
2. Visit provided URL for job details
3. Check stdout/stderr for error messages

### Contact & Support

- **📧 Email**: [contact@herdora.com](mailto:contact@herdora.com) - Direct support
- **🐛 GitHub Issues**: Report bugs and request features
- **💬 GitHub Discussions**: Ask questions and share ideas

### Reporting Issues

When contacting support or reporting issues, include:
1. Python version and OS
2. Complete error output
3. Minimal reproduction code
4. System info (use script above)
5. Job ID (if applicable)

```bash
# Collect debug info
python -c "
import sys, torch, os
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Chisel env: {dict((k,v) for k,v in os.environ.items() if k.startswith(\"CHISEL\"))}')
"
```

**Next:** [Development](development.md) | [API Reference](api-reference.md)