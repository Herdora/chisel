# Troubleshooting

Common issues and solutions when using Chisel CLI with the simplified API.

## Installation Issues

### ModuleNotFoundError: No module named 'chisel'

**Solutions:**
```bash
# Install from GitHub
pip install git+https://github.com/Herdora/chisel.git@dev

# Check installation
pip list | grep chisel
chisel --version

# Verify import works
python -c "from chisel import capture_trace; print('✅ Import successful')"
```

### externally-managed-environment Error

**Solution:**
```bash
# Use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR .venv\Scripts\activate  # Windows

pip install git+https://github.com/Herdora/chisel.git@dev
```

### Permission Errors

**Solutions:**
```bash
# Virtual environment (best)
python -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/Herdora/chisel.git@dev

# User installation (alternative)
pip install --user git+https://github.com/Herdora/chisel.git@dev
```

## CLI Issues

### Command Not Found: chisel

**Solutions:**
```bash
# Check if chisel is in PATH
which chisel

# If not found, try full path
python -m chisel python script.py

# Or reinstall
pip uninstall chisel-cli
pip install git+https://github.com/Herdora/chisel.git@dev
```

### Interactive Mode Not Working

**Problem:** CLI doesn't prompt for configuration

**Solutions:**
```bash
# Force interactive mode
chisel python script.py --interactive

# Check terminal supports input
python -c "input('Test: ')"

# Use explicit flags instead
chisel python script.py --app-name "test" --gpu 1
```

### Script Not Found Error

**Error:** `❌ Script not found in uploaded archive`

**Solutions:**
```bash
# Ensure script is in upload directory
ls -la my_script.py  # Check script exists

# Use correct upload directory
chisel python script.py --upload-dir "."

# Use relative path
chisel python src/script.py --upload-dir "."
```

## Authentication Issues

### Browser Doesn't Open

**Solutions:**
```bash
# SSH environments - use port forwarding
ssh -L 8000:localhost:8000 user@remote-host

# Manual authentication
python -c "from chisel.auth import authenticate; authenticate()"

# Check if browser command works
python -c "import webbrowser; webbrowser.open('https://google.com')"
```

### Authentication Fails

**Solutions:**
```bash
# Clear credentials and retry
chisel --logout
chisel python script.py  # Will re-authenticate

# Check backend connectivity
curl -v http://localhost:8000/health

# Verify backend URL
python -c "import os; print(os.environ.get('CHISEL_BACKEND_URL', 'http://localhost:8000'))"
```

### API Key Errors

**Error:** `❌ Authentication failed. Unable to get valid API key.`

**Solutions:**
```bash
# Re-authenticate
chisel --logout
chisel python script.py

# Check credentials file
ls -la ~/.chisel/credentials.json

# Manual cleanup
rm ~/.chisel/credentials.json
chisel python script.py  # Will re-authenticate
```

## Runtime Issues

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**

1. **Reduce batch size:**
```python
# Before
batch_size = 256

# After
batch_size = 32  # or smaller
```

2. **Clear GPU cache:**
```python
import torch

@capture_trace(trace_name="memory_efficient")
def my_function():
    torch.cuda.empty_cache()  # Clear at start
    
    # Your code here
    
    torch.cuda.empty_cache()  # Clear at end
```

3. **Process in chunks:**
```python
@capture_trace(trace_name="chunked_processing")
def process_large_data(data):
    import torch
    
    chunk_size = 1000
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        result = process_chunk(chunk)
        results.append(result.cpu())  # Move to CPU
        
        # Cleanup
        del chunk, result
        if i % (chunk_size * 10) == 0:
            torch.cuda.empty_cache()
    
    return torch.cat(results)
```

4. **Use larger GPU:**
```bash
# Use more GPUs for more memory
chisel python script.py --gpu 4  # 320GB total
chisel python script.py --gpu 8  # 640GB total
```

### Job Submission Fails

**Error:** `❌ Work upload failed`

**Solutions:**

1. **Check file size:**
```bash
# Check upload directory size
du -sh .
du -sh * | sort -h

# Keep under 100MB for optimal upload
```

2. **Optimize upload directory:**
```bash
# Add to .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".venv/" >> .gitignore
echo "data/" >> .gitignore
echo "*.pkl" >> .gitignore

# Upload only source code
chisel python src/main.py --upload-dir "./src"
```

3. **Check network connectivity:**
```bash
# Test backend connection
curl -v http://localhost:8000/health

# Check upload timeout (handled automatically)
```

### Import Errors on Cloud

**Error:** `ModuleNotFoundError` on cloud but works locally

**Solutions:**

1. **Check requirements.txt:**
```txt
# Ensure all dependencies are listed
torch>=2.0.0
numpy>=1.21.0
your-custom-package>=1.0.0
```

2. **Use correct requirements file:**
```bash
# Specify custom requirements
chisel python script.py --requirements "dev.txt"

# Check file exists
ls -la requirements.txt
```

3. **Install missing packages inline:**
```python
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import my_package
except ImportError:
    print(f"Installing my_package...")
    install_package("my_package")
    import my_package
```

### Argument Parsing Issues

**Error:** `error: unrecognized arguments`

**Solutions:**

1. **Separate chisel and script arguments:**
```bash
# Correct: chisel flags first, then script arguments
chisel python train.py --app-name "job" --gpu 2 --epochs 100

# Incorrect: mixed arguments
chisel python train.py --epochs 100 --gpu 2  # gpu not recognized by script
```

2. **Use quotes for complex arguments:**
```bash
# For arguments with spaces
chisel python script.py --text "hello world"

# For JSON arguments
chisel python script.py --config '{"lr": 0.001, "batch_size": 32}'
```

3. **Debug argument parsing:**
```python
import sys
print("Arguments received:", sys.argv)

# In your script
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int)
args = parser.parse_args()
print("Parsed arguments:", args)
```

## Performance Issues

### Slow Execution

**Solutions:**

1. **Verify GPU usage:**
```python
import torch

@capture_trace(trace_name="gpu_check")
def check_gpu():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name()}")
        
        # Test GPU computation
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.mm(x, x)
        print(f"GPU computation successful: {y.shape}")
```

2. **Optimize data transfer:**
```python
@capture_trace(trace_name="optimized_transfer")
def optimize_data_transfer():
    import torch
    
    # Move data to GPU once
    data = torch.tensor(data, device='cuda')
    
    # Avoid repeated CPU-GPU transfers
    # Bad: result.cpu().cuda()
    # Good: keep on GPU until final result
    
    return result.cpu()  # Only at the end
```

3. **Use mixed precision:**
```python
from torch.cuda.amp import autocast

@capture_trace(trace_name="mixed_precision")
def mixed_precision_training():
    with autocast():
        # Automatic mixed precision
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    return loss
```

### Memory Leaks

**Solutions:**

1. **Explicit cleanup:**
```python
import torch
import gc

@capture_trace(trace_name="memory_managed")
def memory_managed_function():
    # Your computation
    large_tensor = torch.randn(10000, 10000, device='cuda')
    result = large_tensor.sum()
    
    # Explicit cleanup
    del large_tensor
    torch.cuda.empty_cache()
    gc.collect()
    
    return result
```

2. **Use context managers:**
```python
@capture_trace(trace_name="context_managed")
def context_managed_function():
    with torch.no_grad():  # Disable gradients
        result = model(data)
    
    return result
```

3. **Monitor memory usage:**
```python
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

@capture_trace(trace_name="memory_monitored")
def memory_monitored_function():
    print_gpu_memory()  # Before
    
    # Your computation
    result = compute_something()
    
    print_gpu_memory()  # After
    return result
```

## Network Issues

### Connection Timeouts

**Solutions:**

1. **Check backend status:**
```bash
# Test backend connectivity
curl http://localhost:8000/health

# Check response time
time curl http://localhost:8000/health
```

2. **Use alternative backend:**
```bash
# Set custom backend
export CHISEL_BACKEND_URL="https://backup-api.herdora.com"
chisel python script.py
```

3. **Configure proxy:**
```bash
# If behind corporate proxy
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1

chisel python script.py
```

### Upload Failures

**Solutions:**

1. **Check file size limits:**
```bash
# Check upload size
du -sh .

# Compress if needed
tar -czf project.tar.gz src/
ls -lh project.tar.gz
```

2. **Use smaller upload directory:**
```bash
# Upload only necessary files
chisel python src/main.py --upload-dir "./src"
```

3. **Check network stability:**
```bash
# Test network stability
ping -c 10 your-backend-url

# Monitor network during upload
iftop  # or similar network monitoring tool
```

4. **Large file caching issues:**
```bash
# If upload fails with large files (>1GB)
# ⚠️ IMPORTANT: Do NOT refresh browser during upload!
# Refreshing interrupts caching and can cause failures

# Check for large files
find . -size +1G -type f

# Remove large files from upload if not needed
chisel python script.py --upload-dir "./src"  # exclude data directory
```

**Expected upload times for large files:**
- **1GB**: ~2-5 minutes (first) → ~10-30 seconds (cached)
- **5GB**: ~10-25 minutes (first) → ~30-60 seconds (cached)  
- **10GB+**: ~20-50+ minutes (first) → ~1-2 minutes (cached)

If uploads are taking significantly longer, check your internet connection or try during off-peak hours.

## Debugging

### Enable Debug Mode

```python
import os
import logging

# Enable debug logging
if os.getenv('CHISEL_DEBUG'):
    logging.basicConfig(level=logging.DEBUG)
    
    # Enable PyTorch debugging
    import torch
    torch.autograd.set_detect_anomaly(True)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

```bash
# Run with debug mode
CHISEL_DEBUG=1 chisel python script.py
```

### Minimal Reproduction

```python
from chisel import capture_trace

@capture_trace(trace_name="debug_test")
def minimal_test():
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        x = torch.tensor([1.0, 2.0], device='cuda')
        print(f"GPU tensor: {x}")
        return x.cpu().numpy()
    else:
        return [1.0, 2.0]

if __name__ == "__main__":
    result = minimal_test()
    print(f"Result: {result}")
```

### System Information

```python
def print_system_info():
    import sys
    import os
    
    print("=== System Information ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch: Not installed")
    
    print("\n=== Chisel Environment ===")
    chisel_vars = {k: v for k, v in os.environ.items() if k.startswith('CHISEL')}
    for k, v in chisel_vars.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    print_system_info()
```

## Common Error Patterns

| Error                    | Cause                     | Solution                                    |
| ------------------------ | ------------------------- | ------------------------------------------- |
| `ModuleNotFoundError`    | Missing dependency        | Add to requirements.txt                     |
| `CUDA out of memory`     | Insufficient GPU memory   | Reduce batch size or use larger GPU         |
| `Script not found`       | Wrong upload directory    | Check `--upload-dir` parameter              |
| `Authentication failed`  | Invalid credentials       | Run `chisel --logout` and re-authenticate   |
| `Connection timeout`     | Network issues            | Check backend connectivity                  |
| `Argument parsing error` | Mixed chisel/script flags | Separate chisel flags from script arguments |

## Getting Help

### Collect Debug Information

```bash
# Collect system info
python -c "
import sys, os
print(f'Python: {sys.version}')
print(f'Platform: {sys.platform}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')
except: pass

chisel_env = {k:v for k,v in os.environ.items() if k.startswith('CHISEL')}
print(f'Chisel env: {chisel_env}')
"

# Test basic functionality
python -c "from chisel import capture_trace; print('✅ Import works')"
chisel --version
```

### Contact Support

When reporting issues, include:

1. **System information** (use script above)
2. **Complete error output**
3. **Minimal reproduction script**
4. **Command used**
5. **Job ID** (if applicable)

**Support channels:**
- **📧 Email**: [contact@herdora.com](mailto:contact@herdora.com)
- **🐛 GitHub Issues**: [Report bugs](https://github.com/Herdora/chisel/issues)
- **💬 GitHub Discussions**: [Ask questions](https://github.com/Herdora/chisel/discussions)

### Self-Help Checklist

Before contacting support:

- [ ] Tested with minimal reproduction script
- [ ] Tried locally with `python script.py`
- [ ] Checked requirements.txt includes all dependencies
- [ ] Verified script is in upload directory
- [ ] Cleared credentials with `chisel --logout`
- [ ] Checked system information
- [ ] Read relevant documentation sections

**Next:** [Development](development.md) | [API Reference](api-reference.md)