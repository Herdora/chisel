# Chisel CLI Examples

This directory contains examples showing how to use Chisel CLI in different scenarios.

## Examples

### Basic Usage (`basic_usage.py`)
Shows fundamental Chisel CLI usage:
- Using the `@capture_trace` decorator
- GPU matrix operations with profiling
- Local and cloud execution

```bash
# Local execution
python examples/basic_usage.py

# Cloud GPU execution
chisel python examples/basic_usage.py
```

### Model Profiling (`model_profiling_example.py`)
Demonstrates automatic profiling of PyTorch models:
- Using the `@capture_model` decorator
- Layer-level timing and shape analysis
- Automatic profiling of every forward pass
- Real-time layer summaries during execution

```bash
# Local execution (profiling simulated)
python examples/model_profiling_example.py

# Cloud GPU execution (full profiling)
chisel python examples/model_profiling_example.py
```

**Features demonstrated:**
- Automatic profiling of every model forward pass
- Layer-level timing analysis with detailed breakdowns
- Shape tracking for each layer and operation
- Support for complex models (CNNs, Transformers)
- Real-time layer summaries during execution
- Detailed trace files saved to backend

### Command Line Arguments (`args_example.py`)
Demonstrates passing command line arguments to your script:

```bash
chisel python examples/args_example.py --iterations 5
```

### Requirements (`requirements_example.py`)
Shows how to use custom requirements files:

```bash
chisel python examples/requirements_example.py
```

### Inline Tracing (`specific_call.py`)
Demonstrates inline function wrapping for external functions:

```bash
chisel python examples/specific_call.py
```

### Simple Example (`simple_example.py`)
Minimal example showing the new simplified usage pattern:

```bash
chisel python examples/simple_example.py
```

### Large File Caching Examples (`large_file_test/`)
Comprehensive examples demonstrating automatic caching of large files (>1GB):

```bash
# Quick test - create and process a 1.1GB file
chisel python examples/large_file_test/quick_test.py

# Full featured example with multiple large files
chisel python examples/large_file_caching_example.py

# Test deduplication with identical files
chisel python examples/large_file_test/dedup_test.py
```

**Features demonstrated:**
- Automatic detection and caching of files >1GB
- SHA256-based deduplication 
- Transparent placeholder/restoration system
- Significant speedup on repeat runs

## Installation

Make sure you have Chisel CLI installed:

```bash
# From GitHub (recommended for development)
pip install git+https://github.com/Herdora/chisel.git@dev

# Or if published to PyPI
pip install chisel-cli
```

## Prerequisites

- Python 3.8+
- PyTorch (for GPU examples)

## Running Examples

### Local Execution
Run directly with Python for local testing:
```bash
python examples/basic_usage.py
```

### Cloud GPU Execution
Use the `chisel` command for cloud GPU execution:
```bash
chisel python examples/basic_usage.py
```

The CLI will prompt you for:
- App name (for job tracking)
- Upload directory (default: current directory)
- Requirements file (default: requirements.txt)
- GPU configuration (1, 2, 4, or 8x A100-80GB)

## Authentication

On first run, Chisel CLI will automatically open your browser for authentication. Your credentials are securely stored in `~/.chisel/credentials.json` for future use.

## Notes

- GPU examples will automatically detect and use CUDA if available
- CPU fallback is provided for all GPU examples
- Trace files are saved to the backend and can be downloaded for analysis
- All examples include comprehensive error handling and user feedback
- The same code works both locally and on cloud GPUs

### Model Profiling

- Use `@capture_model` decorator to automatically profile PyTorch models
- Every forward pass is automatically profiled and traced
- Layer-level timing shows which layers are bottlenecks
- Shape tracking helps identify memory usage patterns
- Real-time summaries show performance during execution
- Detailed traces can be analyzed with `parse_model_trace()`

### Upload Limits & Large File Handling

**🔄 CACHING DISABLED** - New streamlined upload system:

- ✅ **No caching threshold** - Upload folders of any total size
- ✅ **5GB per-file limit** - Individual files cannot exceed 5GB
- ✅ **Direct uploads** - All files upload directly (faster, simpler)
- ✅ **Clear error messages** - Helpful guidance for oversized files

**For files over 5GB:**
- Download them within your script instead of uploading
- Use `requests.get()`, `boto3`, or similar libraries
- Generate large data programmatically in your script
- More efficient than uploading large files

**Test the upload limits:**
```bash
# Create test files with various sizes
python test_upload_limits.py

# Test upload behavior (will show 5GB limit error)
chisel run --upload-dir upload_limit_test simple_upload_test.py

# Clean up test files
python test_upload_limits.py --cleanup
```

See `UPLOAD_LIMITS_TEST.md` for detailed examples and migration guide.