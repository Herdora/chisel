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

### Large File Caching

- Files >1GB are automatically cached by SHA256 hash
- First run uploads and caches large files
- Subsequent runs with identical files use cache (much faster!)
- Deduplication saves storage - identical files cached only once
- Cache management available in frontend at `/cached-files`
- Files restored automatically during job execution

**⚠️ Important:** When uploading jobs with large cached files, **do not refresh the browser page** during the upload process. Refreshing can interrupt the caching mechanism and cause the upload to fail or restart.

**Upload time estimates:**
- **1GB file**: ~2-5 minutes first upload → ~10-30 seconds subsequent uploads
- **5GB file**: ~10-25 minutes first upload → ~30-60 seconds subsequent uploads  
- **10GB+ files**: ~20-50+ minutes first upload → ~1-2 minutes subsequent uploads

*Actual times depend on your internet connection speed.*