# Chisel

Profile AMD HIP kernels and PyTorch code on RunPod instances with a local feel.

## What It Does

Chisel runs your PyTorch scripts on remote AMD GPUs and automatically downloads comprehensive profiling traces to your local machine. No more SSH juggling or manual trace collection.

## Quick Start

```bash
# Install
pip install -e .

# Set RunPod API key
export RUNPOD_API_KEY="your_key_here"

# Profile a script on AMD GPU
python -m chisel profile amd -f your_script.py
```

## What You Get

After profiling, traces are downloaded to `./chisel_out/`:

- **torch_trace.json** - Chrome DevTools trace (view at `chrome://tracing`)
- **rocprof_trace.csv** - AMD GPU kernel traces
- **rocprof_trace.db** - ROCm profiler database
- **TensorBoard traces** - PyTorch profiler data

## Requirements

- RunPod account with API key
- SSH key added to your RunPod profile
- Python 3.8+

## How It Works

1. **Finds your running pod** - Uses RunPod API to locate available instances
2. **Uploads your script** - Transfers via SCP  
3. **Runs with profiling** - Instruments with `torch.profiler` + `rocprof`
4. **Downloads traces** - Automatically fetches all profiling data

## Advanced Usage

```bash
# Keep connection alive after profiling
python -m chisel profile amd -f script.py --no-cleanup

# Check version
python -m chisel --version
```

## Environment Setup

Chisel automatically installs profiling dependencies on first run:
- `rocprofiler-dev`, `roctracer-dev`, PyTorch with ROCm support

No manual environment setup required.

## Future

**CUDA Support** - Planned support for NVIDIA GPUs with `nsight-compute` profiling. The CLI already accepts `cuda` as a target for future compatibility.

---

**MIT License** • Built for performance engineers who want remote GPU profiling without the hassle. 