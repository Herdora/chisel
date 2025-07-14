# Chisel

Profile AMD HIP kernels and PyTorch code on RunPod instances with a local feel.

## What It Does

Chisel runs your PyTorch scripts and HIP kernels on remote AMD GPUs and automatically downloads comprehensive profiling traces to your local machine. No more SSH juggling or manual trace collection.

## Quick Start

```bash
# Install
pip install -e .

# Set RunPod API key
export RUNPOD_API_KEY="your_key_here"

# Profile a PyTorch script
chisel profile amd -f your_script.py

# Profile HIP kernels with basic tracing
chisel hip-trace your_kernel.hip

# Profile HIP kernels with performance counters
chisel hip-counters your_kernel.hip --counters "SQ_WAVES,GRBM_COUNT"
```

## What You Get

After profiling, traces are downloaded to `./chisel_out/`:

**PyTorch Profiling:**
- **torch_trace.json** - Chrome DevTools trace (view at `chrome://tracing`)
- **TensorBoard traces** - PyTorch profiler data

**HIP Kernel Profiling:**
- **rocprof_trace.csv** - HIP runtime + HSA kernel traces  
- **rocprof_kernel_trace.csv** - Kernel-focused analysis
- **rocprof_system_trace.csv** - Complete system traces
- **rocprof_memory_trace.csv** - Memory operation traces
- **rocprof_counters.csv** - Performance counter data
- **rocprof_trace.db** - ROCm profiler database
- **rocprof_trace.json** - JSON format traces
- **Stats and analysis files** - Detailed performance breakdowns

## Requirements

- RunPod account with API key
- SSH key added to your RunPod profile
- Python 3.8+

## How It Works

1. **Finds your running pod** - Uses RunPod API to locate available instances
2. **Uploads your code** - Transfers via SCP (Python scripts or HIP kernels)
3. **Compiles HIP kernels** - Uses `hipcc` with smart defaults and custom flags
4. **Runs with profiling** - Instruments with `torch.profiler` + `rocprof` for comprehensive analysis
5. **Downloads traces** - Automatically fetches all profiling data to timestamped directories

## Advanced Usage

```bash
# PyTorch profiling with connection persistence
chisel profile amd -f script.py --no-cleanup

# HIP profiling modes
chisel hip-trace kernel.hip                    # Basic HIP + HSA traces
chisel hip-kernel kernel.hip                   # Kernel-focused analysis
chisel hip-system kernel.hip                   # Complete system profiling
chisel hip-memory kernel.hip                   # Memory operations focus
chisel hip-counters kernel.hip --counters SQ_WAVES,GRBM_COUNT

# Custom compilation flags
chisel hip-trace kernel.hip --hipcc-flags "-O3 -DDEBUG"

# Check version and available commands
chisel --version
chisel --help
```

## Environment Setup

Chisel automatically installs profiling dependencies on first run:
- `rocprofiler-dev`, `roctracer-dev` for GPU profiling
- `hipcc` compiler for HIP kernel compilation
- PyTorch with ROCm support for ML workloads

No manual environment setup required.

## Supported Profiling

✅ **HIP Kernels** - Complete support with `hip-trace`, `hip-kernel`, `hip-system`, `hip-memory`, `hip-counters`  
✅ **PyTorch Scripts** - Full instrumentation with `torch.profiler` + `rocprof`  
✅ **Performance Counters** - Custom counter collection (SQ_WAVES, GRBM_COUNT, etc.)  
✅ **Multiple Output Formats** - CSV, JSON, database, and analysis files

## Future

**CUDA Support** - Planned support for NVIDIA GPUs with `nsight-compute` profiling. The CLI already accepts `cuda` as a target for future compatibility.

**Advanced Features** - ATT tracing, kernel filtering, and rocprof-compute-viewer integration.

---

**MIT License** • Built for performance engineers who want remote GPU profiling without the hassle. 