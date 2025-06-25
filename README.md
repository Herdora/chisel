<div align="center">
  <img width="300" height="300" src="https://i.imgur.com/KISXGnH.png" alt="Chisel CLI logo" /> 
	<h1>chisel</h1>
</div>

**TL;DR:** Seamless GPU kernel profiling on cloud infrastructure. Write GPU code, run one command, get profiling results. Zero GPU hardware required.

## Quick Start

Get up and running in 30 seconds:

```bash
# 1. Install chisel
pip install chisel-cli

# 2. Configure with your DigitalOcean API token
chisel configure

# 3. Profile your GPU kernels
chisel profile nvidia kernel.cu    # NVIDIA H100
chisel profile amd kernel.cpp      # AMD MI300X
```

**That's it!** 🚀 No GPU hardware needed—develop and profile GPU kernels from any machine.

> **Need a DigitalOcean API token?** Get one [here](https://amd.digitalocean.com/account/api/tokens) (requires read/write access).

## Commands

Chisel has just **3 commands**:

### `chisel configure`

One-time setup of your DigitalOcean API credentials.

```bash
# Interactive configuration
chisel configure

# Non-interactive with token
chisel configure --token YOUR_TOKEN
```

### `chisel profile nvidia <file_or_command>`

Profile GPU kernels on NVIDIA H100 ($4.89/hour) or L40S ($2.21/hour).

```bash
# Compile and profile CUDA source files
chisel profile nvidia matrix.cu              # Default: H100
chisel profile nvidia kernel.cu --gpu-type l40s  # L40S GPU

# Profile existing binaries or commands
chisel profile nvidia "./my-cuda-app --size=1024"
chisel profile nvidia "nvidia-smi"
```

### `chisel profile amd <file_or_command>`

Profile GPU kernels on AMD MI300X ($1.99/hour).

```bash
# Compile and profile HIP source files
chisel profile amd matrix.cpp
chisel profile amd kernel.hip

# Profile with performance counters
chisel profile amd matrix.cpp --pmc "GRBM_GUI_ACTIVE,SQ_WAVES,SQ_BUSY_CYCLES"

# Profile existing binaries or commands
chisel profile amd "./my-hip-app --iterations=100"
chisel profile amd "rocm-smi"
```

## Examples

### AMD Profiling

```bash
# Create a simple HIP kernel
cat > simple.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void add_kernel(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Your HIP code here
    std::cout << "HIP kernel executed!" << std::endl;
    return 0;
}
EOF

# Profile it
chisel profile amd simple.cpp

# Profile with performance counters
chisel profile amd simple.cpp --pmc "GRBM_GUI_ACTIVE,SQ_WAVES"

# Get a human-readable summary table showing:
# - HIP/HSA API calls with timing breakdown
# - Kernel execution statistics
# - Memory operations and bandwidth analysis
# - Performance counter data (when --pmc used)
```

### NVIDIA Profiling

```bash
# Create a simple CUDA kernel
cat > simple.cu << 'EOF'
#include <cuda_runtime.h>
#include <iostream>

__global__ void multiply_kernel(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    // Your CUDA code here
    std::cout << "CUDA kernel executed!" << std::endl;
    return 0;
}
EOF

# Profile it
chisel profile nvidia simple.cu

# Get a human-readable summary table showing:
# - CUDA kernels with execution time and call counts
# - Memory operations and bandwidth analysis
# - API call timing breakdown
```

## Profiling Output

Chisel generates **human-readable summary tables** that are easy to analyze:

### Sample NVIDIA Output

```
 ** CUDA Summary (API/Kernels/MemOps) (cuda_api_gpu_sum):

 Time (%)  Total Time (ns)  Instances   Avg (ns)   Med (ns)  Min (ns)  Max (ns)   StdDev (ns)   Category      Operation
 --------  ---------------  ---------  ----------  --------  --------  ---------  -----------  -----------  -------------
     99.0        181321127          3  60440375.7  213009.0    194613  180913505  104332790.9  CUDA_API     cudaMalloc
      0.4           660237          2    330118.5  330118.5    318682     341555      16173.7  CUDA_API     cudaMemcpy
      0.1           198058         10     19805.8    3733.5      3006     163066      50342.9  CUDA_API     cudaLaunchKernel
      0.0            45472         10      4547.2    4512.0      4352       4992        183.8  CUDA_KERNEL  simple_kernel(float *, float *, float *, int)
```

### Sample AMD Output

```
|                    NAME                    |      DOMAIN       |      CALLS      | DURATION (nsec) | AVERAGE (nsec)  | PERCENT (INC) |
|--------------------------------------------|-------------------|-----------------|-----------------|-----------------|---------------|
| hipMemcpy                                  | HIP_API           |               2 |       200447591 |       1.002e+08 |     75.334795 |
| hipMalloc                                  | HIP_API           |               3 |        23377126 |       7.792e+06 |      8.785892 |
| hipLaunchKernel                            | HIP_API           |              10 |          429928 |       4.299e+04 |      0.161581 |
| simple_kernel(float*, float*, float*, int) | KERNEL_DISPATCH   |              10 |           41615 |       4.162e+03 |      0.015640 |
```

**Results are saved to:** `chisel-results/TIMESTAMP/profile_summary.txt`

## GPU Support

| GPU         | Size                | Region | Cost/Hour | Profiling                       |
| ----------- | ------------------- | ------ | --------- | ------------------------------- |
| NVIDIA H100 | `gpu-h100x1-80gb`   | NYC2   | $4.89     | nsight-compute + nsight-systems |
| NVIDIA L40S | `gpu-l40sx1-48gb`   | TOR1   | $2.21     | nsight-compute + nsight-systems |
| AMD MI300X  | `gpu-mi300x1-192gb` | ATL1   | $1.99     | rocprofv3                       |

## Development Setup

```bash
# With uv (recommended)
uv sync
uv run chisel <command>

# With pip
pip install -e .
```
