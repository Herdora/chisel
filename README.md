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
Profile GPU kernels on NVIDIA H100 ($4.89/hour).

```bash
# Compile and profile CUDA source files
chisel profile nvidia matrix.cu
chisel profile nvidia kernel.cu

# Profile existing binaries or commands
chisel profile nvidia "./my-cuda-app --size=1024"
chisel profile nvidia "nvidia-smi"
```

**What it does:**
- Creates H100 droplet if needed (reuses existing)
- Auto-syncs source files to droplet
- Compiles with `nvcc` for `.cu` files
- Runs your code (profiling tools coming soon)
- Downloads results locally
- Shows cost estimate

### `chisel profile amd <file_or_command>`
Profile GPU kernels on AMD MI300X ($1.99/hour).

```bash
# Compile and profile HIP source files
chisel profile amd matrix.cpp
chisel profile amd kernel.hip

# Profile existing binaries or commands  
chisel profile amd "./my-hip-app --iterations=100"
chisel profile amd "rocm-smi"
```

**What it does:**
- Creates MI300X droplet if needed (reuses existing)
- Auto-syncs source files to droplet
- Compiles with `hipcc` for `.cpp/.hip` files
- Runs `rocprof` profiler with hip,hsa traces
- Downloads profiling results locally
- Shows summary of top GPU kernels

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
```

## How It Works

1. **Automatic Infrastructure**: Creates GPU droplets on DigitalOcean automatically
2. **Smart Detection**: Detects source files vs commands and handles appropriately
3. **Seamless Sync**: Your code is automatically transferred to the GPU droplet
4. **Auto Compilation**: Compiles `.cu` files with `nvcc`, `.cpp/.hip` files with `hipcc`
5. **Profiling**: Runs appropriate profilers (rocprof for AMD, nsight planned for NVIDIA)
6. **Results Download**: Profiling data automatically downloaded to `./chisel-results/`
7. **Cost Management**: Droplets reused between commands, auto-cleanup after 15min inactivity

## GPU Support

| GPU | Size | Region | Cost/Hour | Profiling |
|-----|------|---------|-----------|----------|
| NVIDIA H100 | `gpu-h100x1-80gb` | NYC2 | $4.89 | Coming soon |
| AMD MI300X | `gpu-mi300x1-192gb` | ATL1 | $1.99 | ✅ rocprof |

## Development Setup

```bash
# With uv (recommended)
uv sync
uv run chisel <command>

# With pip
pip install -e .
```

## Cost Management

- **Droplet reuse**: Existing droplets are reused between profiling runs
- **Auto-cleanup**: Droplets automatically destroyed after 15 minutes of inactivity
- **Cost tracking**: Each command shows estimated cost
- **No surprises**: Always shows what you'll be charged for

## Architecture

Chisel's simplified architecture focuses on the core use case: profiling GPU kernels.

- **ProfileManager**: Orchestrates the entire workflow (droplet, sync, compile, profile, download)
- **Automatic droplet management**: Creates, reuses, and cleans up GPU droplets transparently  
- **Smart target detection**: Automatically detects source files vs commands
- **Unified results**: Consistent output format for both AMD and NVIDIA

## Future

- [ ] NVIDIA profiling (nsight-compute, nsight-systems)
- [ ] Support other cloud backends
- [ ] Advanced profiling options