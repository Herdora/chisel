<div align="center">
  <img width="450" height="300" src="https://i.imgur.com/H32IKRZ.jpeg" alt="Chisel CLI logo" /> 
	<h1>chisel</h1>
</div>

**TL;DR:** Seamless GPU kernel profiling on cloud infrastructure. Write GPU code, run one command, get profiling results. Zero GPU hardware required.

> 🚀 **Recent Releases**
>
> - **Python Support**: Direct profiling of Python GPU applications (PyTorch, TensorFlow, etc.)
> - **AMD rocprofv3 Support**: Full integration with AMD's latest profiling tool

> 🔮 **Upcoming Features**
>
> - **Web Dashboard**: Browser-based visualization of profiling results.
> - **Multi-GPU Support**: Profile the same kernel across multiple GPU types simultaneously.
> - **Profiling Backend**: Use Chisel’s built-in backend to run profiling workloads—no cloud account or API token required.
> - **More GPU Architectures**: Support for most requested GPU type.
> - **Auto cleaning**: Smart cleaning and release of your DigitalOcean resources (destroying the droplet after you don't use it).

> 💡 **Feature Requests**
>
> Got a feature idea you'd love to see in Chisel? We'd love to hear from you!
>
> - Open a feature request on [GitHub Issues](https://github.com/Herdora/chisel/issues)
> - Or email us directly at **contact [at] herdora [dot] com**
>
> Please include a short description of the feature, how you'd use it, and any context that might help us prioritize.

## About

Testing and profiling GPU kernels across different hardware is time-consuming and hardware-dependent, as it often involves complex setup, driver compatibility issues, dependency mismatches, and access to specialized GPUs. Chisel removes that friction by letting you run and profile GPU code in the cloud with a single command - no GPU required. It’s the fastest way to validate kernel performance on real NVIDIA and AMD GPUs.

## Quick Start

Get up and running in 30 seconds:

```bash
# 1. Install chisel
pip install chisel-cli

# 2. Choose your authentication method:
# Option A: Use Chisel's managed authentication (includes $10 free credits)
chisel login

# Option B: Use your own DigitalOcean API token
chisel configure

# 3. Compile your code into an executable
hipcc --amdgpu-target=gfx940 -o examples/simple-mm examples/simple-mm.hip
 # for inlined kernels to python on amd.
nvcc -arch=sm_90 -o examples/my_kernel examples/my_kernel.cu
nvcc -arch=sm_90 -Xcompiler -fPIC -shared -o examples/libvector_add.so examples/vector_add.cu # for inlined kernels to python on nvidia.


# 4. Profile your GPU kernels and applications
chisel profile --rocprofv3="--sys-trace" -f examples/simple-mm "./simple-mm" # since this just copies the file, it isn't placed in a dir on the server.
chisel profile --rocprofv3="--sys-trace" -f examples/hip_vector_add_test.py -f examples/libvector_add_hip.so "python hip_vector_add_test.py"
chisel profile --nsys="--trace=cuda --cuda-memory-usage=true" -f examples/kernel.out "./kernel.out"
chisel profile --nsys="--trace=cuda --cuda-memory-usage=true" -f examples "python examples/simple_gpu_test.py" # syncs the entire examples directory and runs simple_gpu_test.py
# TODO: add the case for when user has "-f examples/"
# TODO: make names in examples/ directory more descriptive.
```

**That's it!** 🚀 No GPU hardware needed—develop and profile GPU kernels from any machine.

> **Getting Started:**
>
> - **Free Credits**: Email **contact@herdora.com** to get your account with $10 in free credits (no DigitalOcean account needed)
> - **Bring Your Own Account**: Get a DigitalOcean API token [here](https://amd.digitalocean.com/account/api/tokens) (requires read/write access)
>
> If using your own DigitalOcean account, ensure to destroy droplets via their dashboard when done.

## Commands

Chisel has **4 commands**:

### `chisel login`

Authenticate with Chisel's managed backend (includes $10 free credits).

- Email contact@herdora.com to activate an account.

```bash
# Login with your Chisel token
chisel login
```

### `chisel configure`

One-time setup of your DigitalOcean API credentials (for users with their own accounts).

```bash
# Interactive configuration
chisel configure

# Non-interactive with token
chisel configure --token YOUR_TOKEN
```

## GPU Support

| GPU         | Size                | Region | Profiling                       |
| ----------- | ------------------- | ------ | ------------------------------- |
| NVIDIA H100 | `gpu-h100x1-80gb`   | NYC2   | nsight-compute + nsight-systems |
| NVIDIA L40S | `gpu-l40sx1-48gb`   | TOR1   | nsight-compute + nsight-systems |
| AMD MI300X  | `gpu-mi300x1-192gb` | ATL1   | rocprofv3                       |

## Development Setup

```bash
# With uv (recommended)
uv sync
uv run chisel <command>

# With pip
pip install -e .
```

## Making updates to PyPI

```bash
rm -rf dist/ build/ *.egg-info && python -m build && twine upload dist/*
```
