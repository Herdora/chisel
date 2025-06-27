<div align="center">
  <img width="300" height="300" src="https://i.imgur.com/KISXGnH.png" alt="Chisel CLI logo" /> 
	<h1>chisel</h1>
</div>

**TL;DR:** Seamless GPU kernel profiling on cloud infrastructure. Write GPU code, run one command, get profiling results. Zero GPU hardware required.

> 🚀 **Recent Releases**
> 
> ### Latest
> - **Python Support**: Direct profiling of Python GPU applications (PyTorch, TensorFlow, etc.)
> - **AMD rocprofv3 Support**: Full integration with AMD's latest profiling tool
> - **Automatic Cleanup**: Remote files are automatically cleaned up after profiling

> 🔮 **Upcoming Features**
> 
> ### In Development
> - **Web Dashboard**: Browser-based visualization of profiling results.
> - **Multi-GPU Support**: Profile the same kernel across multiple GPU types simultaneously.
> - **Profiling Backend**: Bypass the need for a DigitalOcean account by using a public backend.

## Quick Start

Get up and running in 30 seconds:

```bash
# 1. Install chisel
pip install chisel-cli

# 2. Configure with your DigitalOcean API token
chisel configure

# 3. Compile your code into an executable
hipcc --amdgpu-target=gfx940 -o examples/simple-mm examples/simple-mm.hip
hipcc --amdgpu-target=gfx940 -fPIC -shared -o examples/libvector_add_hip.so examples/vector_add_hip.hip # for inlined kernels to python on amd.
nvcc -arch=sm_90 -o examples/my_kernel examples/my_kernel.cu
nvcc -arch=sm_90 -Xcompiler -fPIC -shared -o examples/libvector_add.so examples/vector_add.cu # for inlined kernels to python on nvidia.


# 4. Profile your GPU kernels and applications  
chisel profile --rocprofv3="--sys-trace" -f examples/simple-mm.hip "./simple-mm" # since this just copies the file, it isn't placed in a dir on the server.
chisel profile --rocprofv3="--sys-trace" -f examples/hip_vector_add_test.py -f examples/libvector_add_hip.so "python hip_vector_add_test.py"
chisel profile --rocprofv3="--sys-trace" -f examples/attention_block.py "python attention_block.py"
chisel profile --nsys="--trace=cuda --cuda-memory-usage=true" -f examples "python examples/main.py" # syncs the entire examples directory and runs main.py
# TODO: add the case for when user has "-f examples/"
# TODO: make names in examples/ directory more descriptive.
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

## Making updates to PyPI

```bash
rm -rf dist/ build/ *.egg-info && python -m build && twine upload dist/*
```

