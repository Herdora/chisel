# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

### Installation
```bash
# Development with uv (recommended):
uv sync
uv run chisel <command>

# Development with pip:
pip install -e .

# When making changes, prefix all chisel commands with 'uv run' if using uv
```

### Running Commands
Since this is a CLI tool, all functionality is accessed through the `chisel` command. During development, always use `uv run chisel` to ensure you're using the development version.

## Architecture Overview

Chisel is a CLI tool for GPU kernel development on DigitalOcean droplets. The architecture follows a modular design:

### Core Components

1. **CLI Interface** (`main.py`): 
   - Entry point using Typer framework
   - Contains all command definitions
   - Handles GPU context resolution (explicit `--gpu-type` flag or active context)

2. **State Management** (`state.py`):
   - Tracks active GPU context in `~/.cache/chisel/context.txt`
   - Stores droplet information per GPU type in `~/.cache/chisel/state.json`
   - Supports concurrent multi-droplet workflows (AMD and NVIDIA simultaneously)

3. **DigitalOcean Integration** (`do_client.py`, `droplet.py`):
   - Manages droplet lifecycle (create, destroy, list)
   - GPU profiles defined in `gpu_profiles.py`:
     - AMD MI300X: `gpu-mi300x1-192gb` size, ATL1 region, $1.99/hour
     - NVIDIA H100: `gpu-h100x1-80gb` size, NYC2 region, $4.89/hour
   - Auto-configures environment (ROCm paths for AMD)

4. **SSH Operations** (`ssh_manager.py`):
   - File syncing via rsync
   - Remote command execution with live output streaming
   - Profiling workflow (compile, profile, download results)
   - Cost warnings for droplets running >12 hours

### Key Workflows

1. **Context-Based Operations**: Most commands support both explicit `--gpu-type` flag and context-based operation:
   ```python
   # Pattern used throughout:
   resolved_gpu_type = resolve_gpu_type(gpu_type)  # Falls back to active context
   ```

2. **Profiling Pipeline**:
   - Auto-detects source files vs commands
   - Syncs source files if needed
   - Compiles with appropriate compiler (hipcc for AMD, nvcc for NVIDIA)
   - Runs profiler (currently only rocprof for AMD implemented)
   - Downloads and extracts results
   - Shows summary of top kernels

3. **State Persistence**: 
   - Droplets are tracked per GPU type
   - Allows reusing existing droplets across commands
   - Cleans up state when droplets are destroyed

## Important Implementation Notes

- **No NVIDIA Profiling Yet**: While NVIDIA H100 is supported for compute, Nsight profiling tools are not yet implemented
- **AMD-Focused Profiling**: Current `profile` command only supports `rocprof` for AMD GPUs
- **SSH Key Management**: Automatically uses all SSH keys from DigitalOcean account
- **Cost Management**: Droplets self-destruct after 15 minutes of inactivity; warnings shown after 12 hours

## Testing

Currently, there is no testing infrastructure in place. When implementing tests in the future:
- Consider mocking DigitalOcean API calls
- Test SSH operations with paramiko mocks
- Verify state management persistence and recovery

## Future Simplified Interface

The codebase is being restructured to provide just three commands:
- `chisel configure` - One-time setup
- `chisel profile nvidia <file_or_command>` - Profile on NVIDIA H100
- `chisel profile amd <file_or_command>` - Profile on AMD MI300X

This will internally orchestrate all current commands (up, sync, profile, pull) automatically.