# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chisel is a Python tool for profiling AMD HIP kernels and PyTorch code on RunPod instances with a local development experience. It automates the process of running profiling scripts on remote GPUs and collecting traces.

## Key Commands

### Development Setup
```bash
# Install in development mode with dependencies
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install using uv package manager (alternative)
uv pip install -e .
```

### Running Chisel
```bash
# Profile a script on AMD GPU
python -m chisel profile amd -f script.py

# Profile with no cleanup (keeps pod connection alive)
python -m chisel profile amd -f script.py --no-cleanup

# Check version
python -m chisel --version

# Or use the installed CLI command
chisel profile amd -f script.py
```

## Project Architecture

### Core Components

1. **cli.py** - Command-line interface using Click framework
   - Entry point for all user commands
   - Handles argument parsing and validation
   - Provides user feedback with emoji indicators

2. **pod_manager.py** - High-level pod orchestration
   - Manages RunPod instance lifecycle
   - Handles SSH connections and file transfers
   - Coordinates profiling execution and trace collection
   - Implements automatic cleanup with context management

3. **pod.py** - Low-level pod abstraction
   - Direct interface to RunPod API
   - SSH/SCP operations wrapper
   - Error handling for network operations

### Directory Structure
- `chisel/` - Main package source code
- `examples/` - Example scripts for testing profiling
- `chisel_out/` - Default output directory for downloaded traces
- `chisel_results/` - Timestamped subdirectories for organized results

### Key Design Patterns

1. **Context Management**: PodManager uses context managers for automatic cleanup
2. **Error Handling**: Custom PodManagerError exception for pod-related failures
3. **Progressive Feedback**: CLI provides real-time status updates with emojis
4. **Modular Architecture**: Clear separation between CLI, management, and pod layers

## Environment Variables

- `RUNPOD_API_KEY` - Required for RunPod API authentication
- SSH keys must be configured in RunPod profile for pod access

## Output Files

Profiling generates these trace files in `chisel_out/`:
- `torch_trace.json` - Chrome DevTools format trace
- `rocprof_trace.csv` - AMD GPU kernel timing data
- `rocprof_trace.db` - ROCm profiler database
- TensorBoard event files for PyTorch profiler

## Development Notes

- Python 3.12+ required (note: pyproject.toml shows 3.12+ but classifiers list 3.8-3.11)
- Uses modern Python packaging with pyproject.toml
- Dependencies managed with uv package manager (uv.lock file)
- Currently AMD-only; CUDA support is planned but not implemented
- No test suite exists yet - consider adding pytest tests when implementing new features

## Common Development Tasks

When modifying the profiling workflow:
1. Check `pod_manager.py` for the main orchestration logic
2. Remote execution happens via SSH commands in the pod abstraction
3. Profiling setup includes installing ROCm tools on first run
4. Trace collection uses SCP to download results

When adding new CLI commands:
1. Add new command to `cli.py` using Click decorators
2. Follow existing patterns for error handling and user feedback
3. Use PodManager for any pod-related operations