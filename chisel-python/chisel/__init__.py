"""Chisel - Seamless GPU kernel profiling on cloud infrastructure.

This package provides both a CLI interface and a programmatic API for
profiling GPU kernels on cloud infrastructure.

CLI usage:
    chisel profile nvidia kernel.cu
    chisel profile amd kernel.hip

Programmatic usage:
    from chisel.core import ProfilingManager

    manager = ProfilingManager()
    result = manager.profile("nvidia", "kernel.cu")
"""

__version__ = "0.1.5"

# Expose core API for programmatic use
from chisel.core import (
    DropletService,
    Droplet,
    GPU_PROFILES,
    GPUType,
    GPUProfile,
    GPURegion,
    ProfilingManager,
)

# Expose CLI functionality
from chisel.cli import main, run_cli, create_app

__all__ = [
    # Version
    "__version__",
    # Core API
    "DropletService",
    "Droplet",
    "GPU_PROFILES",
    "GPUType",
    "GPUProfile",
    "GPURegion",
    "ProfilingManager",
    # CLI
    "main",
    "run_cli",
    "create_app",
]
