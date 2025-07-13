"""Command-line interface for Chisel - pure argument parser."""

from typing import Optional, Callable, List

import typer

from chisel.cli.commands import (
    handle_configure,
    handle_profile,
    handle_version,
    handle_install_completion,
    handle_login,
)

# Sentinel value to distinguish between "not provided" and "provided as empty"
NOT_PROVIDED = object()


def vendor_completer(incomplete: str):
    """Custom completer for vendor argument."""
    vendors = ["nvidia", "amd"]
    return [vendor for vendor in vendors if vendor.startswith(incomplete)]


def gpu_type_completer(incomplete: str):
    """Custom completer for gpu-type option."""
    gpu_types = ["h100", "l40s"]
    return [
        gpu_type for gpu_type in gpu_types if gpu_type.startswith(incomplete)
    ]


def create_profiler_callback(
    profiler_name: str,
) -> Callable[[Optional[str]], str]:
    """Create a callback that detects when a profiler flag is used."""

    def callback(value: Optional[str]) -> str:
        # If value is None, the flag wasn't used
        # If value is a string (even empty), the flag was used
        if value is None:
            return ""  # Flag not used
        else:
            return (
                value  # Flag was used, return the value (could be empty string)
            )

    return callback


def create_app() -> typer.Typer:
    """Create and configure the Typer app with all commands."""
    app = typer.Typer(
        name="chisel",
        help="Seamless GPU kernel profiling on cloud infrastructure",
        add_completion=True,
    )

    @app.command()
    def configure(
        token: Optional[str] = typer.Option(
            None, "--token", "-t", help="DigitalOcean API token"
        ),
    ):
        """Configure Chisel with your DigitalOcean API token."""
        exit_code = handle_configure(token=token)
        raise typer.Exit(exit_code)

    @app.command()
    def login(
        token: str = typer.Argument(..., help="Chisel token for managed access"),
    ):
        """Login with a Chisel token for managed access."""
        exit_code = handle_login(token=token)
        raise typer.Exit(exit_code)

    @app.command()
    def profile(
        rocprofv3: Optional[str] = typer.Option(
            None,
            "--rocprofv3",
            help="Run rocprofv3 profiler (AMD). Use --rocprofv3 for default, or --rocprofv3='extra flags' for custom options",
        ),
        rocprof_compute: Optional[str] = typer.Option(
            None,
            "--rocprof-compute",
            help="Run rocprof-compute profiler (AMD). Use --rocprof-compute for default, or --rocprof-compute='extra flags' for custom options",
        ),
        nsys: Optional[str] = typer.Option(
            None,
            "--nsys",
            help="Run nsys profiler (NVIDIA). Use --nsys for default, or --nsys='extra flags' for custom options",
        ),
        ncompute: Optional[str] = typer.Option(
            None,
            "--ncompute",
            help="Run ncu (nsight-compute) profiler (NVIDIA). Use --ncompute for default, or --ncompute='extra flags' for custom options",
        ),
        output_dir: Optional[str] = typer.Option(
            None,
            "--output-dir",
            "-o",
            help="Output directory for profiling results. If not specified, uses default timestamped directory.",
        ),
        gpu_type: Optional[str] = typer.Option(
            None,
            "--gpu-type",
            help="GPU type: 'h100' (default) or 'l40s' (NVIDIA only)",
            autocompletion=gpu_type_completer,
        ),
        files_to_sync: Optional[List[str]] = typer.Option(
            None,
            "--files-to-sync",
            "-f",
            help="Files to sync to the remote machine. If not specified, no files will be synced.",
        ),
        command_to_profile: Optional[str] = typer.Argument(
            None,
            help="Target to profile: source file (e.g., kernel.cu, train.py), executable, or command",
        ),
    ):
        """Profile GPU kernels and applications on cloud infrastructure.

        The target can be:
        • Source files (.cu, .cpp, .hip, .py) - automatically synced and executed
        • Executables or commands - run directly on remote GPU
        • Python applications - automatically detects PyTorch/TensorFlow GPU usage

        Examples:
            chisel profile --rocprofv3 ./matrix_multiply           # AMD: executable/command
            chisel profile --nsys ./kernel.cu                     # NVIDIA: CUDA source file
            chisel profile --rocprofv3 ./train.py                 # AMD: Python GPU application
            chisel profile --nsys="--trace=cuda,nvtx" ./train.py  # NVIDIA: Python with tracing
            chisel profile --rocprofv3="--sys-trace" ./saxpy.cpp  # AMD: HIP source with flags
            chisel profile --nsys --gpu-type l40s ./matmul.cu     # NVIDIA: specify GPU type
            chisel profile --nsys --ncompute --output-dir ./results ./cuda_kernel  # Multiple profilers
        """
        exit_code = handle_profile(
            command_to_profile=command_to_profile,
            files_to_sync=files_to_sync,
            rocprofv3=rocprofv3,
            rocprof_compute=rocprof_compute,
            nsys=nsys,
            ncompute=ncompute,
            output_dir=output_dir,
            gpu_type=gpu_type,
        )
        raise typer.Exit(exit_code)

    @app.command("install-completion")
    def install_completion(
        shell: Optional[str] = typer.Option(
            None,
            "--shell",
            help="Shell to install completion for: bash, zsh, fish, powershell. Auto-detects if not specified.",
        ),
    ):
        """Install shell completion for the chisel command."""
        exit_code = handle_install_completion(shell=shell)
        raise typer.Exit(exit_code)

    @app.command()
    def version():
        """Show Chisel version."""
        exit_code = handle_version()
        raise typer.Exit(exit_code)

    return app


def run_cli():
    """Main CLI entry point."""
    app = create_app()
    app()
