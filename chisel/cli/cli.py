"""Command-line interface for Chisel - pure argument parser."""

from typing import Optional, Callable

import typer

from chisel.cli.commands import (
    handle_configure,
    handle_profile,
    handle_version,
    handle_install_completion,
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
    return [gpu_type for gpu_type in gpu_types if gpu_type.startswith(incomplete)]


def create_profiler_callback(profiler_name: str) -> Callable[[Optional[str]], str]:
    """Create a callback that detects when a profiler flag is used."""

    def callback(value: Optional[str]) -> str:
        # If value is None, the flag wasn't used
        # If value is a string (even empty), the flag was used
        if value is None:
            return ""  # Flag not used
        else:
            return value  # Flag was used, return the value (could be empty string)

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
        token: Optional[str] = typer.Option(None, "--token", "-t", help="DigitalOcean API token"),
    ):
        """Configure Chisel with your DigitalOcean API token."""
        exit_code = handle_configure(token=token)
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
        target: Optional[str] = typer.Argument(
            None, help="File to compile and profile (e.g., kernel.cu) or command to run"
        ),
    ):
        """Profile a GPU kernel or command on cloud infrastructure.

        Examples:
            chisel profile --rocprofv3 ./matrix_multiply
            chisel profile --nsys ./kernel.cu
            chisel profile --rocprofv3="--sys-trace --pmc SQ_BUSY_CYCLES,SQ_WAVES" ./saxpy.cpp
            chisel profile --nsys --gpu-type l40s ./matmul.cu
            chisel profile --rocprofv3 --rocprof-compute --output-dir ./results ./gemm
            chisel profile --nsys --ncompute --output-dir ./my_results ./cuda_kernel
        """
        exit_code = handle_profile(
            target=target,
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
