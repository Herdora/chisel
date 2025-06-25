"""Command-line interface for Chisel - pure argument parser."""

from typing import Optional

import typer

from chisel.cli.commands import (
    handle_configure,
    handle_profile,
    handle_version,
    handle_install_completion,
)


def vendor_completer(incomplete: str):
    """Custom completer for vendor argument."""
    vendors = ["nvidia", "amd"]
    return [vendor for vendor in vendors if vendor.startswith(incomplete)]


def gpu_type_completer(incomplete: str):
    """Custom completer for gpu-type option."""
    gpu_types = ["h100", "l40s"]
    return [gpu_type for gpu_type in gpu_types if gpu_type.startswith(incomplete)]


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
        vendor: str = typer.Argument(
            ...,
            help="GPU vendor: 'nvidia' for H100/L40s or 'amd' for MI300X",
            autocompletion=vendor_completer,
        ),
        target: str = typer.Argument(
            ..., help="File to compile and profile (e.g., kernel.cu) or command to run"
        ),
        pmc: Optional[str] = typer.Option(
            None,
            "--pmc",
            help="Performance counters to collect (AMD only). Comma-separated list, e.g., 'GRBM_GUI_ACTIVE,SQ_WAVES,SQ_BUSY_CYCLES'",
        ),
        gpu_type: Optional[str] = typer.Option(
            None,
            "--gpu-type",
            help="GPU type: 'h100' (default) or 'l40s' (NVIDIA only)",
            autocompletion=gpu_type_completer,
        ),
    ):
        """Profile a GPU kernel or command on cloud infrastructure.

        Examples:
            chisel profile amd matrix.cpp                                       # Basic profiling
            chisel profile nvidia kernel.cu                                     # NVIDIA H100 profiling
            chisel profile nvidia kernel.cu --gpu-type l40s                     # NVIDIA L40s profiling
            chisel profile amd kernel.cpp --pmc "GRBM_GUI_ACTIVE,SQ_WAVES"     # AMD with counters
        """
        exit_code = handle_profile(vendor=vendor, target=target, pmc=pmc, gpu_type=gpu_type)
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
