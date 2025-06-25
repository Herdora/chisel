"""Command-line interface for Chisel - pure argument parser."""

from typing import Optional

import typer

from chisel.commands import handle_configure, handle_profile, handle_version


def create_app() -> typer.Typer:
    """Create and configure the Typer app with all commands."""
    app = typer.Typer(
        name="chisel",
        help="Seamless GPU kernel profiling on cloud infrastructure",
        add_completion=False,
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
            ..., help="GPU vendor: 'nvidia' for H100/L40s or 'amd' for MI300X"
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
            None, "--gpu-type", help="GPU type: 'h100' (default) or 'l40s' (NVIDIA only)"
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
