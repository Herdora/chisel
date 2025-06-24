"""Chisel - Seamless GPU kernel profiling on cloud infrastructure."""

from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from chisel.config import Config
from chisel.do_client import DOClient
from chisel.profile_manager import ProfileManager

app = typer.Typer(
    name="chisel",
    help="Seamless GPU kernel profiling on cloud infrastructure",
    add_completion=False,
)
console = Console()


@app.command()
def configure(
    token: Optional[str] = typer.Option(
        None, "--token", "-t", help="DigitalOcean API token"
    ),
):
    """Configure Chisel with your DigitalOcean API token."""
    config = Config()

    # Check if token already exists
    existing_token = config.token

    if token:
        # Token provided via command line
        api_token = token
    elif existing_token:
        # Token exists in config/env
        console.print("[green]Found existing DigitalOcean API token.[/green]")
        overwrite = Prompt.ask(
            "Do you want to update the token?", choices=["y", "n"], default="n"
        )
        if overwrite.lower() == "n":
            api_token = existing_token
        else:
            api_token = Prompt.ask("Enter your DigitalOcean API token", password=True)
    else:
        # No token found, prompt for it
        console.print("[yellow]No DigitalOcean API token found.[/yellow]")
        console.print("\nTo get your API token:")
        console.print("1. Go to: https://amd.digitalocean.com/account/api/tokens")
        console.print("2. Generate a new token with read and write access")
        console.print("3. Copy the token (you won't be able to see it again)\n")

        api_token = Prompt.ask("Enter your DigitalOcean API token", password=True)

    # Validate token
    console.print("\n[cyan]Validating API token...[/cyan]")

    try:
        do_client = DOClient(api_token)
        valid, account_info = do_client.validate_token()

        if valid and account_info:
            # Save token to config
            config.token = api_token

            # Display account info
            console.print("[green]✓ Token validated successfully![/green]\n")

            # Create account info table
            table = Table(title="Account Information", show_header=False)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")

            account_data = account_info.get("account", {})
            table.add_row("Email", account_data.get("email", "N/A"))
            table.add_row("Status", account_data.get("status", "N/A"))
            table.add_row(
                "Droplet Limit", str(account_data.get("droplet_limit", "N/A"))
            )

            console.print(table)

            console.print(
                f"\n[green]Configuration saved to:[/green] {config.config_file}"
            )
            console.print(
                "\n[green]✓ Chisel is now configured and ready to use![/green]"
            )
            console.print("\n[cyan]Usage:[/cyan]")
            console.print("  chisel profile nvidia <file_or_command>  # Profile on NVIDIA H100")
            console.print("  chisel profile amd <file_or_command>     # Profile on AMD MI300X")

        else:
            console.print(
                "[red]✗ Invalid API token. Please check your token and try again.[/red]"
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error validating token: {e}[/red]")
        console.print(
            "[yellow]Please ensure you have a valid DigitalOcean API token with read and write permissions.[/yellow]"
        )
        raise typer.Exit(1)


@app.command()
def profile(
    vendor: str = typer.Argument(
        ..., 
        help="GPU vendor: 'nvidia' for H100/L40s or 'amd' for MI300X"
    ),
    target: str = typer.Argument(
        ..., 
        help="File to compile and profile (e.g., kernel.cu) or command to run"
    ),
    pmc: Optional[str] = typer.Option(
        None,
        "--pmc",
        help="Performance counters to collect (AMD only). Comma-separated list, e.g., 'GRBM_GUI_ACTIVE,SQ_WAVES,SQ_BUSY_CYCLES'"
    ),
    gpu_type: Optional[str] = typer.Option(
        None,
        "--gpu-type",
        help="GPU type: 'h100' (default) or 'l40s' (NVIDIA only)"
    ),
):
    """Profile a GPU kernel or command on cloud infrastructure.
    
    Examples:
        chisel profile amd matrix.cpp                                       # Basic profiling
        chisel profile nvidia kernel.cu                                     # NVIDIA H100 profiling  
        chisel profile nvidia kernel.cu --gpu-type l40s                     # NVIDIA L40s profiling
        chisel profile amd kernel.cpp --pmc "GRBM_GUI_ACTIVE,SQ_WAVES"     # AMD with counters
    """
    # Validate vendor
    if vendor not in ["nvidia", "amd"]:
        console.print(f"[red]Error: vendor must be 'nvidia' or 'amd', not '{vendor}'[/red]")
        raise typer.Exit(1)
    
    # Validate PMC option
    if pmc and vendor != "amd":
        console.print("[red]Error: --pmc flag is only supported for AMD profiling[/red]")
        raise typer.Exit(1)
    
    # Validate GPU type option
    if gpu_type and vendor != "nvidia":
        console.print("[red]Error: --gpu-type flag is only supported for NVIDIA profiling[/red]")
        raise typer.Exit(1)
    
    if gpu_type and gpu_type not in ["h100", "l40s"]:
        console.print(f"[red]Error: --gpu-type must be 'h100' or 'l40s', not '{gpu_type}'[/red]")
        raise typer.Exit(1)
    
    # Check configuration
    config = Config()
    if not config.token:
        console.print("[red]Error: No API token configured.[/red]")
        console.print("[yellow]Run 'chisel configure' first to set up your DigitalOcean API token.[/yellow]")
        raise typer.Exit(1)
    
    try:
        # Use ProfileManager to handle everything
        manager = ProfileManager()
        result = manager.profile(vendor, target, pmc_counters=pmc, gpu_type=gpu_type)
        
        # Display results
        result.display_summary()
        
        if not result.success:
            raise typer.Exit(1)
            
    except RuntimeError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Profile interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show Chisel version."""
    from chisel import __version__

    console.print(f"Chisel version {__version__}")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()