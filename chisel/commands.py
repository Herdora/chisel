"""Command handlers for Chisel - contains the business logic for each CLI command."""

from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from chisel.config import Config
from chisel.do_client import DOClient
from chisel.profile_manager import ProfileManager

console = Console()


def handle_configure(token: Optional[str] = None) -> int:
    """Handle the configure command logic.

    Args:
        token: Optional API token from command line

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    config = Config()

    # Check if token already exists
    existing_token = config.token

    if token:
        # Token provided via command line
        api_token = token
    elif existing_token:
        # Token exists in config/env
        console.print("[green]Found existing DigitalOcean API token.[/green]")
        overwrite = Prompt.ask("Do you want to update the token?", choices=["y", "n"], default="n")
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
            table.add_row("Droplet Limit", str(account_data.get("droplet_limit", "N/A")))

            console.print(table)

            console.print(f"\n[green]Configuration saved to:[/green] {config.config_file}")
            console.print("\n[green]✓ Chisel is now configured and ready to use![/green]")
            console.print("\n[cyan]Usage:[/cyan]")
            console.print("  chisel profile nvidia <file_or_command>  # Profile on NVIDIA H100")
            console.print("  chisel profile amd <file_or_command>     # Profile on AMD MI300X")

            return 0

        else:
            console.print("[red]✗ Invalid API token. Please check your token and try again.[/red]")
            return 1

    except Exception as e:
        console.print(f"[red]Error validating token: {e}[/red]")
        console.print(
            "[yellow]Please ensure you have a valid DigitalOcean API token with read and write permissions.[/yellow]"
        )
        return 1


def handle_profile(
    vendor: str, target: str, pmc: Optional[str] = None, gpu_type: Optional[str] = None
) -> int:
    """Handle the profile command logic.

    Args:
        vendor: GPU vendor ('nvidia' or 'amd')
        target: File or command to profile
        pmc: Performance counters for AMD (optional)
        gpu_type: GPU type override for NVIDIA (optional)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Validate vendor
    if vendor not in ["nvidia", "amd"]:
        console.print(f"[red]Error: vendor must be 'nvidia' or 'amd', not '{vendor}'[/red]")
        return 1

    # Validate PMC option
    if pmc and vendor != "amd":
        console.print("[red]Error: --pmc flag is only supported for AMD profiling[/red]")
        return 1

    # Validate GPU type option
    if gpu_type and vendor != "nvidia":
        console.print("[red]Error: --gpu-type flag is only supported for NVIDIA profiling[/red]")
        return 1

    if gpu_type and gpu_type not in ["h100", "l40s"]:
        console.print(f"[red]Error: --gpu-type must be 'h100' or 'l40s', not '{gpu_type}'[/red]")
        return 1

    # Check configuration
    config = Config()
    if not config.token:
        console.print("[red]Error: No API token configured.[/red]")
        console.print(
            "[yellow]Run 'chisel configure' first to set up your DigitalOcean API token.[/yellow]"
        )
        return 1

    try:
        # Use ProfileManager to handle everything
        manager = ProfileManager()
        result = manager.profile(vendor, target, pmc_counters=pmc, gpu_type=gpu_type)

        # Display results
        result.display_summary()

        return 0 if result.success else 1

    except RuntimeError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Profile interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return 1


def handle_version() -> int:
    """Handle the version command logic.

    Returns:
        Exit code (always 0 for success)
    """
    from chisel import __version__

    console.print(f"Chisel version {__version__}")
    return 0
