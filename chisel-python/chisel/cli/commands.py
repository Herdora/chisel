"""Command handlers for Chisel - contains the business logic for each CLI command."""

from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from chisel.core.chisel_client import ChiselClient
from chisel.core.config import get_token, save_chisel_token, save_token, get_chisel_token, get_auth_mode
from chisel.core.droplet_service import DropletService
from chisel.core.profiling_manager import ProfilingManager
from chisel.core.types.gpu_profiles import get_gpu_type_from_command

console = Console()


def handle_configure(token: Optional[str] = None) -> int:
    """Handle the configure command logic.

    Args:
        token: Optional API token from command line

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    existing_token = get_token()

    if token:
        api_token = token
    elif existing_token:
        console.print("[green]Found existing DigitalOcean API token.[/green]")
        overwrite = Prompt.ask(
            "Do you want to update the token?", choices=["y", "n"], default="n"
        )
        if overwrite.lower() == "n":
            api_token = existing_token
        else:
            api_token = Prompt.ask(
                "Enter your DigitalOcean API token", password=True
            )
    else:
        console.print(
            "[yellow]No DigitalOcean API token found.[/yellow]\n"
            "To get your API token:\n"
            "1. Go to: https://cloud.digitalocean.com/account/api/tokens\n"
            "2. Generate a new token with read and write access\n"
            "3. Copy the token (you won't be able to see it again)\n"
        )
        api_token = Prompt.ask(
            "Enter your DigitalOcean API token", password=True
        )

    console.print("\n[cyan]Validating API token...[/cyan]")

    try:
        droplet_service = DropletService(api_token)
        valid, account_info = droplet_service.validate_token()

        if valid and account_info:
            save_token(api_token)

            console.print(
                "[green]✓ Token validated successfully![/green]\n"
                "\n[green]Configuration saved![/green]\n"
                "\n[green]✓ Chisel is now configured and ready to use![/green]\n"
                "\n[cyan]Usage:[/cyan]\n"
                "  chisel profile --nsys ./kernel.cu        # Profile on NVIDIA\n"
                "  chisel profile --rocprofv3 ./kernel.hip  # Profile on AMD"
            )

            if account_info:
                table = Table(title="Account Information", show_header=False)
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")

                account_data = (
                    account_info.get("account", {})
                    if isinstance(account_info, dict)
                    else {}
                )
                table.add_row("Email", account_data.get("email", "N/A"))
                table.add_row("Status", account_data.get("status", "N/A"))
                table.add_row(
                    "Droplet Limit",
                    str(account_data.get("droplet_limit", "N/A")),
                )

                console.print(table)

            return 0

        else:
            console.print(
                "[red]✗ Invalid API token. Please check your token and try again.[/red]"
            )
            return 1

    except Exception as e:
        console.print(
            f"[red]Error validating token: {e}[/red]\n"
            "[yellow]Please ensure you have a valid DigitalOcean API token with read and write permissions.[/yellow]"
        )
        return 1


def handle_profile(
    command_to_profile: Optional[str],
    files_to_sync: Optional[List[str]],
    gpu_type: Optional[str] = None,
    output_dir: Optional[str] = None,
    rocprofv3: Optional[str] = None,
    rocprof_compute: Optional[str] = None,
    nsys: Optional[str] = None,
    ncompute: Optional[str] = None,
) -> int:
    """Handle the profile command logic.

    Args:
        target: File or command to profile
        gpu_type: GPU type override for NVIDIA (optional)
        output_dir: Output directory for results (optional)
        rocprofv3: Whether to run rocprofv3 (AMD)
        rocprof_compute: Whether to run rocprof-compute (AMD)
        nsys: Whether to run nsys (NVIDIA)
        ncompute: Whether to run ncu (NVIDIA)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if command_to_profile is None:
        console.print("[red]Error: Target file or command is required[/red]")
        return 1

    if rocprof_compute is not None:
        console.print(
            "[red]Error: --rocprof-compute support is not yet implemented[/red]"
        )
        return 1

    profilers_enabled = [
        p for p in [rocprofv3, rocprof_compute, nsys, ncompute] if p is not None
    ]
    if not profilers_enabled:
        console.print(
            "[red]Error: No profiler specified.[/red]\n"
            "[yellow]Use one or more of: --rocprofv3, --nsys, --ncompute[/yellow]"
        )
        return 1
    console.print(f"Profilers enabled: {profilers_enabled}")

    # Determine auth mode and get appropriate tokens
    auth_mode = get_auth_mode()
    
    if auth_mode == "none":
        console.print(
            "[red]Error: No authentication configured.[/red]\n"
            "[yellow]Run 'chisel configure' for direct DigitalOcean access, or 'chisel login <token>' for managed access.[/yellow]"
        )
        return 1
    
    chisel_client = None
    api_token = None
    
    if auth_mode == "managed":
        chisel_token = get_chisel_token()
        chisel_client = ChiselClient(chisel_token)
        
        # Get DO token from backend
        gpu_type_for_token = "mi300x" if rocprofv3 in profilers_enabled else "h100"
        do_response = chisel_client.get_do_token(gpu_type_for_token)
        
        if do_response.get("error"):
            console.print(f"[red]Error: {do_response['error']}[/red]")
            return 1
            
        api_token = do_response.get("do_token")
        console.print("[green]✓ Using managed Chisel access[/green]")
    else:
        api_token = get_token()
        console.print("[green]✓ Using direct DigitalOcean access[/green]")

    if gpu_type is None:
        # TODO: infer gpu type from command; later, we change this to use the --gpu-type flag
        gpu_type = "mi300x" if rocprofv3 in profilers_enabled else "h100"

    files_to_sync = files_to_sync or []
    output_dir = output_dir or "./chisel-results"

    try:
        manager = ProfilingManager(api_token, chisel_client)
        result = manager.profile(
            command_to_profile=command_to_profile,
            gpu_type=get_gpu_type_from_command(gpu_type),
            files_to_sync=files_to_sync,
            output_dir=Path(output_dir),
            rocprofv3_flag=rocprofv3,
            rocprof_compute_flag=rocprof_compute,
            nsys_flag=nsys,
            ncompute_flag=ncompute,
        )
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


def handle_install_completion(shell: Optional[str] = None) -> int:
    """Handle the install-completion command logic.

    Args:
        shell: Shell type to install completion for (bash, zsh, fish, powershell)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import subprocess
    import sys

    console.print("[cyan]Installing shell completion for Chisel...[/cyan]")

    if shell:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chisel.main",
                    "--install-completion",
                    shell,
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                console.print(
                    f"[green]✓ Shell completion installed for {shell}[/green]\n"
                    f"[yellow]Restart your {shell} session or run 'source ~/.{shell}rc' to enable completion[/yellow]"
                )
                return 0
            else:
                console.print(
                    f"[red]Failed to install completion for {shell}: {result.stderr}[/red]"
                )
                return 1

        except Exception as e:
            console.print(f"[red]Error installing completion: {e}[/red]")
            return 1
    else:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "chisel.main", "--install-completion"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                console.print(
                    "[green]✓ Shell completion installed![/green]\n"
                    "[yellow]Restart your shell session to enable completion[/yellow]\n"
                    "\n[cyan]Usage examples with completion:[/cyan]\n"
                    "  chisel prof<TAB>        # Completes to 'profile'\n"
                    "  chisel profile <TAB>    # Shows available flags\n"
                    "  chisel profile --rocprof<TAB>  # Shows '--rocprofv3'"
                )
                return 0
            else:
                console.print(
                    f"[red]Failed to install completion: {result.stderr}[/red]"
                )
                return 1

        except Exception as e:
            console.print(f"[red]Error installing completion: {e}[/red]")
            return 1


def handle_login(token: str) -> int:
    """Handle the login command logic.

    Args:
        token: Chisel token to save

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    console.print("[cyan]Validating Chisel token...[/cyan]")

    try:
        client = ChiselClient(token)
        validation = client.validate_token()

        if validation.get("valid"):
            save_chisel_token(token)

            credits = validation.get("credits_remaining", 0)
            warning = validation.get("warning")

            console.print(
                "[green]✓ Token validated successfully![/green]\n"
                f"[green]Credits remaining: ${credits:.2f}[/green]\n"
                "[green]✓ Chisel is now configured for managed access![/green]\n"
            )

            if warning:
                console.print(f"\n[yellow]{warning}[/yellow]")

            return 0
        else:
            console.print(
                "[red]✗ Invalid Chisel token. Please check your token and try again.[/red]"
            )
            return 1

    except Exception as e:
        console.print(
            f"[red]Error validating token: {e}[/red]\n"
            "[yellow]Please ensure you have a valid Chisel token and internet connection.[/yellow]"
        )
        return 1


def handle_version() -> int:
    """Handle the version command logic.

    Returns:
        Exit code (always 0 for success)
    """
    from chisel import __version__

    console.print(f"Chisel version {__version__}")
    return 0
