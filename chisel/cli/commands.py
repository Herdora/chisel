"""Command handlers for Chisel - contains the business logic for each CLI command."""

from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from chisel.core.config import Config
from chisel.core.do_client import DOClient
from chisel.core.profiling_manager import ProfilingManager

console = Console()


def handle_configure(token: Optional[str] = None) -> int:
    """Handle the configure command logic.

    Args:
        token: Optional API token from command line

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    config = Config()
    existing_token = config.token

    if token:
        api_token = token
    elif existing_token:
        console.print("[green]Found existing DigitalOcean API token.[/green]")
        overwrite = Prompt.ask("Do you want to update the token?", choices=["y", "n"], default="n")
        if overwrite.lower() == "n":
            api_token = existing_token
        else:
            api_token = Prompt.ask("Enter your DigitalOcean API token", password=True)
    else:
        console.print(
            "[yellow]No DigitalOcean API token found.[/yellow]\n"
            "To get your API token:\n"
            "1. Go to: https://amd.digitalocean.com/account/api/tokens\n"
            "2. Generate a new token with read and write access\n"
            "3. Copy the token (you won't be able to see it again)\n"
        )
        api_token = Prompt.ask("Enter your DigitalOcean API token", password=True)

    console.print("\n[cyan]Validating API token...[/cyan]")

    try:
        do_client = DOClient(api_token)
        valid, account_info = do_client.validate_token()

        if valid and account_info:
            config.token = api_token

            console.print(
                "[green]✓ Token validated successfully![/green]\n"
                "\n[green]Configuration saved to:[/green] {}\n"
                "\n[green]✓ Chisel is now configured and ready to use![/green]\n"
                "\n[cyan]Usage:[/cyan]\n"
                "  chisel profile nvidia <file_or_command>  # Profile on NVIDIA H100\n"
                "  chisel profile amd <file_or_command>     # Profile on AMD MI300X".format(
                    config.config_file
                )
            )

            table = Table(title="Account Information", show_header=False)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")

            account_data = account_info.get("account", {})
            table.add_row("Email", account_data.get("email", "N/A"))
            table.add_row("Status", account_data.get("status", "N/A"))
            table.add_row("Droplet Limit", str(account_data.get("droplet_limit", "N/A")))

            console.print(table)

            return 0

        else:
            console.print("[red]✗ Invalid API token. Please check your token and try again.[/red]")
            return 1

    except Exception as e:
        console.print(
            f"[red]Error validating token: {e}[/red]\n"
            "[yellow]Please ensure you have a valid DigitalOcean API token with read and write permissions.[/yellow]"
        )
        return 1


def handle_profile(
    target: Optional[str],
    pmc_counters: Optional[str] = None,
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
        pmc_counters: Performance counters for AMD (optional)
        gpu_type: GPU type override for NVIDIA (optional)
        output_dir: Output directory for results (optional)
        rocprofv3: Whether to run rocprofv3 (AMD)
        rocprof_compute: Whether to run rocprof-compute (AMD)
        nsys: Whether to run nsys (NVIDIA)
        ncompute: Whether to run ncu (NVIDIA)

    Returns:
        Exit code (0 for success, 1 for failure)
    """

    # Handle the case where target is None but a profiler flag has the target value
    # This happens when Typer interprets the target as a value for a profiler flag
    if target is None:
        # Check if any profiler flag has a value that looks like a file path
        potential_targets = []
        for profiler_name, profiler_value in [
            ("rocprofv3", rocprofv3),
            ("rocprof_compute", rocprof_compute),
            ("nsys", nsys),
            ("ncompute", ncompute),
        ]:
            if (
                profiler_value
                and profiler_value != "None"
                and (
                    "/" in profiler_value
                    or profiler_value.endswith((".hip", ".cu", ".cpp", ".c", ".py"))
                )
            ):
                potential_targets.append((profiler_name, profiler_value))

        if len(potential_targets) == 1:
            # Found exactly one potential target, use it
            profiler_name, target_value = potential_targets[0]
            console.print(
                f"[yellow]Warning: Target '{target_value}' was interpreted as value for --{profiler_name}. Using it as target.[/yellow]"
            )
            target = target_value
            # Clear the profiler flag
            if profiler_name == "rocprofv3":
                rocprofv3 = ""
            elif profiler_name == "rocprof_compute":
                rocprof_compute = ""
            elif profiler_name == "nsys":
                nsys = ""
            elif profiler_name == "ncompute":
                ncompute = ""
        elif len(potential_targets) > 1:
            console.print(
                "[red]Error: Multiple potential targets found. Please specify the target explicitly.[/red]"
            )
            return 1
        else:
            console.print("[red]Error: Target file or command is required[/red]")
            return 1

    # Validate target parameter
    if target is None:
        console.print("[red]Error: Target file or command is required[/red]")
        return 1

    # At this point, target is guaranteed to be a string
    assert target is not None
    target_str = target

    # Check that at least one profiler is specified
    profilers_enabled = [p for p in [rocprofv3, rocprof_compute, nsys, ncompute] if p is not None]
    if not profilers_enabled:
        console.print(
            "[red]Error: No profiler specified.[/red]\n"
            "[yellow]Use one or more of: --rocprofv3, --rocprof-compute, --nsys, --ncompute[/yellow]"
        )
        return 1

    # Determine vendor based on profiler flags
    amd_profilers = [p for p in [rocprofv3, rocprof_compute] if p is not None]
    nvidia_profilers = [p for p in [nsys, ncompute] if p is not None]

    if amd_profilers and nvidia_profilers:
        console.print("[red]Error: Cannot mix AMD and NVIDIA profilers in the same command[/red]")
        return 1

    vendor = "amd" if amd_profilers else "nvidia"

    # Validation
    if pmc_counters and vendor != "amd":
        console.print("[red]Error: --pmc flag is only supported for AMD profiling[/red]")
        return 1
    if gpu_type and vendor != "nvidia":
        console.print("[red]Error: --gpu-type flag is only supported for NVIDIA profiling[/red]")
        return 1
    if gpu_type and gpu_type not in ["h100", "l40s"]:
        console.print(f"[red]Error: --gpu-type must be 'h100' or 'l40s', not '{gpu_type}'[/red]")
        return 1
    if not Config().token:
        console.print(
            "[red]Error: No API token configured.[/red]\n"
            "[yellow]Run 'chisel configure' first to set up your DigitalOcean API token.[/yellow]"
        )
        return 1

    try:
        manager = ProfilingManager()

        # Build profiler commands - if string is provided, use target + extra flags, otherwise just target
        def build_cmd(base_target: str, profiler_value: Optional[str]) -> Optional[str]:
            if profiler_value is None:
                return None
            elif profiler_value == "":  # Boolean flag was used (empty string)
                return base_target
            else:  # String value was provided (extra flags)
                return f"{base_target} {profiler_value}"

        result = manager.profile(
            vendor=vendor,
            target=target_str,
            pmc_counters=pmc_counters,
            gpu_type=gpu_type,
            output_dir=output_dir,
            rocprofv3_cmd=build_cmd(target_str, rocprofv3),
            rocprof_compute_cmd=build_cmd(target_str, rocprof_compute),
            nsys_cmd=build_cmd(target_str, nsys),
            ncompute_cmd=build_cmd(target_str, ncompute),
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
    from pathlib import Path

    console.print("[cyan]Installing shell completion for Chisel...[/cyan]")

    if shell:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "chisel.main", "--install-completion", shell],
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
                    "  chisel profile <TAB>    # Shows 'nvidia' and 'amd'\n"
                    "  chisel profile nvidia --gpu<TAB>  # Shows '--gpu-type'"
                )
                return 0
            else:
                console.print(f"[red]Failed to install completion: {result.stderr}[/red]")
                return 1

        except Exception as e:
            console.print(f"[red]Error installing completion: {e}[/red]")
            return 1


def handle_version() -> int:
    """Handle the version command logic.

    Returns:
        Exit code (always 0 for success)
    """
    from chisel import __version__

    console.print(f"Chisel version {__version__}")
    return 0
