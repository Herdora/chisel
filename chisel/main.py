import os
import subprocess
import time
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from chisel.config import Config
from chisel.do_client import DOClient
from chisel.droplet import DropletManager
from chisel.gpu_profiles import AMD_MI300X, GPU_PROFILES
from chisel.ssh_manager import SSHManager
from chisel.state import State

app = typer.Typer(
    name="chisel",
    help="A CLI tool for managing DigitalOcean GPU droplets for HIP kernel development",
    add_completion=False,
)
console = Console()


def resolve_gpu_type(explicit_gpu_type: Optional[str]) -> str:
    """Resolve GPU type from explicit flag or active context."""
    if explicit_gpu_type:
        # Validate explicit GPU type
        if explicit_gpu_type not in GPU_PROFILES:
            console.print(f"[red]Error: Invalid GPU type '{explicit_gpu_type}'[/red]")
            console.print(
                f"[yellow]Available types: {', '.join(GPU_PROFILES.keys())}[/yellow]"
            )
            raise typer.Exit(1)
        return explicit_gpu_type
    
    # Try to get from active context
    state = State()
    active_context = state.get_active_context()
    
    if active_context:
        # Validate context still exists in profiles
        if active_context not in GPU_PROFILES:
            console.print(f"[red]Error: Active context '{active_context}' no longer exists[/red]")
            console.print("Please switch to a valid context: chisel switch <type>")
            console.print(f"[yellow]Available types: {', '.join(GPU_PROFILES.keys())}[/yellow]")
            raise typer.Exit(1)
        return active_context
    
    # No explicit type and no context
    console.print("[red]Error: No GPU context set. Use 'chisel switch <type>' or '--gpu-type' flag.[/red]")
    console.print(f"[yellow]Available types: {', '.join(GPU_PROFILES.keys())}[/yellow]")
    raise typer.Exit(1)


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

            # Get and display balance info
            # balance_info = do_client.get_balance()
            # if balance_info:
            # balance_data = balance_info.get("balance", {})
            # console.print(f"\n[cyan]Account Balance:[/cyan] ${balance_data.get('account_balance', 'N/A')}")
            # console.print(f"[cyan]Month-to-date Usage:[/cyan] ${balance_data.get('month_to_date_usage', 'N/A')}")

            console.print(
                f"\n[green]Configuration saved to:[/green] {config.config_file}"
            )
            console.print(
                "\n[green]✓ Chisel is now configured and ready to use![/green]"
            )

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
def up(
    gpu_type: Optional[str] = typer.Option(
        None, "--gpu-type", help="GPU type to use (amd-mi300x, nvidia-h100)"
    ),
):
    """Create or reuse a GPU droplet for development."""
    config = Config()

    # Resolve GPU type from flag or context
    resolved_gpu_type = resolve_gpu_type(gpu_type)

    # Check if configured
    if not config.token:
        console.print("[red]Error: No API token configured.[/red]")
        console.print(
            "[yellow]Run 'chisel configure' first to set up your DigitalOcean API token.[/yellow]"
        )
        raise typer.Exit(1)

    try:
        # Initialize clients
        do_client = DOClient(config.token)
        gpu_profile = GPU_PROFILES[resolved_gpu_type]
        droplet_manager = DropletManager(do_client, gpu_profile, resolved_gpu_type)

        # Create or find droplet
        droplet = droplet_manager.up()

        # Display success info
        console.print("\n[green]✓ Droplet is ready![/green]")
        console.print(f"[cyan]Name:[/cyan] {droplet['name']}")
        console.print(f"[cyan]IP:[/cyan] {droplet.get('ip', 'N/A')}")
        console.print(f"[cyan]Region:[/cyan] {droplet['region']['slug']}")
        console.print(f"[cyan]Size:[/cyan] {droplet['size']['slug']}")
        console.print(f"\n[yellow]SSH:[/yellow] ssh root@{droplet.get('ip', 'N/A')}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def down(
    gpu_type: Optional[str] = typer.Option(
        None, "--gpu-type", help="GPU type to destroy (amd-mi300x, nvidia-h100)"
    ),
):
    """Destroy the current droplet to stop billing."""
    config = Config()

    # Check if configured
    if not config.token:
        console.print("[red]Error: No API token configured.[/red]")
        console.print(
            "[yellow]Run 'chisel configure' first to set up your DigitalOcean API token.[/yellow]"
        )
        raise typer.Exit(1)

    # Resolve GPU type from flag or context
    resolved_gpu_type = resolve_gpu_type(gpu_type)

    try:
        # Initialize clients
        do_client = DOClient(config.token)
        gpu_profile = GPU_PROFILES[resolved_gpu_type]
        droplet_manager = DropletManager(do_client, gpu_profile, resolved_gpu_type)

        # Confirm destruction
        confirm = Prompt.ask(
            f"[yellow]Are you sure you want to destroy the {resolved_gpu_type} droplet?[/yellow]",
            choices=["y", "n"],
            default="n",
        )

        if confirm.lower() == "y":
            success = droplet_manager.down()
            if success and not gpu_type:  # Only clear context if using active context (not explicit --gpu-type)
                state = State()
                if state.get_active_context() == resolved_gpu_type:
                    state.clear_active_context()
                    console.print(f"[yellow]Cleared active context (destroyed {resolved_gpu_type} droplet)[/yellow]")
            if not success:
                raise typer.Exit(1)
        else:
            console.print("[yellow]Cancelled[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list():
    """List all chisel droplets."""
    config = Config()

    # Check if configured
    if not config.token:
        console.print("[red]Error: No API token configured.[/red]")
        console.print(
            "[yellow]Run 'chisel configure' first to set up your DigitalOcean API token.[/yellow]"
        )
        raise typer.Exit(1)

    try:
        # Initialize clients
        do_client = DOClient(config.token)
        # For list command, we don't need a specific GPU type - use AMD as default for the manager
        droplet_manager = DropletManager(do_client, AMD_MI300X, "amd-mi300x")

        # Get droplets
        droplets = droplet_manager.list_droplets()

        if not droplets:
            console.print("[yellow]No chisel droplets found[/yellow]")
            return

        # Create table
        table = Table(title="Chisel Droplets")
        table.add_column("Name", style="cyan")
        table.add_column("GPU Type", style="magenta")
        table.add_column("IP", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Region", style="blue")
        table.add_column("Size", style="white")
        table.add_column("Cost/Hour", style="red")
        table.add_column("Created", style="white")

        for droplet in droplets:
            # Determine GPU type and cost from droplet size
            size = droplet["size"]["slug"]
            if "mi300x" in size:
                gpu_type = "amd-mi300x"
                cost_per_hour = "$1.99"
            elif "h100" in size:
                gpu_type = "nvidia-h100"
                cost_per_hour = "$4.89"
            else:
                gpu_type = "unknown"
                cost_per_hour = "N/A"

            table.add_row(
                droplet["name"],
                gpu_type,
                droplet.get("ip", "N/A"),
                droplet["status"],
                droplet["region"]["slug"],
                size,
                cost_per_hour,
                droplet["created_at"][:19].replace("T", " "),
            )

        console.print(table)

        # Show state info for all tracked droplets
        all_droplets = droplet_manager.state.get_all_droplets()
        if all_droplets:
            console.print("\n[cyan]Tracked droplets in local state:[/cyan]")
            for gpu_type, state_info in all_droplets.items():
                if state_info:
                    console.print(
                        f"  {gpu_type}: {state_info['name']} ({state_info['ip']})"
                    )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def sync(
    source: str = typer.Argument(..., help="Source file or directory to sync"),
    gpu_type: Optional[str] = typer.Option(
        None, "--gpu-type", help="GPU type to sync to (amd-mi300x, nvidia-h100)"
    ),
    destination: Optional[str] = typer.Option(
        None,
        "--dest",
        "-d",
        help="Destination path on droplet (default: /root/chisel/)",
    ),
):
    """Sync files to the droplet."""
    # Resolve GPU type from flag or context
    resolved_gpu_type = resolve_gpu_type(gpu_type)

    try:
        ssh_manager = SSHManager()
        success = ssh_manager.sync(source, destination, resolved_gpu_type)
        if not success:
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def run(
    command: str = typer.Argument(..., help="Command to execute on the droplet"),
    gpu_type: Optional[str] = typer.Option(
        None, "--gpu-type", help="GPU type to run command on (amd-mi300x, nvidia-h100)"
    ),
):
    """Execute a command on the droplet."""
    # Resolve GPU type from flag or context
    resolved_gpu_type = resolve_gpu_type(gpu_type)

    try:
        ssh_manager = SSHManager()
        exit_code = ssh_manager.run(command, resolved_gpu_type)
        raise typer.Exit(exit_code)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def profile(
    command_or_file: str = typer.Argument(
        ..., help="Command to profile, or source file (e.g., 'simple-mm.cpp')"
    ),
    gpu_type: Optional[str] = typer.Option(
        None, "--gpu-type", help="GPU type to profile on (amd-mi300x, nvidia-h100)"
    ),
    trace: Optional[str] = typer.Option(
        "hip,hsa", "--trace", help="Trace options (hip,hsa,roctx)"
    ),
    output_dir: Optional[str] = typer.Option(
        "./out", "--out", help="Local output directory"
    ),
    open_result: bool = typer.Option(
        False, "--open", help="Auto-open results with perfetto"
    ),
    compiler_args: Optional[str] = typer.Option(
        "", "--args", help="Additional compiler arguments (for source files)"
    ),
):
    """Profile a command or source file with rocprof and pull results locally.

    Examples:
        chisel profile simple-mm.cpp                                     # Auto-compile and profile (uses context)
        chisel profile "/tmp/my-binary" --gpu-type nvidia-h100          # Profile existing binary
        chisel profile "ls -la"                                          # Profile any command (uses context)
        chisel profile kernel.cpp --args "-O3"                          # Custom compiler flags (uses context)
    """
    # Resolve GPU type from flag or context
    resolved_gpu_type = resolve_gpu_type(gpu_type)

    try:
        ssh_manager = SSHManager()

        # Check if it's a source file
        if command_or_file.endswith(
            (".cpp", ".c", ".hip", ".cu")
        ) and not command_or_file.startswith("/"):
            # It's a source file - sync and compile first
            from pathlib import Path

            source_file = Path(command_or_file)

            if not source_file.exists():
                console.print(
                    f"[red]Error: Source file '{command_or_file}' not found[/red]"
                )
                raise typer.Exit(1)

            # Sync the source file
            console.print(f"[cyan]Syncing {command_or_file}...[/cyan]")
            if not ssh_manager.sync(str(source_file), None, resolved_gpu_type):
                console.print("[red]Error: Failed to sync source file[/red]")
                raise typer.Exit(1)

            # Determine compiler and remote paths
            remote_source = f"/root/chisel/{source_file.name}"
            remote_binary = f"/tmp/{source_file.stem}"

            if source_file.suffix in [".cpp", ".hip"]:
                compiler = "hipcc"
            elif source_file.suffix == ".cu":
                compiler = "nvcc"  # For CUDA files
            else:
                compiler = "gcc"  # For .c files

            # Build compile and run command
            compile_cmd = f"{compiler} {remote_source} -o {remote_binary}"
            if compiler_args:
                compile_cmd += f" {compiler_args}"

            full_command = f"{compile_cmd} && {remote_binary}"

            console.print(f"[cyan]Compiling with: {compile_cmd}[/cyan]")

        else:
            # It's a direct command
            full_command = command_or_file

        # Profile the command
        local_archive = ssh_manager.profile(
            full_command, resolved_gpu_type, trace, output_dir, open_result
        )

        if local_archive and open_result:
            console.print("[cyan]Opening results with perfetto...[/cyan]")
            import webbrowser

            webbrowser.open("https://ui.perfetto.dev/")
            console.print(
                "[yellow]Load the trace file in perfetto to visualize the results[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def pull(
    remote_path: str = typer.Argument(
        ..., help="Remote file or directory path to pull"
    ),
    gpu_type: Optional[str] = typer.Option(
        None, "--gpu-type", help="GPU type to pull from (amd-mi300x, nvidia-h100)"
    ),
    local_path: Optional[str] = typer.Option(
        None,
        "--local",
        "-l",
        help="Local destination path (default: current directory)",
    ),
):
    """Pull files or directories from the droplet to local machine."""
    # Resolve GPU type from flag or context
    resolved_gpu_type = resolve_gpu_type(gpu_type)

    try:
        ssh_manager = SSHManager()
        success = ssh_manager.pull(remote_path, local_path, resolved_gpu_type)
        if not success:
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def sweep(
    hours: float = typer.Option(
        6.0,
        "--hours",
        help="Droplets older than this many hours will be flagged for cleanup",
    ),
    auto_confirm: bool = typer.Option(
        False, "--yes", "-y", help="Auto-confirm destruction without prompting"
    ),
):
    """Clean up old droplets that have been running too long."""
    config = Config()

    # Check if configured
    if not config.token:
        console.print("[red]Error: No API token configured.[/red]")
        console.print(
            "[yellow]Run 'chisel configure' first to set up your DigitalOcean API token.[/yellow]"
        )
        raise typer.Exit(1)

    try:
        # Initialize clients
        do_client = DOClient(config.token)
        # For sweep command, we don't need a specific GPU type - use AMD as default for the manager
        droplet_manager = DropletManager(do_client, AMD_MI300X, "amd-mi300x")

        # Get all droplets
        droplets = droplet_manager.list_droplets()

        if not droplets:
            console.print("[yellow]No chisel droplets found[/yellow]")
            return

        # Calculate costs and filter old droplets
        old_droplets = []
        total_estimated_cost = 0.0

        for droplet in droplets:
            try:
                from datetime import datetime, timezone

                created_at = datetime.fromisoformat(
                    droplet["created_at"].replace("Z", "+00:00")
                )
                now = datetime.now(timezone.utc)
                uptime_hours = (now - created_at).total_seconds() / 3600
                cost = uptime_hours * 1.99  # $1.99/hour for gpu-mi300x1-192gb

                if uptime_hours >= hours:
                    old_droplets.append(
                        {"droplet": droplet, "uptime_hours": uptime_hours, "cost": cost}
                    )

                total_estimated_cost += cost

            except Exception:
                # If we can't parse the date, skip this droplet
                continue

        if not old_droplets:
            console.print(
                f"[green]No droplets found running longer than {hours} hours[/green]"
            )
            console.print(
                f"[cyan]Total estimated cost for all droplets: ${total_estimated_cost:.2f}[/cyan]"
            )
            return

        # Show summary
        console.print(
            f"\n[yellow]Found {len(old_droplets)} droplet(s) running longer than {hours} hours:[/yellow]\n"
        )

        # Create table
        table = Table(title="Droplets to Clean Up")
        table.add_column("Name", style="cyan")
        table.add_column("IP", style="green")
        table.add_column("Uptime", style="yellow")
        table.add_column("Est. Cost", style="red")
        table.add_column("Status", style="blue")

        total_old_cost = 0.0
        for item in old_droplets:
            droplet = item["droplet"]
            uptime_hours = item["uptime_hours"]
            cost = item["cost"]
            total_old_cost += cost

            table.add_row(
                droplet["name"],
                droplet.get("ip", "N/A"),
                f"{uptime_hours:.1f}h",
                f"${cost:.2f}",
                droplet["status"],
            )

        console.print(table)
        console.print(
            f"\n[red]Total cost for old droplets: ${total_old_cost:.2f}[/red]"
        )
        console.print(
            f"[cyan]Total cost for all droplets: ${total_estimated_cost:.2f}[/cyan]"
        )

        # Confirm cleanup
        if not auto_confirm:
            confirm = Prompt.ask(
                f"\n[yellow]Destroy {len(old_droplets)} droplet(s)?[/yellow]",
                choices=["y", "n"],
                default="n",
            )

            if confirm.lower() != "y":
                console.print("[yellow]Cancelled[/yellow]")
                return

        # Destroy droplets
        destroyed_count = 0
        for item in old_droplets:
            droplet = item["droplet"]
            try:
                console.print(f"[yellow]Destroying {droplet['name']}...[/yellow]")
                droplet_manager.destroy_droplet(droplet["id"])
                destroyed_count += 1
                console.print(f"[green]✓ Destroyed {droplet['name']}[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to destroy {droplet['name']}: {e}[/red]")

        console.print(
            f"\n[green]Successfully destroyed {destroyed_count}/{len(old_droplets)} droplets[/green]"
        )

        # Clear state for any destroyed droplets
        all_droplets = droplet_manager.state.get_all_droplets()
        for gpu_type, state_info in all_droplets.items():
            if state_info:
                for item in old_droplets:
                    if item["droplet"]["id"] == state_info["droplet_id"]:
                        droplet_manager.state.clear_droplet(gpu_type)
                        console.print(
                            f"[yellow]Cleared {gpu_type} droplet state (droplet was destroyed)[/yellow]"
                        )
                        break

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


""" TODO: add this automatically to configuration """


@app.command()
def ssh_setup():
    """Set up SSH access from current machine to chisel droplets."""
    config = Config()

    # Check if configured
    if not config.token:
        console.print("[red]Error: No API token configured.[/red]")
        console.print(
            "[yellow]Run 'chisel configure' first to set up your DigitalOcean API token.[/yellow]"
        )
        raise typer.Exit(1)

    # Ensure local SSH key exists
    ssh_key_paths = [
        (
            os.path.expanduser("~/.ssh/id_ed25519"),
            os.path.expanduser("~/.ssh/id_ed25519.pub"),
        ),
        (os.path.expanduser("~/.ssh/id_rsa"), os.path.expanduser("~/.ssh/id_rsa.pub")),
        (
            os.path.expanduser("~/.ssh/id_ecdsa"),
            os.path.expanduser("~/.ssh/id_ecdsa.pub"),
        ),
    ]

    public_key_path = None
    for _, pub_path in ssh_key_paths:
        if os.path.exists(pub_path):
            public_key_path = pub_path
            break

    if not public_key_path:
        console.print(
            "[yellow]No SSH key found. Generating new ED25519 key...[/yellow]"
        )
        os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)
        private_path = os.path.expanduser("~/.ssh/id_ed25519")
        public_key_path = os.path.expanduser("~/.ssh/id_ed25519.pub")

        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", private_path, "-N", "", "-q"],
            check=True,
        )
        console.print("[green]✓ Generated new SSH key[/green]")

    # Read public key
    with open(public_key_path, "r") as f:
        public_key = f.read().strip()

    console.print(
        f"\n[cyan]Your SSH public key ({os.path.basename(public_key_path)}):[/cyan]"
    )
    console.print(Panel(public_key, border_style="green"))

    # Check if we're on a DigitalOcean droplet
    try:
        import requests

        response = requests.get("http://169.254.169.254/metadata/v1/id", timeout=1)
        is_on_do_droplet = response.status_code == 200
    except:
        is_on_do_droplet = False

    if is_on_do_droplet:
        console.print("\n[yellow]You're running on a DigitalOcean droplet![/yellow]")
        console.print("\n[cyan]To use chisel from this droplet:[/cyan]")
        console.print("1. If automatic addition fails, manually add the SSH key:")
        console.print("   - Go to: https://amd.digitalocean.com/account/keys")
        console.print("   - Click 'Add SSH Key'")
        console.print("   - Paste the key and give it a name (e.g., 'chisel-droplet')")
        console.print("2. Run 'chisel down' to destroy any existing chisel-dev droplet")
        console.print("3. Run 'chisel up' to create a new droplet with your key")
        console.print(
            "\n[green]Then you'll be able to use chisel sync/run/profile from this droplet![/green]"
        )
    else:
        console.print(
            "\n[cyan]If automatic addition fails, manually add this key:[/cyan]"
        )
        console.print("1. Go to: https://amd.digitalocean.com/account/keys")
        console.print("2. Click 'Add SSH Key'")
        console.print("3. Paste the key above and give it a name")
        console.print(
            "4. Future droplets created with 'chisel up' will include this key"
        )

    # Try to add key to DO account via API
    console.print(
        "\n[cyan]Attempting to add SSH key to your DigitalOcean account...[/cyan]"
    )
    try:
        do_client = DOClient(config.token)
        key_name = f"chisel-{os.uname().nodename}-{int(time.time())}"

        response = do_client.client.ssh_keys.create(
            body={"name": key_name, "public_key": public_key}
        )

        if response and "ssh_key" in response:
            console.print(
                f"[green]✓ Successfully added SSH key '{key_name}' to your DigitalOcean account![/green]"
            )
            if is_on_do_droplet:
                console.print(
                    "[yellow]Now run 'chisel down' then 'chisel up' to recreate any existing droplets with this key.[/yellow]"
                )
            else:
                console.print(
                    "[yellow]Future droplets created with 'chisel up' will include this key.[/yellow]"
                )
        else:
            console.print("[red]✗ Failed to add SSH key to DigitalOcean account[/red]")
            console.print(
                "[yellow]Please add it manually using the instructions above.[/yellow]"
            )
    except Exception as e:
        if (
            "already in use" in str(e).lower()
            or "ssh key is already in use" in str(e).lower()
        ):
            console.print(
                "[green]✓ This SSH key is already in your DigitalOcean account[/green]"
            )
            if is_on_do_droplet:
                console.print(
                    "[yellow]If chisel sync/run still fails, run 'chisel down' then 'chisel up' to recreate the droplet.[/yellow]"
                )
        else:
            console.print(
                f"[red]✗ Could not automatically add key to DigitalOcean account: {e}[/red]"
            )
            console.print(
                "[yellow]Please add it manually using the instructions above.[/yellow]"
            )


@app.command()
def switch(
    gpu_type: str = typer.Argument(..., help="GPU type to switch to (amd-mi300x, nvidia-h100)")
):
    """Switch to a specific GPU context."""
    # Validate GPU type
    if gpu_type not in GPU_PROFILES:
        console.print(f"[red]Error: Unknown GPU type '{gpu_type}'[/red]")
        console.print(f"[yellow]Available types: {', '.join(GPU_PROFILES.keys())}[/yellow]")
        raise typer.Exit(1)
    
    # Set active context
    state = State()
    state.set_active_context(gpu_type)
    
    console.print(f"[green]✓ Switched to {gpu_type} context[/green]")


@app.command()
def context():
    """Show the current active GPU context and available droplets."""
    state = State()
    active_context = state.get_active_context()
    
    if active_context:
        console.print(f"Active context: [cyan]{active_context}[/cyan]")
        
        # Show active droplet info if available
        tracked_droplet = state.get_droplet_id(active_context)
        if tracked_droplet:
            console.print(f"Active droplet: [green]{tracked_droplet}[/green]")
        else:
            console.print("[yellow]No droplet running for this context[/yellow]")
            console.print(f"[yellow]Use 'chisel up' to create a {active_context} droplet[/yellow]")
    else:
        console.print("[yellow]No active context set[/yellow]")
        
        # Show available droplets and suggest how to use them
        tracked_droplets = state.get_all_tracked_droplets()
        if tracked_droplets:
            console.print("\n[cyan]Available droplets:[/cyan]")
            for gpu_type, droplet_info in tracked_droplets.items():
                console.print(f"  {gpu_type}: [green]{droplet_info['name']}[/green] ({droplet_info['ip']})")
            console.print(f"\n[yellow]Use 'chisel switch <gpu-type>' to set a context[/yellow]")
        else:
            console.print(f"\n[yellow]Available GPU types: {', '.join(GPU_PROFILES.keys())}[/yellow]")
            console.print("[yellow]Use 'chisel switch <type>' to set a context[/yellow]")
            console.print("[yellow]Use 'chisel up' to create a droplet[/yellow]")


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
