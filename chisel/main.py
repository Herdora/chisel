"""Chisel CLI - A tool for managing DigitalOcean GPU droplets for HIP kernel development."""

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from typing import Optional

from chisel.config import Config
from chisel.do_client import DOClient
from chisel.droplet import DropletManager
from chisel.ssh_manager import SSHManager

app = typer.Typer(
    name="chisel",
    help="A CLI tool for managing DigitalOcean GPU droplets for HIP kernel development",
    add_completion=False,
)
console = Console()


@app.command()
def configure(
    token: Optional[str] = typer.Option(None, "--token", "-t", help="DigitalOcean API token")
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
            "Do you want to update the token?", 
            choices=["y", "n"], 
            default="n"
        )
        if overwrite.lower() == "n":
            api_token = existing_token
        else:
            api_token = Prompt.ask(
                "Enter your DigitalOcean API token",
                password=True
            )
    else:
        # No token found, prompt for it
        console.print("[yellow]No DigitalOcean API token found.[/yellow]")
        console.print("\nTo get your API token:")
        console.print("1. Go to: https://cloud.digitalocean.com/account/api/")
        console.print("2. Generate a new token with read and write access")
        console.print("3. Copy the token (you won't be able to see it again)\n")
        
        api_token = Prompt.ask(
            "Enter your DigitalOcean API token",
            password=True
        )
    
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
            
            # Get and display balance info
            balance_info = do_client.get_balance()
            if balance_info:
                balance_data = balance_info.get("balance", {})
                console.print(f"\n[cyan]Account Balance:[/cyan] ${balance_data.get('account_balance', 'N/A')}")
                console.print(f"[cyan]Month-to-date Usage:[/cyan] ${balance_data.get('month_to_date_usage', 'N/A')}")
            
            console.print(f"\n[green]Configuration saved to:[/green] {config.config_file}")
            console.print("\n[green]✓ Chisel is now configured and ready to use![/green]")
            
        else:
            console.print("[red]✗ Invalid API token. Please check your token and try again.[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error validating token: {e}[/red]")
        console.print("[yellow]Please ensure you have a valid DigitalOcean API token with read and write permissions.[/yellow]")
        raise typer.Exit(1)


@app.command()
def up():
    """Create or reuse a GPU droplet for development."""
    config = Config()
    
    # Check if configured
    if not config.token:
        console.print("[red]Error: No API token configured.[/red]")
        console.print("[yellow]Run 'chisel configure' first to set up your DigitalOcean API token.[/yellow]")
        raise typer.Exit(1)
    
    try:
        # Initialize clients
        do_client = DOClient(config.token)
        droplet_manager = DropletManager(do_client)
        
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
def down():
    """Destroy the current droplet to stop billing."""
    config = Config()
    
    # Check if configured
    if not config.token:
        console.print("[red]Error: No API token configured.[/red]")
        console.print("[yellow]Run 'chisel configure' first to set up your DigitalOcean API token.[/yellow]")
        raise typer.Exit(1)
    
    try:
        # Initialize clients
        do_client = DOClient(config.token)
        droplet_manager = DropletManager(do_client)
        
        # Confirm destruction
        confirm = Prompt.ask(
            "[yellow]Are you sure you want to destroy the droplet?[/yellow]",
            choices=["y", "n"],
            default="n"
        )
        
        if confirm.lower() == "y":
            success = droplet_manager.down()
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
        console.print("[yellow]Run 'chisel configure' first to set up your DigitalOcean API token.[/yellow]")
        raise typer.Exit(1)
    
    try:
        # Initialize clients
        do_client = DOClient(config.token)
        droplet_manager = DropletManager(do_client)
        
        # Get droplets
        droplets = droplet_manager.list_droplets()
        
        if not droplets:
            console.print("[yellow]No chisel droplets found[/yellow]")
            return
        
        # Create table
        table = Table(title="Chisel Droplets")
        table.add_column("Name", style="cyan")
        table.add_column("IP", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Region", style="blue")
        table.add_column("Size", style="magenta")
        table.add_column("Created", style="white")
        
        for droplet in droplets:
            table.add_row(
                droplet["name"],
                droplet.get("ip", "N/A"),
                droplet["status"],
                droplet["region"]["slug"],
                droplet["size"]["slug"],
                droplet["created_at"][:19].replace("T", " ")
            )
        
        console.print(table)
        
        # Show state info
        state_info = droplet_manager.state.get_droplet_info()
        if state_info:
            console.print(f"\n[cyan]Active droplet from state:[/cyan] {state_info['name']} ({state_info['ip']})")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def sync(
    source: str = typer.Argument(..., help="Source file or directory to sync"),
    destination: Optional[str] = typer.Option(None, "--dest", "-d", help="Destination path on droplet (default: /root/chisel/)")
):
    """Sync files to the droplet."""
    try:
        ssh_manager = SSHManager()
        success = ssh_manager.sync(source, destination)
        if not success:
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def run(
    command: str = typer.Argument(..., help="Command to execute on the droplet")
):
    """Execute a command on the droplet."""
    try:
        ssh_manager = SSHManager()
        exit_code = ssh_manager.run(command)
        raise typer.Exit(exit_code)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def profile(
    command_or_file: str = typer.Argument(..., help="Command to profile, or source file (e.g., 'simple-mm.cpp')"),
    trace: Optional[str] = typer.Option("hip,hsa", "--trace", help="Trace options (hip,hsa,roctx)"),
    output_dir: Optional[str] = typer.Option("./out", "--out", help="Local output directory"),
    open_result: bool = typer.Option(False, "--open", help="Auto-open results with perfetto"),
    compiler_args: Optional[str] = typer.Option("", "--args", help="Additional compiler arguments (for source files)")
):
    """Profile a command or source file with rocprof and pull results locally.
    
    Examples:
        chisel profile simple-mm.cpp              # Auto-compile and profile
        chisel profile "/tmp/my-binary"           # Profile existing binary
        chisel profile "ls -la"                   # Profile any command
        chisel profile kernel.cpp --args "-O3"   # Custom compiler flags
    """
    try:
        ssh_manager = SSHManager()
        
        # Check if it's a source file
        if (command_or_file.endswith(('.cpp', '.c', '.hip', '.cu')) and 
            not command_or_file.startswith('/')):
            
            # It's a source file - sync and compile first
            from pathlib import Path
            source_file = Path(command_or_file)
            
            if not source_file.exists():
                console.print(f"[red]Error: Source file '{command_or_file}' not found[/red]")
                raise typer.Exit(1)
            
            # Sync the source file
            console.print(f"[cyan]Syncing {command_or_file}...[/cyan]")
            if not ssh_manager.sync(str(source_file)):
                console.print("[red]Error: Failed to sync source file[/red]")
                raise typer.Exit(1)
            
            # Determine compiler and remote paths
            remote_source = f"/root/chisel/{source_file.name}"
            remote_binary = f"/tmp/{source_file.stem}"
            
            if source_file.suffix in ['.cpp', '.hip']:
                compiler = "hipcc"
            elif source_file.suffix == '.cu':
                compiler = "nvcc"  # For CUDA files
            else:
                compiler = "gcc"   # For .c files
            
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
        local_archive = ssh_manager.profile(full_command, trace, output_dir, open_result)
        
        if local_archive and open_result:
            console.print(f"[cyan]Opening results with perfetto...[/cyan]")
            import webbrowser
            webbrowser.open("https://ui.perfetto.dev/")
            console.print("[yellow]Load the trace file in perfetto to visualize the results[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
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