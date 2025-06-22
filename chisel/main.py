"""Chisel CLI - A tool for managing DigitalOcean GPU droplets for HIP kernel development."""

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from typing import Optional

from chisel.config import Config
from chisel.do_client import DOClient

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
def version():
    """Show Chisel version."""
    from chisel import __version__
    console.print(f"Chisel version {__version__}")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()