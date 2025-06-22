"""SSH and sync operations for chisel."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import paramiko
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .state import State

console = Console()


class SSHManager:
    def __init__(self):
        self.state = State()
        
    def get_droplet_info(self) -> Optional[Dict[str, Any]]:
        """Get droplet info from state."""
        return self.state.get_droplet_info()
        
    def sync(self, source: str, destination: Optional[str] = None) -> bool:
        """Sync files to the droplet using rsync."""
        droplet_info = self.get_droplet_info()
        if not droplet_info:
            console.print("[red]Error: No active droplet found[/red]")
            console.print("[yellow]Run 'chisel up' first to create a droplet[/yellow]")
            return False
            
        # Default destination
        if destination is None:
            destination = "/root/chisel/"
            
        # Ensure source exists
        source_path = Path(source).resolve()
        if not source_path.exists():
            console.print(f"[red]Error: Source path '{source}' does not exist[/red]")
            return False
            
        # Build rsync command
        ip = droplet_info["ip"]
        
        # Add trailing slash for directories to sync contents
        if source_path.is_dir() and not source.endswith('/'):
            source = str(source_path) + '/'
        else:
            source = str(source_path)
            
        rsync_cmd = [
            "rsync",
            "-avz",  # archive, verbose, compress
            "--progress",
            "-e", "ssh -o StrictHostKeyChecking=no",
            source,
            f"root@{ip}:{destination}"
        ]
        
        console.print(f"[cyan]Syncing {source} to {ip}:{destination}[/cyan]")
        
        try:
            # Run rsync
            result = subprocess.run(rsync_cmd, check=True)
            console.print("[green]✓ Sync completed successfully[/green]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error: Sync failed with code {e.returncode}[/red]")
            return False
        except FileNotFoundError:
            console.print("[red]Error: rsync not found. Please install rsync.[/red]")
            return False
            
    def run(self, command: str) -> int:
        """Execute a command on the droplet and stream output."""
        droplet_info = self.get_droplet_info()
        if not droplet_info:
            console.print("[red]Error: No active droplet found[/red]")
            console.print("[yellow]Run 'chisel up' first to create a droplet[/yellow]")
            return 1
            
        ip = droplet_info["ip"]
        
        console.print(f"[cyan]Running on {ip}: {command}[/cyan]")
        
        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # Connect
            ssh.connect(ip, username="root", timeout=10)
            
            # Execute command
            stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
            
            # Get the channel for real-time output
            channel = stdout.channel
            
            # Stream output
            while True:
                # Check if there's data to read
                if channel.recv_ready():
                    data = channel.recv(1024).decode('utf-8', errors='replace')
                    if data:
                        console.print(data, end='')
                        
                if channel.recv_stderr_ready():
                    data = channel.recv_stderr(1024).decode('utf-8', errors='replace')
                    if data:
                        console.print(f"[red]{data}[/red]", end='')
                        
                # Check if command is done
                if channel.exit_status_ready():
                    break
                    
            # Get exit code
            exit_code = channel.recv_exit_status()
            
            # Read any remaining output
            remaining_stdout = stdout.read().decode('utf-8', errors='replace')
            remaining_stderr = stderr.read().decode('utf-8', errors='replace')
            
            if remaining_stdout:
                console.print(remaining_stdout, end='')
            if remaining_stderr:
                console.print(f"[red]{remaining_stderr}[/red]", end='')
                
            if exit_code != 0:
                console.print(f"\n[red]Command exited with code {exit_code}[/red]")
            else:
                console.print("\n[green]✓ Command completed successfully[/green]")
                
            return exit_code
            
        except paramiko.AuthenticationException:
            console.print("[red]Error: SSH authentication failed[/red]")
            return 1
        except paramiko.SSHException as e:
            console.print(f"[red]Error: SSH connection failed: {e}[/red]")
            return 1
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1
        finally:
            ssh.close()