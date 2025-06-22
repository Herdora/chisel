"""SSH and sync operations for chisel."""

import os
import signal
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import paramiko
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .state import State

console = Console()


class InterruptHandler:
    """Handle graceful interrupts for long-running operations."""
    
    def __init__(self):
        self.interrupted = False
        self.old_handler = None
        
    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGINT, self._signal_handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_handler:
            signal.signal(signal.SIGINT, self.old_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signal."""
        self.interrupted = True
        console.print("\n[yellow]Interrupt received. Cleaning up...[/yellow]")
        
    def check_interrupted(self):
        """Check if interrupted and raise KeyboardInterrupt if so."""
        if self.interrupted:
            raise KeyboardInterrupt("Operation interrupted by user")


class SSHManager:
    def __init__(self):
        self.state = State()
        self._ensure_local_ssh_key()
        
    def get_droplet_info(self) -> Optional[Dict[str, Any]]:
        """Get droplet info from state."""
        return self.state.get_droplet_info()
    
    def _ensure_local_ssh_key(self) -> Optional[str]:
        """Ensure local SSH key exists and return path to public key."""
        ssh_key_paths = [
            (os.path.expanduser("~/.ssh/id_ed25519"), os.path.expanduser("~/.ssh/id_ed25519.pub")),
            (os.path.expanduser("~/.ssh/id_rsa"), os.path.expanduser("~/.ssh/id_rsa.pub")),
            (os.path.expanduser("~/.ssh/id_ecdsa"), os.path.expanduser("~/.ssh/id_ecdsa.pub")),
        ]
        
        # Check for existing keys
        for private_path, public_path in ssh_key_paths:
            if os.path.exists(private_path) and os.path.exists(public_path):
                return public_path
        
        # Generate new ED25519 key if none exist
        try:
            private_path = os.path.expanduser("~/.ssh/id_ed25519")
            public_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
            
            # Ensure .ssh directory exists
            os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)
            
            # Generate key
            subprocess.run([
                "ssh-keygen", "-t", "ed25519", "-f", private_path, "-N", "", "-q"
            ], check=True, capture_output=True)
            
            console.print("[green]Generated new SSH key[/green]")
            return public_path
        except Exception:
            return None
    
    def _ensure_ssh_access(self, ip: str) -> bool:
        """Ensure we can SSH to the droplet, adding our key if necessary."""
        # First, try to connect
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username="root", timeout=5)
            ssh.close()
            return True
        except paramiko.AuthenticationException:
            # Authentication failed
            console.print("[red]SSH authentication failed.[/red]")
            
            # Get local SSH public key
            public_key_path = self._ensure_local_ssh_key()
            if not public_key_path or not os.path.exists(public_key_path):
                console.print("[red]No SSH key found. Generating one...[/red]")
                return False
            
            with open(public_key_path, 'r') as f:
                public_key = f.read().strip()
            
            console.print(f"\n[yellow]To enable SSH access from this machine to the droplet:[/yellow]")
            console.print(f"\n[cyan]Option 1: Add your SSH key to DigitalOcean and recreate the droplet:[/cyan]")
            console.print(f"1. Go to: https://amd.digitalocean.com/account/keys")
            console.print(f"2. Click 'Add SSH Key'")
            console.print(f"3. Paste this key and give it a name:")
            console.print(f"\n[white]{public_key}[/white]\n")
            console.print(f"4. Run 'chisel down' then 'chisel up' to recreate the droplet with your key")
            
            console.print(f"\n[cyan]Option 2: Manually add the key to the existing droplet:[/cyan]")
            console.print(f"1. SSH into the droplet from a machine that has access")
            console.print(f"2. Run: echo '{public_key}' >> ~/.ssh/authorized_keys")
            
            return False
        except Exception as e:
            console.print(f"[red]SSH connection error: {e}[/red]")
            return False
    
    def _show_cost_warning(self) -> None:
        """Show cost warning if droplet has been running for a while."""
        should_warn, uptime_hours, estimated_cost = self.state.should_warn_cost()
        
        if should_warn:
            console.print(f"\n[yellow]⚠️  Cost Warning: Droplet has been running for {uptime_hours:.1f} hours[/yellow]")
            console.print(f"[yellow]   Estimated cost: ${estimated_cost:.2f} (at $1.99/hour)[/yellow]")
            console.print(f"[yellow]   Run 'chisel down' to stop billing[/yellow]\n")
        
    def sync(self, source: str, destination: Optional[str] = None) -> bool:
        """Sync files to the droplet using rsync."""
        droplet_info = self.get_droplet_info()
        if not droplet_info:
            console.print("[red]Error: No active droplet found[/red]")
            console.print("[yellow]Run 'chisel up' first to create a droplet[/yellow]")
            return False
        
        # Show cost warning
        self._show_cost_warning()
        
        # Ensure SSH access
        ip = droplet_info["ip"]
        if not self._ensure_ssh_access(ip):
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
        
        # Show cost warning
        self._show_cost_warning()
            
        ip = droplet_info["ip"]
        
        console.print(f"[cyan]Running on {ip}: {command}[/cyan]")
        
        with InterruptHandler() as interrupt_handler:
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
                    # Check for interrupt
                    interrupt_handler.check_interrupted()
                    
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
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Operation interrupted by user[/yellow]")
                # Terminate the remote command if possible
                try:
                    if 'channel' in locals() and not channel.closed:
                        channel.close()
                except:
                    pass
                return 130  # Standard exit code for Ctrl+C
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
    
    def profile(self, command: str, trace: str = "hip,hsa", output_dir: str = "./out", open_result: bool = False) -> Optional[str]:
        """Profile a command with rocprof and pull results locally."""
        droplet_info = self.get_droplet_info()
        if not droplet_info:
            console.print("[red]Error: No active droplet found[/red]")
            console.print("[yellow]Run 'chisel up' first to create a droplet[/yellow]")
            return None
        
        # Show cost warning
        self._show_cost_warning()
            
        ip = droplet_info["ip"]
        
        # Create remote profile directory
        remote_profile_dir = "/tmp/chisel_profile"
        
        # Build rocprof command
        trace_flags = []
        if "hip" in trace:
            trace_flags.append("--hip-trace")
        if "hsa" in trace:
            trace_flags.append("--hsa-trace")
        if "roctx" in trace:
            trace_flags.append("--roctx-trace")
        
        trace_flags.append("--stats")
        
        # Create the profile command  
        profile_cmd = f"""
        rm -rf {remote_profile_dir} && 
        mkdir -p {remote_profile_dir} && 
        cd {remote_profile_dir} && 
        rocprof -d {remote_profile_dir} {' '.join(trace_flags)} -o results.csv {command}
        """
        
        console.print(f"[cyan]Profiling on {ip}: {command}[/cyan]")
        console.print(f"[cyan]Trace options: {trace}[/cyan]")
        
        with InterruptHandler() as interrupt_handler:
            # Execute profiling
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            try:
                # Connect and run profiling
                ssh.connect(ip, username="root", timeout=10)
                
                # Run profiling command
                stdin, stdout, stderr = ssh.exec_command(profile_cmd, get_pty=True)
                
                # Stream output
                channel = stdout.channel
                while True:
                    # Check for interrupt
                    interrupt_handler.check_interrupted()
                    
                    if channel.recv_ready():
                        data = channel.recv(1024).decode('utf-8', errors='replace')
                        if data:
                            console.print(data, end='')
                            
                    if channel.recv_stderr_ready():
                        data = channel.recv_stderr(1024).decode('utf-8', errors='replace')
                        if data:
                            console.print(f"[yellow]{data}[/yellow]", end='')
                            
                    if channel.exit_status_ready():
                        break
                        
                exit_code = channel.recv_exit_status()
                
                if exit_code != 0:
                    console.print(f"\n[red]Profiling failed with exit code {exit_code}[/red]")
                    return None
                
                console.print("\n[green]✓ Profiling completed[/green]")
                
                # Create archive on remote
                archive_cmd = f"cd /tmp && tar -czf chisel_profile.tgz chisel_profile"
                stdin, stdout, stderr = ssh.exec_command(archive_cmd)
                
                archive_exit_code = stdout.channel.recv_exit_status()
                if archive_exit_code != 0:
                    console.print("[red]Error: Failed to create archive[/red]")
                    return None
                
                console.print("[cyan]Pulling results to local machine...[/cyan]")
                
                # Pull archive using scp
                local_output_dir = Path(output_dir)
                local_output_dir.mkdir(parents=True, exist_ok=True)
                
                local_archive_path = local_output_dir / "chisel_profile.tgz"
                
                # Use scp to download
                scp_cmd = [
                    "scp", 
                    "-o", "StrictHostKeyChecking=no",
                    f"root@{ip}:/tmp/chisel_profile.tgz",
                    str(local_archive_path)
                ]
                
                result = subprocess.run(scp_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    console.print(f"[red]Error: Failed to download archive: {result.stderr}[/red]")
                    return None
                
                # Extract archive
                with tarfile.open(local_archive_path, 'r:gz') as tar:
                    tar.extractall(local_output_dir)
                
                # Clean up archive
                local_archive_path.unlink()
                
                # Clean up remote files
                cleanup_cmd = f"rm -rf {remote_profile_dir} /tmp/chisel_profile.tgz"
                ssh.exec_command(cleanup_cmd)
                
                console.print(f"[green]✓ Profile results saved to {local_output_dir / 'chisel_profile'}[/green]")
                
                # Show summary if results files exist (try JSON first, then CSV)
                json_file = local_output_dir / "chisel_profile" / "results.json"
                csv_file = local_output_dir / "chisel_profile" / "results.csv"
                stats_csv_file = local_output_dir / "chisel_profile" / "results.stats.csv"
                
                if json_file.exists():
                    self._show_profile_summary(json_file)
                elif csv_file.exists():
                    self._show_profile_summary(csv_file)
                elif stats_csv_file.exists():
                    self._show_profile_summary(stats_csv_file)
                
                return str(local_output_dir / "chisel_profile")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Profiling interrupted by user[/yellow]")
                # Clean up remote files
                try:
                    cleanup_cmd = f"rm -rf {remote_profile_dir} /tmp/chisel_profile.tgz"
                    ssh.exec_command(cleanup_cmd)
                except:
                    pass
                return None
            except Exception as e:
                console.print(f"[red]Error during profiling: {e}[/red]")
                return None
            finally:
                ssh.close()
    
    def pull(self, remote_path: str, local_path: Optional[str] = None) -> bool:
        """Pull files or directories from the droplet to local machine."""
        droplet_info = self.get_droplet_info()
        if not droplet_info:
            console.print("[red]Error: No active droplet found[/red]")
            console.print("[yellow]Run 'chisel up' first to create a droplet[/yellow]")
            return False
        
        # Show cost warning
        self._show_cost_warning()
            
        ip = droplet_info["ip"]
        
        # Default local path is current directory with remote filename
        if local_path is None:
            remote_basename = os.path.basename(remote_path.rstrip('/'))
            if not remote_basename:
                remote_basename = "pulled_files"
            local_path = f"./{remote_basename}"
        
        # Resolve local path
        local_path_obj = Path(local_path).resolve()
        
        console.print(f"[cyan]Pulling {remote_path} from {ip} to {local_path}[/cyan]")
        
        # First check if remote path exists and get info
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            ssh.connect(ip, username="root", timeout=10)
            
            # Check if remote path exists and if it's a file or directory
            stdin, stdout, stderr = ssh.exec_command(f"test -e '{remote_path}' && echo 'exists' || echo 'missing'")
            exists_result = stdout.read().decode().strip()
            
            if exists_result == 'missing':
                console.print(f"[red]Error: Remote path '{remote_path}' does not exist[/red]")
                return False
            
            # Check if it's a directory
            stdin, stdout, stderr = ssh.exec_command(f"test -d '{remote_path}' && echo 'dir' || echo 'file'")
            path_type = stdout.read().decode().strip()
            
            ssh.close()
            
            # Use scp for file transfer
            if path_type == "dir":
                # For directories, use scp -r
                scp_cmd = [
                    "scp", 
                    "-r",
                    "-o", "StrictHostKeyChecking=no",
                    f"root@{ip}:{remote_path}",
                    str(local_path_obj.parent)  # scp -r will create the directory
                ]
            else:
                # For files, create parent directory if needed
                local_path_obj.parent.mkdir(parents=True, exist_ok=True)
                scp_cmd = [
                    "scp",
                    "-o", "StrictHostKeyChecking=no", 
                    f"root@{ip}:{remote_path}",
                    str(local_path_obj)
                ]
            
            # Execute scp
            result = subprocess.run(scp_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"[red]Error: Failed to pull files: {result.stderr}[/red]")
                return False
            
            console.print(f"[green]✓ Successfully pulled to {local_path}[/green]")
            return True
            
        except paramiko.AuthenticationException:
            console.print("[red]Error: SSH authentication failed[/red]")
            return False
        except paramiko.SSHException as e:
            console.print(f"[red]Error: SSH connection failed: {e}[/red]")
            return False
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error: SCP failed with code {e.returncode}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return False
        finally:
            if 'ssh' in locals():
                ssh.close()
    
    def _show_profile_summary(self, stats_file: Path) -> None:
        """Show a summary of the profiling results."""
        try:
            import json
            
            console.print("\n[cyan]Top GPU Kernels by Total Time:[/cyan]")
            
            # Try to parse as JSON trace format
            if stats_file.suffix == '.json' or stats_file.name == 'results.json':
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                
                kernels = []
                for event in data.get('traceEvents', []):
                    if (event.get('ph') == 'X' and 
                        'pid' in event and 
                        event.get('pid') in [6, 7] and  # GPU pids
                        'DurationNs' in event.get('args', {})):
                        
                        kernel_name = event.get('name', '')
                        duration_ns = int(event['args']['DurationNs'])
                        
                        kernels.append({
                            'name': kernel_name,
                            'total_time': duration_ns / 1_000_000,  # Convert to ms
                            'duration_ns': duration_ns
                        })
                
                # Sort by total time
                kernels.sort(key=lambda x: x['total_time'], reverse=True)
                
                # Show kernels
                for i, kernel in enumerate(kernels):
                    console.print(f"  {i+1:2d}. {kernel['name'][:60]:<60} {kernel['total_time']:8.3f} ms")
                
                # Also show top HIP API calls
                hip_calls = []
                for event in data.get('traceEvents', []):
                    if (event.get('ph') == 'X' and 
                        event.get('pid') == 2 and  # CPU HIP API pid
                        'DurationNs' in event.get('args', {})):
                        
                        api_name = event.get('name', '')
                        duration_ns = int(event['args']['DurationNs'])
                        
                        hip_calls.append({
                            'name': api_name,
                            'total_time': duration_ns / 1_000_000,  # Convert to ms
                            'duration_ns': duration_ns
                        })
                
                # Sort by total time  
                hip_calls.sort(key=lambda x: x['total_time'], reverse=True)
                
                if hip_calls:
                    console.print("\n[cyan]Top HIP API Calls by Total Time:[/cyan]")
                    for i, call in enumerate(hip_calls[:5]):
                        console.print(f"  {i+1:2d}. {call['name'][:60]:<60} {call['total_time']:8.3f} ms")
                
            else:
                # Try CSV format
                import csv
                kernels = []
                with open(stats_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'KernelName' in row and 'TotalDurationNs' in row:
                            kernels.append({
                                'name': row['KernelName'],
                                'total_time': float(row['TotalDurationNs']) / 1_000_000,  # Convert to ms
                                'calls': int(row.get('Calls', 0))
                            })
                
                # Sort by total time
                kernels.sort(key=lambda x: x['total_time'], reverse=True)
                
                # Show top 10
                for i, kernel in enumerate(kernels[:10]):
                    console.print(f"  {i+1:2d}. {kernel['name'][:60]:<60} {kernel['total_time']:8.2f} ms ({kernel['calls']} calls)")
                
                if len(kernels) > 10:
                    console.print(f"  ... and {len(kernels) - 10} more kernels")
                
        except Exception as e:
            console.print(f"[yellow]Could not parse profile summary: {e}[/yellow]")