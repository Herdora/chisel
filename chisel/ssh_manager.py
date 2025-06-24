"""SSH and sync operations for chisel."""

import os
import signal
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

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

    def get_droplet_info(self, gpu_type: str) -> Optional[Dict[str, Any]]:
        """Get droplet info from state for specific GPU type."""
        return self.state.get_droplet_info(gpu_type)

    def _ensure_local_ssh_key(self) -> Optional[str]:
        """Ensure local SSH key exists and return path to public key."""
        ssh_key_paths = [
            (
                os.path.expanduser("~/.ssh/id_ed25519"),
                os.path.expanduser("~/.ssh/id_ed25519.pub"),
            ),
            (
                os.path.expanduser("~/.ssh/id_rsa"),
                os.path.expanduser("~/.ssh/id_rsa.pub"),
            ),
            (
                os.path.expanduser("~/.ssh/id_ecdsa"),
                os.path.expanduser("~/.ssh/id_ecdsa.pub"),
            ),
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
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", private_path, "-N", "", "-q"],
                check=True,
                capture_output=True,
            )

            console.print("[green]Generated new SSH key[/green]")
            return public_path
        except Exception:
            return None

    def _ensure_ssh_access(self, ip: str) -> bool:
        """Ensure we can SSH to the droplet, adding our key if necessary."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Verifying SSH connection...", total=None)
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(ip, username="root", timeout=10)
                ssh.close()
            return True
        except paramiko.AuthenticationException:
            console.print("\n[bold red]SSH Key Not Authorized[/bold red]")
            console.print(
                f"The droplet at [cyan]{ip}[/cyan] did not accept your SSH key."
            )

            public_key_path = self._ensure_local_ssh_key()
            if not public_key_path or not os.path.exists(public_key_path):
                console.print("[red]Could not find or generate a local SSH key.[/red]")
                return False

            with open(public_key_path, "r") as f:
                public_key = f.read().strip()

            console.print(
                "\n[yellow]To fix this, you need to add your public SSH key to the droplet.[/yellow]"
            )
            console.print(
                "\n[bold cyan]Option 1: (Recommended) Add key to DigitalOcean and recreate[/bold cyan]"
            )
            console.print(
                "1. Go to your DigitalOcean account settings: [link=https://cloud.digitalocean.com/account/security]https://cloud.digitalocean.com/account/security[/link]"
            )
            console.print("2. Click 'Add SSH Key'.")
            console.print("3. Paste the following key:")
            console.print(f"\n[white on black] {public_key} [/white on black]\n")
            console.print("4. Once added, destroy and recreate the droplet with:")
            console.print("   [bold]chisel down && chisel up[/bold]")

            console.print(
                "\n[bold cyan]Option 2: Manually add the key to the running droplet[/bold cyan]"
            )
            console.print(
                "1. Log in to the droplet using a different machine or the web console."
            )
            console.print("2. Run this command to add your key:")
            console.print(
                f"   [bold]echo '{public_key}' >> ~/.ssh/authorized_keys[/bold]"
            )

            return False
        except (paramiko.SSHException, TimeoutError) as e:
            console.print("\n[bold red]SSH Connection Failed[/bold red]")
            console.print(f"Could not connect to the droplet at [cyan]{ip}[/cyan].")
            console.print(f"Reason: {e}")
            console.print("\n[yellow]Please check the following:[/yellow]")
            console.print(
                "1. Is the droplet running? Check the DigitalOcean dashboard."
            )
            console.print("2. Is your internet connection working?")
            console.print("3. Are there any firewalls blocking port 22?")
            return False
        except Exception as e:
            console.print(
                f"\n[bold red]An unexpected SSH error occurred: {e}[/bold red]"
            )
            return False

    def _show_cost_warning(self, gpu_type: str) -> None:
        """Show cost warning if droplet has been running for a while."""
        # Use appropriate hourly rate based on GPU type
        hourly_rate = 4.89 if "nvidia" in gpu_type else 1.99
        should_warn, uptime_hours, estimated_cost = self.state.should_warn_cost(
            gpu_type, hourly_rate=hourly_rate
        )

        if should_warn:
            console.print(
                f"\n[yellow]⚠️  Cost Warning: {gpu_type} droplet has been running for {uptime_hours:.1f} hours[/yellow]"
            )
            console.print(
                f"[yellow]   Estimated cost: ${estimated_cost:.2f} (at ${hourly_rate}/hour)[/yellow]"
            )
            console.print(
                f"[yellow]   Run 'chisel down --gpu-type {gpu_type}' to stop billing[/yellow]\n"
            )

    def sync(
        self, source: str, destination: Optional[str] = None, gpu_type: str = None
    ) -> bool:
        """Sync files to the droplet using rsync."""
        if not gpu_type:
            console.print("[red]Error: GPU type is required[/red]")
            return False

        droplet_info = self.get_droplet_info(gpu_type)
        if not droplet_info:
            console.print(f"[red]Error: No {gpu_type} droplet found[/red]")
            console.print(
                f"[yellow]Run 'chisel up --gpu-type {gpu_type}' first to create a droplet[/yellow]"
            )
            return False

        # Show cost warning
        self._show_cost_warning(gpu_type)

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
        if source_path.is_dir() and not source.endswith("/"):
            source = str(source_path) + "/"
        else:
            source = str(source_path)

        rsync_cmd = [
            "rsync",
            "-avz",  # archive, verbose, compress
            "--progress",
            "-e",
            "ssh -o StrictHostKeyChecking=no",
            source,
            f"root@{ip}:{destination}",
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

    def run(self, command: str, gpu_type: str = None) -> int:
        """Execute a command on the droplet and stream output."""
        if not gpu_type:
            console.print("[red]Error: GPU type is required[/red]")
            return 1

        droplet_info = self.get_droplet_info(gpu_type)
        if not droplet_info:
            console.print(f"[red]Error: No {gpu_type} droplet found[/red]")
            console.print(
                f"[yellow]Run 'chisel up --gpu-type {gpu_type}' first to create a droplet[/yellow]"
            )
            return 1

        # Show cost warning
        self._show_cost_warning(gpu_type)

        ip = droplet_info["ip"]

        # Ensure SSH access before running command
        if not self._ensure_ssh_access(ip):
            return 1

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
                        data = channel.recv(1024).decode("utf-8", errors="replace")
                        if data:
                            console.print(data, end="")

                    if channel.recv_stderr_ready():
                        data = channel.recv_stderr(1024).decode(
                            "utf-8", errors="replace"
                        )
                        if data:
                            console.print(f"[red]{data}[/red]", end="")

                    # Check if command is done
                    if channel.exit_status_ready():
                        break

                # Get exit code
                exit_code = channel.recv_exit_status()

                # Read any remaining output
                remaining_stdout = stdout.read().decode("utf-8", errors="replace")
                remaining_stderr = stderr.read().decode("utf-8", errors="replace")

                if remaining_stdout:
                    console.print(remaining_stdout, end="")
                if remaining_stderr:
                    console.print(f"[red]{remaining_stderr}[/red]", end="")

                if exit_code != 0:
                    console.print(f"\n[red]Command exited with code {exit_code}[/red]")
                else:
                    console.print("\n[green]✓ Command completed successfully[/green]")

                return exit_code

            except KeyboardInterrupt:
                console.print("\n[yellow]Operation interrupted by user[/yellow]")
                # Terminate the remote command if possible
                try:
                    if "channel" in locals() and not channel.closed:
                        channel.close()
                except:
                    pass
                return 130  # Standard exit code for Ctrl+C
            except paramiko.AuthenticationException:
                console.print(
                    "[bold red]Error: SSH authentication failed unexpectedly.[/bold red]"
                )
                console.print(
                    "This can happen if the droplet's SSH keys were changed after the connection was established."
                )
                return 1
            except paramiko.SSHException as e:
                console.print(
                    f"[bold red]Error: SSH connection lost during command execution: {e}[/bold red]"
                )
                return 1
            except Exception as e:
                console.print(f"[red]An unexpected error occurred: {e}[/red]")
                return 1
            finally:
                ssh.close()

    def profile(
        self,
        command: str,
        gpu_type: str = None,
        trace: str = "hip,hsa",
        output_dir: str = "./out",
        open_result: bool = False,
    ) -> Optional[str]:
        """Profile a command with rocprof and pull results locally."""
        if not gpu_type:
            console.print("[red]Error: GPU type is required[/red]")
            return None

        droplet_info = self.get_droplet_info(gpu_type)
        if not droplet_info:
            console.print(f"[red]Error: No {gpu_type} droplet found[/red]")
            console.print(
                f"[yellow]Run 'chisel up --gpu-type {gpu_type}' first to create a droplet[/yellow]"
            )
            return None

        # Show cost warning
        self._show_cost_warning(gpu_type)

        ip = droplet_info["ip"]

        # Ensure SSH access before profiling
        if not self._ensure_ssh_access(ip):
            return None

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
        rocprof -d {remote_profile_dir} {" ".join(trace_flags)} -o results.csv {command}
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
                        data = channel.recv(1024).decode("utf-8", errors="replace")
                        if data:
                            console.print(data, end="")

                    if channel.recv_stderr_ready():
                        data = channel.recv_stderr(1024).decode(
                            "utf-8", errors="replace"
                        )
                        if data:
                            console.print(f"[yellow]{data}[/yellow]", end="")

                    if channel.exit_status_ready():
                        break

                exit_code = channel.recv_exit_status()

                if exit_code != 0:
                    console.print(
                        f"\n[red]Profiling failed with exit code {exit_code}[/red]"
                    )
                    return None

                console.print("\n[green]✓ Profiling completed[/green]")

                # Create archive on remote
                archive_cmd = "cd /tmp && tar -czf chisel_profile.tgz chisel_profile"
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
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"root@{ip}:/tmp/chisel_profile.tgz",
                    str(local_archive_path),
                ]

                result = subprocess.run(scp_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    console.print(
                        f"[red]Error: Failed to download archive: {result.stderr}[/red]"
                    )
                    return None

                # Extract archive
                with tarfile.open(local_archive_path, "r:gz") as tar:
                    tar.extractall(local_output_dir)

                # Clean up archive
                local_archive_path.unlink()

                # Clean up remote files
                cleanup_cmd = f"rm -rf {remote_profile_dir} /tmp/chisel_profile.tgz"
                ssh.exec_command(cleanup_cmd)

                console.print(
                    f"[green]✓ Profile results saved to {local_output_dir / 'chisel_profile'}[/green]"
                )

                # Show summary if results files exist (try JSON first, then CSV)
                json_file = local_output_dir / "chisel_profile" / "results.json"
                csv_file = local_output_dir / "chisel_profile" / "results.csv"
                stats_csv_file = (
                    local_output_dir / "chisel_profile" / "results.stats.csv"
                )

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

    def pull(
        self, remote_path: str, local_path: Optional[str] = None, gpu_type: str = None
    ) -> bool:
        """Pull files or directories from the droplet to local machine."""
        if not gpu_type:
            console.print("[red]Error: GPU type is required[/red]")
            return False

        droplet_info = self.get_droplet_info(gpu_type)
        if not droplet_info:
            console.print(f"[red]Error: No {gpu_type} droplet found[/red]")
            console.print(
                f"[yellow]Run 'chisel up --gpu-type {gpu_type}' first to create a droplet[/yellow]"
            )
            return False

        # Show cost warning
        self._show_cost_warning(gpu_type)

        ip = droplet_info["ip"]

        # Ensure SSH access before pulling
        if not self._ensure_ssh_access(ip):
            return False

        # Default local path is current directory with remote filename
        if local_path is None:
            remote_basename = os.path.basename(remote_path.rstrip("/"))
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
            stdin, stdout, stderr = ssh.exec_command(
                f"test -e '{remote_path}' && echo 'exists' || echo 'missing'"
            )
            exists_result = stdout.read().decode().strip()

            if exists_result == "missing":
                console.print(
                    f"[red]Error: Remote path '{remote_path}' does not exist[/red]"
                )
                return False

            # Check if it's a directory
            stdin, stdout, stderr = ssh.exec_command(
                f"test -d '{remote_path}' && echo 'dir' || echo 'file'"
            )
            path_type = stdout.read().decode().strip()

            ssh.close()

            # Use scp for file transfer
            if path_type == "dir":
                # For directories, use scp -r
                scp_cmd = [
                    "scp",
                    "-r",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"root@{ip}:{remote_path}",
                    # scp -r will create the directory
                    str(local_path_obj.parent),
                ]
            else:
                # For files, create parent directory if needed
                local_path_obj.parent.mkdir(parents=True, exist_ok=True)
                scp_cmd = [
                    "scp",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"root@{ip}:{remote_path}",
                    str(local_path_obj),
                ]

            # Execute scp
            result = subprocess.run(scp_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                console.print(
                    f"[red]Error: Failed to pull files: {result.stderr}[/red]"
                )
                return False

            console.print(f"[green]✓ Successfully pulled to {local_path}[/green]")
            return True

        except paramiko.AuthenticationException:
            console.print(
                "[bold red]Error: SSH authentication failed unexpectedly.[/bold red]"
            )
            return False
        except paramiko.SSHException as e:
            console.print(
                f"[bold red]Error: SSH connection failed while checking remote path: {e}[/bold red]"
            )
            return False
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error: SCP failed with code {e.returncode}[/red]")
            if e.stderr:
                console.print(f"[red]Stderr: {e.stderr}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]An unexpected error occurred while pulling: {e}[/red]")
            return False
        finally:
            if "ssh" in locals():
                ssh.close()

    def _show_profile_summary(self, stats_file: Path) -> None:
        """Show a summary of the profiling results."""
        try:
            import json

            console.print("\n[cyan]Top GPU Kernels by Total Time:[/cyan]")

            # Try to parse as JSON trace format
            if stats_file.suffix == ".json" or stats_file.name == "results.json":
                with open(stats_file, "r") as f:
                    data = json.load(f)

                kernels = []
                for event in data.get("traceEvents", []):
                    if (
                        event.get("ph") == "X"
                        and "pid" in event
                        and event.get("pid") in [6, 7]  # GPU pids
                        and "DurationNs" in event.get("args", {})
                    ):
                        kernel_name = event.get("name", "")
                        duration_ns = int(event["args"]["DurationNs"])

                        kernels.append(
                            {
                                "name": kernel_name,
                                "total_time": duration_ns / 1_000_000,  # Convert to ms
                                "duration_ns": duration_ns,
                            }
                        )

                # Sort by total time
                kernels.sort(key=lambda x: x["total_time"], reverse=True)

                # Show kernels
                for i, kernel in enumerate(kernels):
                    console.print(
                        f"  {i + 1:2d}. {kernel['name'][:60]:<60} {kernel['total_time']:8.3f} ms"
                    )

                # Also show top HIP API calls
                hip_calls = []
                for event in data.get("traceEvents", []):
                    if (
                        event.get("ph") == "X"
                        and event.get("pid") == 2  # CPU HIP API pid
                        and "DurationNs" in event.get("args", {})
                    ):
                        api_name = event.get("name", "")
                        duration_ns = int(event["args"]["DurationNs"])

                        hip_calls.append(
                            {
                                "name": api_name,
                                "total_time": duration_ns / 1_000_000,  # Convert to ms
                                "duration_ns": duration_ns,
                            }
                        )

                # Sort by total time
                hip_calls.sort(key=lambda x: x["total_time"], reverse=True)

                if hip_calls:
                    console.print("\n[cyan]Top HIP API Calls by Total Time:[/cyan]")
                    for i, call in enumerate(hip_calls[:5]):
                        console.print(
                            f"  {i + 1:2d}. {call['name'][:60]:<60} {call['total_time']:8.3f} ms"
                        )

            else:
                # Try CSV format
                import csv

                kernels = []
                with open(stats_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "KernelName" in row and "TotalDurationNs" in row:
                            kernels.append(
                                {
                                    "name": row["KernelName"],
                                    # Convert to ms
                                    "total_time": float(row["TotalDurationNs"])
                                    / 1_000_000,
                                    "calls": int(row.get("Calls", 0)),
                                }
                            )

                # Sort by total time
                kernels.sort(key=lambda x: x["total_time"], reverse=True)

                # Show top 10
                for i, kernel in enumerate(kernels[:10]):
                    console.print(
                        f"  {i + 1:2d}. {kernel['name'][:60]:<60} {kernel['total_time']:8.2f} ms ({kernel['calls']} calls)"
                    )

                if len(kernels) > 10:
                    console.print(f"  ... and {len(kernels) - 10} more kernels")

        except Exception as e:
            console.print(f"[yellow]Could not parse profile summary: {e}[/yellow]")
