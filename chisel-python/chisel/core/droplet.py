import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import paramiko
from rich.console import Console

from .types.pydo_create_api import PydoDropletObject

console = Console()


class Droplet:
    """
    Represents a DigitalOcean droplet with SSH functionality.
    Initialized from DropletObject dataclass.
    """

    def __init__(
        self, droplet_data: PydoDropletObject, ssh_username: str = "root"
    ):
        self.droplet = droplet_data
        self.id = self.droplet.id
        self.name = self.droplet.name
        self.status = self.droplet.status
        self.region = self.droplet.region.slug
        self.size = self.droplet.size_slug
        self.image = self.droplet.image.slug or str(self.droplet.image.id)
        self.ip = self._extract_public_ip()
        self.ssh_username = ssh_username
        self.gpu_type = self.name

    def sync_file(self, local_path: str, remote_path: str) -> bool:
        if not self.ip:
            console.print("[red]Error: No public IP found for droplet.[/red]")
            return False

        source_path = Path(local_path).resolve()
        if not source_path.exists():
            console.print(
                f"[red]Error: Source path '{local_path}' does not exist[/red]"
            )
            return False
        else:
            source = str(source_path)

        rsync_cmd = [
            "rsync",
            "-avz",  # archive, verbose, compress
            "--progress",
            "-e",
            "ssh -o StrictHostKeyChecking=no",
            source,
            f"{self.ssh_username}@{self.ip}:{remote_path}",
        ]
        console.print(
            f"[cyan]Syncing {source} to {self.ip}:{remote_path}[/cyan]"
        )

        try:
            _ = subprocess.run(rsync_cmd, check=True)
            console.print("[green]✓ Sync completed successfully[/green]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(
                f"[red]Error: Sync failed with code {e.returncode}[/red]"
            )
            return False
        except FileNotFoundError as e:
            console.print(
                f"[red]Error: rsync not found. Please install rsync. {e}[/red]"
            )
            return False

    def _extract_public_ip(self) -> Optional[str]:
        for net in self.droplet.networks.v4:
            if net.type == "public":
                return net.ip_address
        return None

    def get_ssh_client(self, timeout: int = 10) -> paramiko.SSHClient:
        if not self.ip:
            raise ValueError("No public IP found for droplet.")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.ip, username=self.ssh_username, timeout=timeout)
        return ssh

    def run_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        result = {"stdout": "", "stderr": "", "exit_code": None}
        try:
            ssh = self.get_ssh_client(timeout=timeout)
            stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
            result["stdout"] = stdout.read().decode("utf-8", errors="replace")
            result["stderr"] = stderr.read().decode("utf-8", errors="replace")
            result["exit_code"] = stdout.channel.recv_exit_status()
            ssh.close()
        except Exception as e:
            result["stderr"] += f"\n[SSH ERROR] {e}"
        return result

    def run_container_command(
        self, command: str, timeout: int = 30
    ) -> Dict[str, Any]:
        venv_command = f"source /opt/venv/bin/activate && cd /workspace && {command}"  # TODO: change name to run_command
        return self.run_command(venv_command, timeout=timeout)

    def is_ssh_ready(self, timeout: int = 5) -> bool:
        if not self.ip:
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((self.ip, 22))
            sock.close()
            return result == 0
        except Exception:
            return False
