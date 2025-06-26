import socket
import subprocess
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, cast

import pydo
import paramiko
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .droplet import Droplet
from .types.gpu_profiles import GPU_PROFILES, GPUType

console = Console()

DIGITAL_OCEAN_METADATA_URL = "http://169.254.169.254/metadata/v1/id"


class DropletService:
    """Manages all droplets for a given DigitalOcean account."""

    def __init__(self, token: str):
        self.pydo_client = pydo.Client(token=token)

    def create_droplet(
        self,
        gpu_type: GPUType,
        droplet_region: str,
        droplet_size: str,
        droplet_image: str,
        replace: bool = False,
    ) -> Droplet:
        # (unchanged) tear down duplicates when replace=True …

        ssh_keys = self.list_account_ssh_keys()

        # ----------------------------------------------------------------- #
        # new cloud-init script: installs Docker, pulls your image, starts it,
        # and drops every future SSH login straight into the container.
        # ----------------------------------------------------------------- #
        user_data = """#!/bin/bash
        set -e

        # Install Docker
        apt-get update
        apt-get install -y docker.io

        # Start and enable Docker
        systemctl enable docker
        systemctl start docker
        usermod -aG docker root

        # Setup ROCm (optional redundancy)
        echo 'export PATH=/opt/rocm/bin:$PATH' >> /etc/profile.d/rocm.sh
        echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> /etc/profile.d/rocm.sh
        echo 'export HIP_PATH=/opt/rocm' >> /etc/profile.d/rocm.sh
        ln -sf /opt/rocm/bin/hipcc /usr/local/bin/hipcc

        # Pull official AMD ROCm PyTorch image
        docker pull rocm/pytorch:rocm6.4.0_ubuntu22.04_py3.10_pytorch_2.6.0

        # Run persistent container (named 'ml')
        docker run -dit \
        --name ml \
        --restart=always \
        --network host \
        --ipc=host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v /mnt/share:/workspace \
        rocm/pytorch:rocm6.4.0_ubuntu22.04_py3.10_pytorch_2.6.0 bash

        # Optional: install Python packages into container globally
        docker exec ml pip install rich pydo paramiko jupyterlab vllm triton sglang

        # Automatically enter the container on SSH login
        echo 'exec docker exec -it ml bash' >> /root/.bash_profile

        echo "Setup complete"
        """

        body = {
            "name": gpu_type.value,
            "region": droplet_region,
            "size": droplet_size,
            "image": droplet_image,
            "ssh_keys": ssh_keys,
            "user_data": user_data,
            "tags": [gpu_type.value],
        }

        response = self.pydo_client.droplets.create(body=body)
        return Droplet(response["droplet"])

    def get_droplet_by_id(self, droplet_id: int) -> Optional[Droplet]:
        """Get a droplet by ID."""
        try:
            response = self.pydo_client.droplets.get(droplet_id)
            return Droplet(response["droplet"])
        except Exception as e:
            console.print(f"[red]Error fetching droplet: {e}[/red]")
            return None

    def get_droplet_by_type(self, droplet_name: str) -> Optional[Droplet]:
        """Get a droplet by name."""
        try:
            response = self.pydo_client.droplets.list()
            droplets = response.get("droplets", [])
            for droplet in droplets:
                if droplet["name"] == droplet_name:
                    return Droplet(droplet)
            return None
        except Exception:
            return None

    def get_or_create_droplet_by_type(self, gpu_type: GPUType) -> Droplet:
        """Create or reuse a droplet."""
        existing = self.get_droplet_by_type(gpu_type.value)

        if existing:
            console.print(f"[green]Found existing droplet: {existing.name}[/green]")
            return existing

        gpu_profile_for_gpu_type = GPU_PROFILES[gpu_type]
        console.print(f"[yellow]Creating new {gpu_type} droplet...[/yellow]")
        droplet = self.create_droplet(
            gpu_type,
            gpu_profile_for_gpu_type.region.value,
            gpu_profile_for_gpu_type.size,
            gpu_profile_for_gpu_type.image,
        )

        if droplet.id is None:
            raise ValueError("Droplet ID is None after creation.")
        droplet = self.wait_for_droplet(droplet.id)
        if droplet.ip is None:
            raise ValueError("Droplet IP is None after creation.")
        console.print(
            f"[yellow]Warning: SSH may not be fully ready yet for {gpu_type}[/yellow]"
            if not self.wait_for_ssh(droplet.ip)
            else f"[green]{gpu_type} droplet ready![/green]"
        )

        return droplet

    def destroy_droplet(self, droplet_id: int) -> None:
        """Destroy a droplet."""
        self.pydo_client.droplets.destroy(droplet_id)

    def delete_all_droplets_by_type(self, gpu_type: GPUType) -> None:
        """Delete all droplets by type."""
        droplets = self.list_droplets()
        for droplet in droplets:
            if droplet.name == gpu_type.value:
                if droplet.id is not None:
                    self.destroy_droplet(droplet.id)

    def list_account_ssh_keys(self) -> List[int]:
        """List all SSH key IDs in the DO account."""
        try:
            response = self.pydo_client.ssh_keys.list()
            return [key["id"] for key in response.get("ssh_keys", [])]
        except Exception as e:
            console.print(f"[red]Error fetching SSH keys: {e}[/red]")
            return []

    # TODO: how much do we need to do this
    def add_local_ssh_key_to_do(self) -> Optional[int]:
        """If on a DO droplet and a local SSH key exists, add it to DO."""
        try:
            import requests
            import os
            import time

            if requests.get(DIGITAL_OCEAN_METADATA_URL, timeout=1).ok:
                ssh_pub_paths = [
                    os.path.expanduser("~/.ssh/id_rsa.pub"),
                    os.path.expanduser("~/.ssh/id_ed25519.pub"),
                    os.path.expanduser("~/.ssh/id_ecdsa.pub"),
                ]

                for pub_path in ssh_pub_paths:
                    if os.path.exists(pub_path):
                        with open(pub_path, "r") as f:
                            pub_key_content = f.read().strip()

                        # Try to add the key
                        key_name = f"chisel-temp-{os.path.basename(pub_path)}-{int(time.time())}"
                        try:
                            key_response = self.pydo_client.ssh_keys.create(
                                body={"name": key_name, "public_key": pub_key_content}
                            )
                            return key_response["ssh_key"]["id"]
                        except Exception as e:
                            if "already in use" in str(e):
                                # Try to find existing key
                                for key in self.pydo_client.ssh_keys.list().get("ssh_keys", []):
                                    if key.get("public_key", "").strip() == pub_key_content:
                                        return key["id"]
            return None
        except Exception as e:
            console.print(f"[yellow]Failed to import local SSH key: {e}[/yellow]")
            return None

    def list_droplets(self) -> List[Droplet]:
        """List all droplets."""
        try:
            response = self.pydo_client.droplets.list()
            droplets = response.get("droplets", [])
            return [Droplet(d) for d in droplets]
        except Exception as e:
            console.print(f"[red]Error listing droplets: {e}[/red]")
            return []

    def wait_for_droplet(self, droplet_id: int, timeout: int = 300) -> Droplet:
        """Wait for droplet to be active."""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Activating droplet...", total=None)

            while time.time() - start_time < timeout:
                response = self.pydo_client.droplets.get(droplet_id)
                droplet = response["droplet"]

                if droplet["status"] == "active":
                    return Droplet(droplet)

                time.sleep(5)

        raise TimeoutError("Droplet failed to become active within timeout")

    def wait_for_ssh(self, ip: str, timeout: int = 300) -> bool:
        """Wait for SSH to be available."""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Waiting for SSH to be ready...", total=None)

            while time.time() - start_time < timeout:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((ip, 22))
                    sock.close()

                    if result == 0:
                        # Try actual SSH connection
                        ssh = paramiko.SSHClient()
                        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                        try:
                            ssh.connect(ip, username="root", timeout=5)
                            ssh.close()
                            return True
                        except Exception:
                            pass

                except Exception:
                    pass

                time.sleep(5)

        return False

    def validate_token(self) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate the API token and return account info.

        Returns:
            (success, account_info) - success is True if token is valid
        """
        try:
            account = self.pydo_client.account.get()
            return True, cast(Dict[str, Any], account)
        except Exception:
            return False, None

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        try:
            return cast(Dict[str, Any], self.pydo_client.account.get())
        except Exception as e:
            console.print(f"[red]Error fetching account info: {e}[/red]")
            return None

    def get_balance(self) -> Optional[Dict[str, Any]]:
        """Get account balance."""
        try:
            return cast(Dict[str, Any], self.pydo_client.balance.get())
        except Exception as e:
            console.print(f"[red]Error fetching balance: {e}[/red]")
            return None
