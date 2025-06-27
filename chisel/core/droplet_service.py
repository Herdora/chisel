import socket
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
        # Cloud-init script: installs ROCm, PyTorch, and development tools
        # directly on the host system with automatic virtual environment activation.
        # ----------------------------------------------------------------- #
        user_data = """#!/bin/bash
set -e

### System prep -----------------------------------------------------------
apt-get update
apt-get install -y python3-venv build-essential cmake git python3-pip wget curl gnupg software-properties-common

### Create working directory ---------------------------------------------- 
mkdir -p /workspace
chmod 755 /workspace

### Install ROCm (for AMD GPUs) -------------------------------------------
# Add ROCm repository
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.4.1 jammy main' > /etc/apt/sources.list.d/rocm.list
apt-get update

# Install ROCm packages
apt-get install -y rocm-dev rocm-libs rocprofiler-dev roctracer-dev rocm-profiler || echo 'Some ROCm packages may not be available for this GPU type'

# Add user to video and render groups
usermod -aG video root
usermod -aG render root

### Install NVIDIA CUDA (for NVIDIA GPUs) ---------------------------------
# Add NVIDIA repository
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

# Install CUDA toolkit and profiling tools
apt-get install -y cuda-toolkit-12-6 || echo 'CUDA packages not available for this GPU type'

### Create Python virtual environment -------------------------------------
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# Upgrade pip and install base packages
pip install --upgrade pip setuptools wheel

# Install PyTorch with ROCm support (for AMD)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1 || echo 'ROCm PyTorch not available, trying CUDA version'

# Install PyTorch with CUDA support (for NVIDIA) - fallback if ROCm fails
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || echo 'CUDA PyTorch not available, using CPU version'

# Install additional ML and profiling packages
pip install rich pydo paramiko jupyterlab transformers accelerate || echo 'Some Python packages may have failed to install'

### Set up environment variables ------------------------------------------
# ROCm environment
echo 'export ROCM_PATH=/opt/rocm' >> /root/.bashrc
echo 'export PATH=$PATH:/opt/rocm/bin' >> /root/.bashrc
echo 'export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/' >> /root/.bashrc

# CUDA environment  
echo 'export CUDA_HOME=/usr/local/cuda' >> /root/.bashrc
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> /root/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> /root/.bashrc

# Automatically activate virtual environment on login
echo 'source /opt/venv/bin/activate' >> /root/.bashrc
echo 'cd /workspace' >> /root/.bashrc

# Set up bash_profile to ensure environment is loaded
echo 'source ~/.bashrc' >> /root/.bash_profile

echo "Setup complete - ROCm/CUDA and PyTorch installed with virtual environment"
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
        console.print(f"[cyan]Ensuring {gpu_type.value} droplet is ready...[/cyan]")
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
