"""DigitalOcean droplet management for chisel."""

import socket
import time
from typing import Any, Dict, List, Optional

import paramiko
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .do_client import DOClient
from .state import State

console = Console()


class DropletManager:
    def __init__(self, client: DOClient):
        self.client = client
        self.state = State()
        self.droplet_name = "chisel-dev"
        self.droplet_size = "gpu-mi300x1-192gb"
        self.droplet_image = "gpu-amd-base"
        self.droplet_region = (
            "atl1"  # TODO: check where AMD droplets available and default to that
        )

    def get_ssh_keys(self) -> List[int]:
        """Get all SSH key IDs from DO account, including current droplet's key if on DO."""
        ssh_key_ids = []
        
        try:
            # Get existing keys from DO account
            response = self.client.client.ssh_keys.list()
            keys = response.get("ssh_keys", [])
            ssh_key_ids = [key["id"] for key in keys]
        except Exception:
            pass
        
        # Check if we're running on a DigitalOcean droplet and add local SSH key
        try:
            import requests
            # Check DO metadata service
            metadata_response = requests.get(
                "http://169.254.169.254/metadata/v1/id",
                timeout=1
            )
            if metadata_response.status_code == 200:
                # We're on a DO droplet - check for local SSH keys
                import os
                ssh_pub_paths = [
                    os.path.expanduser("~/.ssh/id_rsa.pub"),
                    os.path.expanduser("~/.ssh/id_ed25519.pub"),
                    os.path.expanduser("~/.ssh/id_ecdsa.pub"),
                ]
                
                for pub_path in ssh_pub_paths:
                    if os.path.exists(pub_path):
                        with open(pub_path, 'r') as f:
                            pub_key_content = f.read().strip()
                        
                        # Add this key to DO account temporarily
                        try:
                            key_name = f"chisel-temp-{os.path.basename(pub_path)}-{int(time.time())}"
                            key_response = self.client.client.ssh_keys.create(
                                body={
                                    "name": key_name,
                                    "public_key": pub_key_content
                                }
                            )
                            if key_response and "ssh_key" in key_response:
                                ssh_key_ids.append(key_response["ssh_key"]["id"])
                                console.print(f"[green]Added local SSH key to droplet[/green]")
                        except Exception as e:
                            # Key might already exist, try to find it
                            if "already in use" in str(e):
                                # Find the existing key
                                for key in keys:
                                    if key.get("public_key", "").strip() == pub_key_content:
                                        if key["id"] not in ssh_key_ids:
                                            ssh_key_ids.append(key["id"])
                                        break
                        break  # Only add first found key
        except Exception:
            # Not on DO droplet or can't access metadata, continue normally
            pass
        
        return ssh_key_ids

    def find_existing_droplet(self) -> Optional[Dict[str, Any]]:
        """Find existing chisel-dev droplet."""
        try:
            response = self.client.client.droplets.list()
            droplets = response.get("droplets", [])

            for droplet in droplets:
                if droplet["name"] == self.droplet_name:
                    return droplet
            return None
        except Exception:
            return None

    def create_droplet(self) -> Dict[str, Any]:
        """Create a new droplet."""
        ssh_keys = self.get_ssh_keys()

        user_data = """#!/bin/bash
# Update package list
apt-get update

# Install basic development tools  
apt-get install -y build-essential git wget curl

# ROCm should be pre-installed on gpu-mi300x1-base image
# Just make sure environment is properly set up
echo 'export PATH=/opt/rocm/bin:$PATH' >> /etc/profile.d/rocm.sh
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> /etc/profile.d/rocm.sh
echo 'export HIP_PATH=/opt/rocm' >> /etc/profile.d/rocm.sh

# Make hipcc globally accessible
ln -sf /opt/rocm/bin/hipcc /usr/local/bin/hipcc

echo "Setup completed"
"""

        body = {
            "name": self.droplet_name,
            "region": self.droplet_region,
            "size": self.droplet_size,
            "image": self.droplet_image,
            "ssh_keys": ssh_keys,
            "user_data": user_data,
            "tags": ["chisel"],
        }

        response = self.client.client.droplets.create(body=body)
        return response["droplet"]

    def wait_for_droplet(self, droplet_id: int, timeout: int = 300) -> Dict[str, Any]:
        """Wait for droplet to be active."""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Activating droplet...", total=None)

            while time.time() - start_time < timeout:
                response = self.client.client.droplets.get(droplet_id)
                droplet = response["droplet"]

                if droplet["status"] == "active":
                    # Get the public IP
                    for network in droplet["networks"]["v4"]:
                        if network["type"] == "public":
                            droplet["ip"] = network["ip_address"]
                            return droplet

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

    def destroy_droplet(self, droplet_id: int) -> None:
        """Destroy a droplet."""
        self.client.client.droplets.destroy(droplet_id)

    def up(self) -> Dict[str, Any]:
        """Create or reuse a droplet."""
        # Check for existing droplet
        existing = self.find_existing_droplet()

        if existing:
            console.print(f"[green]Found existing droplet: {existing['name']}[/green]")

            # Get the public IP
            for network in existing["networks"]["v4"]:
                if network["type"] == "public":
                    existing["ip"] = network["ip_address"]
                    break

            # Save state with creation time from droplet info
            self.state.save(
                existing["id"],
                existing.get("ip", ""),
                existing["name"],
                existing.get("created_at"),
            )
            return existing

        # Create new droplet
        console.print("[yellow]Creating new droplet...[/yellow]")
        droplet = self.create_droplet()

        # Wait for it to be active
        droplet = self.wait_for_droplet(droplet["id"])

        # Wait for SSH
        if self.wait_for_ssh(droplet["ip"]):
            console.print("[green]Droplet ready![/green]")
        else:
            console.print("[yellow]Warning: SSH may not be fully ready yet[/yellow]")

        # Save state with creation time
        self.state.save(
            droplet["id"], droplet["ip"], droplet["name"], droplet.get("created_at")
        )

        return droplet

    def down(self) -> bool:
        """Destroy the current droplet."""
        state_info = self.state.get_droplet_info()

        if not state_info:
            console.print("[yellow]No active droplet found[/yellow]")
            return False

        try:
            console.print(
                f"[yellow]Destroying droplet {state_info['name']}...[/yellow]"
            )
            self.destroy_droplet(state_info["droplet_id"])
            self.state.clear()
            console.print("[green]Droplet destroyed[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Error destroying droplet: {e}[/red]")
            # Clear state anyway if droplet doesn't exist
            if "404" in str(e):
                self.state.clear()
            return False

    def list_droplets(self) -> List[Dict[str, Any]]:
        """List all chisel droplets."""
        try:
            response = self.client.client.droplets.list()
            droplets = response.get("droplets", [])

            # Filter for chisel droplets
            chisel_droplets = []
            for droplet in droplets:
                if droplet["name"] == self.droplet_name or "chisel" in droplet.get(
                    "tags", []
                ):
                    # Get the public IP
                    for network in droplet["networks"]["v4"]:
                        if network["type"] == "public":
                            droplet["ip"] = network["ip_address"]
                            break
                    chisel_droplets.append(droplet)

            return chisel_droplets
        except Exception as e:
            console.print(f"[red]Error listing droplets: {e}[/red]")
            return []
