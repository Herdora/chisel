import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import paramiko
import pydo
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .chisel_client import ChiselClient
from .droplet import Droplet
from .types.gpu_profiles import GPU_PROFILES, GPU_TYPE_TO_STRING, GPUType
from .types.pydo_create_api import (
    DropletCreateRequest,
    DropletCreateResponse,
    PydoDropletObject,
)

console = Console()

DIGITAL_OCEAN_METADATA_URL = "http://169.254.169.254/metadata/v1/id"
STARTUP_SCRIPTS_DIR = Path(__file__).parent / "startup_scripts"


class DropletService:
    def __init__(
        self, token: str, chisel_client: Optional[ChiselClient] = None
    ):
        self.pydo_client = pydo.Client(token=token)
        self.chisel_client = chisel_client
        self._managed_mode = chisel_client is not None

    def _get_local_ssh_key(self) -> Optional[str]:
        ssh_pub_paths = [
            os.path.expanduser("~/.ssh/id_rsa.pub"),
            os.path.expanduser("~/.ssh/id_ed25519.pub"),
            os.path.expanduser("~/.ssh/id_ecdsa.pub"),
        ]

        for pub_path in ssh_pub_paths:
            if os.path.exists(pub_path):
                with open(pub_path, "r") as f:
                    return f.read().strip()
        return None

    def _validate_ssh_key_exists(self) -> None:
        local_key = self._get_local_ssh_key()
        if not local_key:
            raise ValueError(
                "No local SSH key found. Please generate an SSH key pair using 'ssh-keygen -t ed25519' "
                "and add the public key to your DigitalOcean account at https://amd.digitalocean.com/account/security"
            )

        try:
            response = self.pydo_client.ssh_keys.list()
            account_keys = response.get("ssh_keys", [])

            for key in account_keys:
                if key.get("public_key", "").strip() == local_key:
                    return

            raise ValueError(
                "Your local SSH key is not found in your DigitalOcean account. "
                "Please add your public key at https://amd.digitalocean.com/account/security"
            )

        except Exception as e:
            if "Your local SSH key is not found" in str(e):
                raise
            raise ValueError(f"Failed to validate SSH keys: {e}")

    def create_droplet(
        self,
        gpu_type: GPUType,
        droplet_region: str,
        droplet_size: str,
        droplet_image: str,
    ) -> Droplet:
        if self._managed_mode and self.chisel_client:
            gpu_profile = GPU_PROFILES[gpu_type]
            estimated_cost = gpu_profile.hourly_cost

            if not self.chisel_client.check_credits(estimated_cost):
                status = self.chisel_client.get_user_status()
                credits = status.get("credits_remaining", 0)
                raise ValueError(
                    f"Insufficient credits for {gpu_type.value} (${estimated_cost:.2f}/hour). "
                    f"You have ${credits:.2f} remaining."
                )

            console.print(
                f"[green]✓ Credit check passed for {gpu_type.value}[/green]"
            )

        self._validate_ssh_key_exists()

        script_path = STARTUP_SCRIPTS_DIR / f"{GPU_TYPE_TO_STRING[gpu_type]}.sh"
        with open(script_path, "r") as f:
            user_data = f.read()

        droplet_request = DropletCreateRequest(
            name=gpu_type.value,
            region=droplet_region,
            size=droplet_size,
            image=droplet_image,
            ssh_keys=cast(List[Union[str, int]], self.list_account_ssh_keys()),
            user_data=user_data,
            tags=[gpu_type.value],
        )

        response = self.pydo_client.droplets.create(
            body=droplet_request.to_dict()
        )
        if not response.get("droplet"):
            raise ValueError("Failed to create droplet")
        return Droplet(
            DropletCreateResponse.from_dict(
                cast(Dict[str, Any], response)
            ).droplet
        )

    def get_droplet_by_id(self, droplet_id: int) -> Optional[Droplet]:
        try:
            response = self.pydo_client.droplets.get(droplet_id)
            droplet_obj = PydoDropletObject.from_dict(response["droplet"])
            return Droplet(droplet_obj)
        except Exception as e:
            console.print(f"[red]Error fetching droplet: {e}[/red]")
            return None

    def get_droplet_by_type(self, droplet_name: str) -> Optional[Droplet]:
        try:
            response = self.pydo_client.droplets.list()
            droplets = response.get("droplets", [])
            for droplet_data in droplets:
                if droplet_data["name"] == droplet_name:
                    droplet_obj = PydoDropletObject.from_dict(droplet_data)
                    return Droplet(droplet_obj)
            return None
        except Exception:
            return None

    def get_or_create_droplet_by_type(self, gpu_type: GPUType) -> Droplet:
        console.print(
            f"[cyan]Ensuring {gpu_type.value} droplet is ready...[/cyan]"
        )
        existing = self.get_droplet_by_type(gpu_type.value)

        if existing:
            console.print(
                f"[green]Found existing droplet: {existing.name}[/green]"
            )
            # Register/update activity for managed users
            if self._managed_mode and self.chisel_client and existing.id:
                self.chisel_client.register_droplet(str(existing.id), gpu_type.value)
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

        # Register newly created droplet for managed users
        if self._managed_mode and self.chisel_client and droplet.id:
            result = self.chisel_client.register_droplet(str(droplet.id), gpu_type.value)
            if result.get("success"):
                console.print(f"[green]✓ Droplet registered for usage tracking[/green]")

        return droplet

    def destroy_droplet(self, droplet_id: int) -> None:
        self.pydo_client.droplets.destroy(droplet_id)

    def delete_all_droplets_by_type(self, gpu_type: GPUType) -> None:
        droplets = self.list_droplets()
        for droplet in droplets:
            if droplet.name == gpu_type.value:
                if droplet.id is not None:
                    self.destroy_droplet(droplet.id)

    def list_account_ssh_keys(self) -> List[int]:
        try:
            response = self.pydo_client.ssh_keys.list()
            return [key["id"] for key in response.get("ssh_keys", [])]
        except Exception as e:
            console.print(f"[red]Error fetching SSH keys: {e}[/red]")
            return []

    def list_droplets(self) -> List[Droplet]:
        try:
            response = self.pydo_client.droplets.list()
            droplets = response.get("droplets", [])
            return [Droplet(PydoDropletObject.from_dict(d)) for d in droplets]
        except Exception as e:
            console.print(f"[red]Error listing droplets: {e}[/red]")
            return []

    def wait_for_droplet(self, droplet_id: int, timeout: int = 300) -> Droplet:
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(
                "Activating droplet. (1-2 minutes remaining)...",
                total=None,
            )

            while time.time() - start_time < timeout:
                response = self.pydo_client.droplets.get(droplet_id)
                droplet_data = response["droplet"]

                if droplet_data["status"] == "active":
                    droplet_obj = PydoDropletObject.from_dict(droplet_data)
                    return Droplet(droplet_obj)

                time.sleep(5)

        raise TimeoutError("Droplet failed to become active within timeout")

    def wait_for_ssh(self, ip: str, timeout: int = 300) -> bool:
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(
                "Waiting for SSH to be ready. (< 30 seconds remaining)...",
                total=None,
            )

            while time.time() - start_time < timeout:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((ip, 22))
                    sock.close()

                    if result == 0:
                        ssh = paramiko.SSHClient()
                        ssh.set_missing_host_key_policy(
                            paramiko.AutoAddPolicy()
                        )
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
        try:
            account = self.pydo_client.account.get()
            return True, cast(Dict[str, Any], account)
        except Exception:
            return False, None

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        try:
            return cast(Dict[str, Any], self.pydo_client.account.get())
        except Exception as e:
            console.print(f"[red]Error fetching account info: {e}[/red]")
            return None

    def get_balance(self) -> Optional[Dict[str, Any]]:
        try:
            return cast(Dict[str, Any], self.pydo_client.balance.get())
        except Exception as e:
            console.print(f"[red]Error fetching balance: {e}[/red]")
            return None
