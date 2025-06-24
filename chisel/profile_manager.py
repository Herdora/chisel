"""Profile manager for orchestrating GPU profiling workflows."""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console

from chisel.config import Config
from chisel.do_client import DOClient
from chisel.droplet import DropletManager
from chisel.gpu_profiles import GPU_PROFILES
from chisel.ssh_manager import SSHManager

console = Console()


@dataclass
class TargetInfo:
    """Information about the profiling target."""

    raw_target: str
    is_source_file: bool
    file_path: Optional[Path] = None
    file_extension: Optional[str] = None
    compiler: Optional[str] = None


@dataclass
class ProfileResult:
    """Result of a profiling operation."""

    success: bool
    output_dir: Path
    stdout: str
    stderr: str
    summary: Dict[str, any]
    cost_estimate: float

    def display_summary(self):
        """Display a summary of the profiling results."""
        if self.success:
            console.print("\n[green]✓ Profiling completed successfully[/green]")
            console.print(f"[cyan]Results saved to:[/cyan] {self.output_dir}")

            # Show cost estimate
            console.print(f"[yellow]Estimated cost:[/yellow] ${self.cost_estimate:.2f}")

            # Show top kernels if available (AMD profiling)
            if "top_kernels" in self.summary:
                console.print("\n[cyan]Top GPU Kernels:[/cyan]")
                for i, kernel in enumerate(self.summary["top_kernels"][:5], 1):
                    console.print(
                        f"  {i}. {kernel['name'][:50]:<50} {kernel['time_ms']:8.3f} ms"
                    )

            # Show NVIDIA profiling results
            if "profile_files" in self.summary:
                console.print(
                    f"\n[cyan]Profile files generated:[/cyan] {len(self.summary['profile_files'])}"
                )
                for profile_file in self.summary["profile_files"]:
                    console.print(f"  • {profile_file}")
        else:
            console.print("\n[red]✗ Profiling failed[/red]")
            if self.stderr:
                console.print(f"[red]Error:[/red] {self.stderr}")


class ProfileManager:
    """Manages the complete profiling workflow for GPU kernels."""

    def __init__(self):
        self.config = Config()
        if not self.config.token:
            raise RuntimeError("No API token configured. Run 'chisel configure' first.")

        self.do_client = DOClient(self.config.token)

        # We'll use a separate state file for the new profiling system
        from chisel.profile_state import ProfileState

        self.state = ProfileState()

    def profile(self, vendor: str, target: str) -> ProfileResult:
        """
        Execute a complete profiling workflow.

        Args:
            vendor: Either "nvidia" or "amd"
            target: File path or command to profile

        Returns:
            ProfileResult with profiling data and summary
        """
        start_time = time.time()

        # Map vendor to GPU type
        gpu_type = "nvidia-h100" if vendor == "nvidia" else "amd-mi300x"

        try:
            # 1. Ensure droplet exists
            console.print(f"[cyan]Ensuring {vendor.upper()} droplet is ready...[/cyan]")
            droplet_info = self._ensure_droplet(gpu_type)

            # 2. Analyze the target
            target_info = self._analyze_target(target)

            # 3. Prepare the command
            if target_info.is_source_file:
                console.print(f"[cyan]Syncing {target_info.file_path.name}...[/cyan]")
                self._sync_file(droplet_info, target_info.file_path)
                command = self._build_command(vendor, target_info)
            else:
                command = target

            # 4. Run profiling
            console.print("[cyan]Running profiler...[/cyan]")
            profile_output = self._run_profiler(droplet_info, vendor, command)

            # 5. Calculate cost
            elapsed_hours = (time.time() - start_time) / 3600
            hourly_rate = 4.89 if vendor == "nvidia" else 1.99
            cost_estimate = elapsed_hours * hourly_rate

            # 6. Update last activity
            self.state.update_activity(gpu_type)

            return ProfileResult(
                success=True,
                output_dir=profile_output["output_dir"],
                stdout=profile_output["stdout"],
                stderr=profile_output["stderr"],
                summary=profile_output["summary"],
                cost_estimate=cost_estimate,
            )

        except Exception as e:
            console.print(f"[red]Error during profiling: {e}[/red]")
            return ProfileResult(
                success=False,
                output_dir=Path("./chisel-results/failed"),
                stdout="",
                stderr=str(e),
                summary={},
                cost_estimate=0.0,
            )

    def _ensure_droplet(self, gpu_type: str) -> Dict[str, any]:
        """Ensure a droplet exists for the given GPU type."""
        # Check if we have an active droplet
        droplet_info = self.state.get_droplet(gpu_type)

        if droplet_info and self._is_droplet_alive(droplet_info):
            console.print(
                f"[green]Using existing droplet: {droplet_info['name']}[/green]"
            )
            return droplet_info

        # Create new droplet
        console.print(f"[yellow]Creating new {gpu_type} droplet...[/yellow]")
        gpu_profile = GPU_PROFILES[gpu_type]
        droplet_manager = DropletManager(self.do_client, gpu_profile, gpu_type)

        # Create droplet with simplified name
        vendor = "nvidia" if "nvidia" in gpu_type else "amd"
        droplet_manager.droplet_name = f"chisel-{vendor}"

        droplet = droplet_manager.up()

        # Save to our state
        droplet_info = {
            "id": droplet["id"],
            "name": droplet["name"],
            "ip": droplet["ip"],
            "gpu_type": gpu_type,
            "created_at": droplet["created_at"],
        }
        self.state.save_droplet(gpu_type, droplet_info)

        return droplet_info

    def _is_droplet_alive(self, droplet_info: Dict[str, any]) -> bool:
        """Check if a droplet is still alive and accessible."""
        try:
            # Try to get droplet from DO API
            response = self.do_client.client.droplets.get(droplet_info["id"])
            if response and response["droplet"]["status"] == "active":
                # Update IP if changed
                current_ip = response["droplet"]["networks"]["v4"][0]["ip_address"]
                if current_ip != droplet_info["ip"]:
                    droplet_info["ip"] = current_ip
                    self.state.save_droplet(droplet_info["gpu_type"], droplet_info)
                return True
        except Exception:
            pass
        return False

    def _analyze_target(self, target: str) -> TargetInfo:
        """Analyze the target to determine if it's a file or command."""
        # Check if it's a file that exists
        target_path = Path(target)

        if target_path.exists() and target_path.is_file():
            extension = target_path.suffix.lower()

            # Determine compiler based on extension
            compiler_map = {
                ".cpp": "hipcc",
                ".hip": "hipcc",
                ".cu": "nvcc",
                ".c": "gcc",
            }

            return TargetInfo(
                raw_target=target,
                is_source_file=True,
                file_path=target_path.resolve(),
                file_extension=extension,
                compiler=compiler_map.get(extension, "gcc"),
            )

        # It's a command
        return TargetInfo(raw_target=target, is_source_file=False)

    def _sync_file(self, droplet_info: Dict[str, any], file_path: Path):
        """Sync a file to the droplet."""
        ssh_manager = SSHManager()

        # For the new system, we'll sync to /tmp for simplicity
        success = ssh_manager.sync(str(file_path), "/tmp/", droplet_info["gpu_type"])

        if not success:
            raise RuntimeError(f"Failed to sync {file_path}")

    def _build_command(self, vendor: str, target_info: TargetInfo) -> str:
        """Build the compilation and execution command."""
        remote_source = f"/tmp/{target_info.file_path.name}"
        binary_name = target_info.file_path.stem
        remote_binary = f"/tmp/{binary_name}"

        if vendor == "nvidia":
            if target_info.compiler == "nvcc":
                # Add -lineinfo for better profiling source mapping
                compile_cmd = f"nvcc -O3 -lineinfo {remote_source} -o {remote_binary}"
            else:
                # For non-CUDA files on NVIDIA
                compile_cmd = f"gcc {remote_source} -o {remote_binary}"
        else:  # AMD
            if target_info.compiler == "hipcc":
                compile_cmd = f"hipcc {remote_source} -o {remote_binary}"
            else:
                compile_cmd = f"gcc {remote_source} -o {remote_binary}"

        # Return compile and run command
        return f"{compile_cmd} && {remote_binary}"

    def _run_profiler(
        self, droplet_info: Dict[str, any], vendor: str, command: str
    ) -> Dict[str, any]:
        """Run the profiler on the droplet."""
        ssh_manager = SSHManager()

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(f"./chisel-results/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

        if vendor == "amd":
            # Use existing profile method for AMD
            ssh_manager.profile(
                command,
                droplet_info["gpu_type"],
                trace="hip,hsa",
                output_dir=str(output_dir),
                open_result=False,
            )

            # Parse results
            summary = self._parse_amd_results(output_dir)

            return {
                "output_dir": output_dir,
                "stdout": "",
                "stderr": "",
                "summary": summary,
            }
        else:
            # NVIDIA profiling with nsight-compute
            try:
                return self._run_nvidia_profiler(droplet_info, command, output_dir)
            except Exception as e:
                console.print(f"[yellow]NVIDIA profiling failed: {e}[/yellow]")
                console.print("[yellow]Falling back to basic execution...[/yellow]")

                # Fallback to basic execution
                exit_code = ssh_manager.run(command, droplet_info["gpu_type"])

                return {
                    "output_dir": output_dir,
                    "stdout": f"Command executed with exit code: {exit_code}",
                    "stderr": str(e),
                    "summary": {},
                }

    def _parse_amd_results(self, output_dir: Path) -> Dict[str, any]:
        """Parse AMD profiling results."""
        summary = {}

        # Look for results files
        profile_dir = output_dir / "chisel_profile"
        if not profile_dir.exists():
            return summary

        # Try to find and parse results
        import json

        # Try JSON first
        json_file = profile_dir / "results.json"
        if json_file.exists():
            try:
                with open(json_file) as f:
                    data = json.load(f)

                kernels = []
                for event in data.get("traceEvents", []):
                    if (
                        event.get("ph") == "X"
                        and "pid" in event
                        and event.get("pid") in [6, 7]
                        and "DurationNs" in event.get("args", {})
                    ):
                        kernels.append(
                            {
                                "name": event.get("name", ""),
                                "time_ms": event["args"]["DurationNs"] / 1_000_000,
                            }
                        )

                # Sort by time
                kernels.sort(key=lambda x: x["time_ms"], reverse=True)
                summary["top_kernels"] = kernels[:10]

            except Exception as e:
                console.print(f"[yellow]Could not parse JSON results: {e}[/yellow]")

        return summary

    def _run_nvidia_profiler(
        self, droplet_info: Dict[str, any], command: str, output_dir: Path
    ) -> Dict[str, any]:
        """Run NVIDIA nsight-compute profiler on the droplet."""
        ssh_manager = SSHManager()

        # Ensure nsight-compute is available
        self._ensure_nsight_compute(droplet_info)

        # Setup remote profiling environment
        remote_profile_dir = "/tmp/chisel_nvidia_profile"
        profile_filename = f"profile_{int(time.time())}"

        # Build nsight-compute profiling command
        profile_setup = f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir} && cd {remote_profile_dir}"

        # For NVIDIA, we need to separate compilation from profiling
        # The command might be a compile+run, so we need to handle this properly
        if " && " in command:
            # Split compilation and execution
            compile_part, execute_part = command.split(" && ", 1)
            # Execute compilation first, then profile the execution
            ncu_cmd = f"{compile_part} && ncu --set full --target-processes all --export {profile_filename}_%p.ncu-rep {execute_part}"
        else:
            # Just profile the single command
            ncu_cmd = f"ncu --set full --target-processes all --export {profile_filename}_%p.ncu-rep {command}"

        full_cmd = f"{profile_setup} && {ncu_cmd}"

        console.print(f"[cyan]Running NVIDIA profiler: {command}[/cyan]")

        # Execute profiling command
        exit_code = ssh_manager.run(full_cmd, droplet_info["gpu_type"])

        if exit_code != 0:
            raise RuntimeError(f"NVIDIA profiling failed with exit code {exit_code}")

        # Download results without parsing - let users analyze .ncu-rep files with proper tools
        profile_files = self._download_nvidia_results(
            droplet_info, remote_profile_dir, output_dir
        )

        # Create basic summary
        summary = {
            "profile_files": profile_files,
            "message": "NVIDIA profiling completed. Use ncu-ui or ncu --import to analyze the .ncu-rep files.",
        }

        # Cleanup remote files
        self._cleanup_nvidia_remote(droplet_info, remote_profile_dir)

        return {
            "output_dir": output_dir,
            "stdout": "NVIDIA profiling completed successfully",
            "stderr": "",
            "summary": summary,
        }

    def _ensure_nsight_compute(self, droplet_info: Dict[str, any]):
        """Ensure nsight-compute is installed on the droplet."""
        ssh_manager = SSHManager()

        try:
            # Check if ncu is already available
            check_cmd = "which ncu && ncu --version"
            exit_code = ssh_manager.run(check_cmd, droplet_info["gpu_type"])

            if exit_code == 0:
                console.print("[green]✓ nsight-compute already available[/green]")
                return

            console.print("[yellow]Installing nsight-compute...[/yellow]")

            # Install nsight-compute with timeout
            install_cmd = """
            timeout 300 bash -c '
            apt-get update -y && 
            apt-get install -y nvidia-nsight-compute
            '
            """

            exit_code = ssh_manager.run(install_cmd, droplet_info["gpu_type"])

            if exit_code != 0:
                raise RuntimeError(
                    "Failed to install nsight-compute. This may be due to package repository issues or network connectivity."
                )

            # Verify installation
            exit_code = ssh_manager.run(
                "which ncu && ncu --version", droplet_info["gpu_type"]
            )

            if exit_code != 0:
                raise RuntimeError(
                    "nsight-compute installation verification failed. The ncu command is not available after installation."
                )

            console.print("[green]✓ nsight-compute installed successfully[/green]")

        except Exception as e:
            raise RuntimeError(f"Failed to setup nsight-compute: {e}")

    def _download_nvidia_results(
        self, droplet_info: Dict[str, any], remote_dir: str, local_output_dir: Path
    ) -> list:
        """Download NVIDIA profiling results and return list of files."""
        import subprocess
        import tarfile

        ssh_manager = SSHManager()
        ip = droplet_info["ip"]

        console.print("[cyan]Downloading NVIDIA profiling results...[/cyan]")

        # Create archive on remote
        archive_cmd = f"cd {remote_dir} && tar -czf nvidia_profile.tgz *.ncu-rep"
        exit_code = ssh_manager.run(archive_cmd, droplet_info["gpu_type"])

        if exit_code != 0:
            console.print(
                "[yellow]Warning: No .ncu-rep files found or archive creation failed[/yellow]"
            )
            return []

        # Download archive
        local_archive_path = local_output_dir / "nvidia_profile.tgz"

        scp_cmd = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            f"root@{ip}:{remote_dir}/nvidia_profile.tgz",
            str(local_archive_path),
        ]

        try:
            result = subprocess.run(
                scp_cmd, capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                console.print(
                    f"[yellow]Warning: Failed to download NVIDIA profile results: {result.stderr}[/yellow]"
                )
                return []

            # Verify archive was downloaded
            if (
                not local_archive_path.exists()
                or local_archive_path.stat().st_size == 0
            ):
                console.print(
                    "[yellow]Warning: Downloaded archive is empty or missing[/yellow]"
                )
                return []

            # Extract archive
            nvidia_results_dir = local_output_dir / "nvidia_profile"
            nvidia_results_dir.mkdir(exist_ok=True)

            with tarfile.open(local_archive_path, "r:gz") as tar:
                tar.extractall(nvidia_results_dir)

            # Verify extraction and return file list
            ncu_files = list(nvidia_results_dir.glob("*.ncu-rep"))
            if not ncu_files:
                console.print(
                    "[yellow]Warning: No .ncu-rep files found in extracted archive[/yellow]"
                )
                return []
            else:
                console.print(
                    f"[green]✓ NVIDIA profile results saved to {nvidia_results_dir} ({len(ncu_files)} files)[/green]"
                )
                return [f.name for f in ncu_files]

        except subprocess.TimeoutExpired:
            console.print("[yellow]Warning: Download timed out[/yellow]")
            return []
        except tarfile.TarError as e:
            console.print(f"[yellow]Warning: Failed to extract archive: {e}[/yellow]")
            return []
        except Exception as e:
            console.print(
                f"[yellow]Warning: Unexpected error during download: {e}[/yellow]"
            )
            return []
        finally:
            # Clean up archive if it exists
            if local_archive_path.exists():
                local_archive_path.unlink()

    def _cleanup_nvidia_remote(self, droplet_info: Dict[str, any], remote_dir: str):
        """Clean up remote NVIDIA profiling files."""
        ssh_manager = SSHManager()

        cleanup_cmd = f"rm -rf {remote_dir}"
        ssh_manager.run(cleanup_cmd, droplet_info["gpu_type"])

        console.print("[green]✓ Remote cleanup completed[/green]")
