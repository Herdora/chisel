"""Profile manager for orchestrating GPU profiling workflows."""

# TODO: Have the name of profile output be <target>-<vendor>-<gpu>-<time>-<date>

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

from rich.console import Console

from chisel.core.config import Config
from chisel.core.do_client import DOClient
from chisel.core.droplet import DropletManager
from chisel.core.gpu_profiles import GPU_PROFILES
from chisel.core.profiling_state import ProfilingState
from chisel.core.ssh_manager import SSHManager

console = Console()

CHISEL_PROFILING_DIR_NAME = "chisel-results"


@dataclass
class TargetInfo:
    """Information about the profiling target."""

    raw_target: str
    is_source_file: bool
    file_path: Optional[Path] = None
    file_extension: Optional[str] = None
    compiler: Optional[str] = None


@dataclass
class ProfilingResults:
    """Result of a profiling operation."""

    success: bool
    output_dir: Path
    stdout: str
    stderr: str
    summary: Dict[str, Any]
    cost_estimate: float

    def display_summary(self):
        """Display a summary of the profiling results."""
        if self.success:
            console.print("\n[green]✓ Profiling completed successfully[/green]")
            console.print(f"[cyan]Results saved to:[/cyan] {self.output_dir}")

            # Show cost estimate
            console.print(f"[yellow]Estimated cost:[/yellow] ${self.cost_estimate:.2f}")

            # Show top kernels if available (AMD legacy profiling)
            if "top_kernels" in self.summary:
                console.print("\n[cyan]Top GPU Kernels:[/cyan]")
                for i, kernel in enumerate(self.summary["top_kernels"][:5], 1):
                    console.print(f"  {i}. {kernel['name'][:50]:<50} {kernel['time_ms']:8.3f} ms")

            # Show profiling results (both AMD and NVIDIA use same structure now)
            if "profile_files" in self.summary:
                summary_file = self.summary.get("summary_file")
                profile_type = self.summary.get("profile_type", "nvidia")

                if summary_file:
                    vendor_name = "AMD rocprofv3" if profile_type == "rocprofv3" else "NVIDIA"
                    console.print(
                        f"\n[cyan]{vendor_name} profile summary generated:[/cyan] {summary_file}"
                    )

                    console.print("\n[cyan]Analysis tools:[/cyan]")
                    console.print("  • View text summary for human-readable kernel analysis")
                else:
                    console.print("\n[cyan]Profile files generated:[/cyan] 0 files")
        else:
            console.print("\n[red]✗ Profiling failed[/red]")
            if self.stderr:
                console.print(f"[red]Error:[/red] {self.stderr}")


class ProfilingManager:
    """Manages the complete profiling workflow for GPU kernels."""

    def __init__(self):
        self.config = Config()
        if not self.config.token:
            raise RuntimeError("No API token configured. Run 'chisel configure' first.")
        self.do_client = DOClient(self.config.token)
        self.state = ProfilingState()

        # We'll use a separate state file for the new profiling system

    def profile(
        self,
        vendor: str,
        target: str,
        gpu_type: Optional[str] = None,
        output_dir: Optional[str] = None,
        rocprofv3_flag: Optional[str] = None,
        rocprof_compute_flag: Optional[str] = None,
        nsys_flag: Optional[str] = None,
        ncompute_flag: Optional[str] = None,
    ) -> ProfilingResults:
        """
        Execute a complete profiling workflow.

        Args:
            vendor: Either "nvidia" or "amd"
            target: File path or command to profile
            gpu_type: GPU type override - "h100" or "l40s" for NVIDIA (optional)
            output_dir: Custom output directory for results (optional)
            rocprofv3_flag: Full command to run with rocprofv3 (AMD)
            rocprof_compute_flag: Full command to run with rocprof-compute (AMD)
            nsys_flag: Full command to run with nsys (NVIDIA)
            ncompute_flag: Full command to run with ncu (NVIDIA)

        Returns:
            ProfilingResults with profiling data and summary
        """

        start_time = time.time()

        # TODO: resolved_gpu_type will be phased out when heterogenous profiling is implemented
        if vendor == "nvidia":
            resolved_gpu_type = f"nvidia-{gpu_type}" if gpu_type else "nvidia-h100"
        else:
            resolved_gpu_type = "amd-mi300x"

        try:
            console.print(f"[cyan]Ensuring {vendor.upper()} droplet is ready...[/cyan]")
            droplet_info = self._ensure_droplet(resolved_gpu_type)
            console.print(f"[green]Droplet {droplet_info['name']} is ready[/green]")
        except Exception as e:
            console.print(f"[red]Error during droplet setup: {e}[/red]")
            return ProfilingResults(
                success=False,
                output_dir=Path(f"./{CHISEL_PROFILING_DIR_NAME}/failed"),
                stdout="",
                stderr=str(e),
                summary={},
                cost_estimate=0.0,
            )

        try:
            target_info = self._get_target_info(target)
            if target_info.is_source_file and target_info.file_path:
                console.print(
                    f"[cyan]Syncing {target_info.file_path.name} to remote server...[/cyan]"
                )

            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                timestamp = datetime.now().strftime("%H%M%S-%Y%m%d")
                output_path = Path(f"./{CHISEL_PROFILING_DIR_NAME}-{timestamp}")
                output_path.mkdir(parents=True, exist_ok=True)

            all_results = []
            if rocprofv3_flag:
                result = self.run_rocprofv3(droplet_info, target_info, output_path, rocprofv3_flag)
                all_results.append(result)
            if rocprof_compute_flag:
                result = self.run_rocprof_compute(
                    droplet_info, target_info.raw_target, output_path, rocprof_compute_flag
                )
                all_results.append(result)
            if nsys_flag:
                result = self.run_nsys(droplet_info, target_info.raw_target, output_path, nsys_flag)
                all_results.append(result)
            if ncompute_flag:
                result = self.run_ncompute(
                    droplet_info, target_info.raw_target, output_path, ncompute_flag
                )
                all_results.append(result)

            elapsed_hours = (time.time() - start_time) / 3600
            hourly_rate = 4.89 if vendor == "nvidia" else 1.99
            cost_estimate = elapsed_hours * hourly_rate

            return ProfilingResults(
                success=True,
                output_dir=output_path,
                stdout="",
                stderr="",
                summary={
                    "profile_files": [result["local_output_dir"] for result in all_results],
                    "summary_file": all_results[0]["summary"]["summary_file"],
                    "profile_type": all_results[0]["summary"]["profile_type"],
                    "message": "Profiling completed. Generated profile data.",
                },
                cost_estimate=cost_estimate,
            )

        except Exception as e:
            console.print(f"[red]Error during profiling: {e}[/red]")
            return ProfilingResults(
                success=False,
                output_dir=Path(f"./{CHISEL_PROFILING_DIR_NAME}/failed"),
                stdout="",
                stderr=str(e),
                summary={},
                cost_estimate=0.0,
            )

    def run_rocprofv3(
        self,
        droplet_info: Dict[str, Any],
        target: TargetInfo,
        local_output_dir: Path,
        extra_flags: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run rocprofv3 on the droplet."""
        ssh_manager = SSHManager()
        self._ensure_rocprofv3(droplet_info)

        remote_profile_dir = "/tmp/chisel-rocprofv3"

        # Get the source file name and binary name
        source_file = target.file_path or Path(target.raw_target)
        source_name = source_file.name
        binary_name = source_file.stem
        remote_source = f"{remote_profile_dir}/{source_name}"
        remote_binary = f"{remote_profile_dir}/{binary_name}"

        EXPORT_LIB_CMD = "export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/"
        RESET_DIR_CMD = f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"
        CD_CMD = f"cd {remote_profile_dir}"
        BUILD_CMD = f"hipcc {remote_source} -o {remote_binary}"  # TODO: add python support
        PROFILE_CMD = f"rocprofv3 -S --summary-output-file amd_profile_summary.txt {extra_flags or '--sys-trace'} -- {remote_binary}"

        # First reset directory, then sync file, then build and profile
        reset_cmd = f"{EXPORT_LIB_CMD} && {RESET_DIR_CMD}"
        exit_code = ssh_manager.run(reset_cmd, droplet_info["gpu_type"])
        if exit_code != 0:
            raise RuntimeError(f"Failed to reset remote directory: {exit_code}")

        # Now sync the file to the clean directory
        _ = self._sync_file(
            droplet_info, target.file_path or Path(target.raw_target), remote_profile_dir
        )

        # Build and profile
        build_profile_cmd = f"{CD_CMD} && {BUILD_CMD} && {PROFILE_CMD}"
        full_cmd = build_profile_cmd
        console.print(f"[cyan]Running AMD rocprofv3 with flags '{full_cmd}'[/cyan]")
        rocprof_exit_code = ssh_manager.run(full_cmd, droplet_info["gpu_type"])
        if rocprof_exit_code != 0:
            raise RuntimeError(f"rocprofv3 profiling failed with exit code {rocprof_exit_code}")

        rocprof_files = self._download_results(droplet_info, remote_profile_dir, local_output_dir)
        self._cleanup_amd_remote(droplet_info, remote_profile_dir)

        return {
            "local_output_dir": local_output_dir,
            "stdout": "AMD rocprofv3 profiling completed successfully",
            "stderr": "",
            "summary": {
                "profile_files": rocprof_files,
                "summary_file": rocprof_files[0] if rocprof_files else None,
                "profile_type": "rocprofv3",
                "message": "AMD rocprofv3 profiling completed. Generated profile summary.",
            },
        }

    def run_rocprof_compute(
        self,
        droplet_info: Dict[str, Any],
        command: str,
        local_output_dir: Path,
        extra_flags: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run rocprof-compute on the droplet."""
        # TODO: Implement rocprof-compute when ready

        console.print("[yellow]rocprof-compute support not yet implemented[/yellow]")
        raise RuntimeError("rocprof-compute is not yet supported")

    def run_nsys(
        self,
        droplet_info: Dict[str, Any],
        command: str,
        local_output_dir: Path,
        extra_flags: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run nsys on the droplet."""
        ssh_manager = SSHManager()
        self._ensure_nvidia_profilers(droplet_info)

        remote_profile_dir = "/tmp/chisel-nsys"

        # Combine setup and profiling into a single atomic command to ensure cd works properly
        def make_full_cmd(remote_profile_dir: str, extra_flags: str):
            return f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir} && cd {remote_profile_dir} && nsys profile {extra_flags or '--stats=true --force-overwrite=true'} -o nvidia_profile -- {command}"

        full_cmd = make_full_cmd(
            remote_profile_dir, extra_flags or "--stats=true --force-overwrite=true"
        )
        console.print(f"[cyan]Running NVIDIA nsys with flags '{full_cmd}'[/cyan]")
        nsys_exit_code = ssh_manager.run(full_cmd, droplet_info["gpu_type"])
        if nsys_exit_code != 0:
            raise RuntimeError(f"nsys profiling failed with exit code {nsys_exit_code}")

        nvidia_files = self._download_results(droplet_info, remote_profile_dir, local_output_dir)

        self._cleanup_nvidia_remote(droplet_info, remote_profile_dir)

        return {
            "local_output_dir": local_output_dir,
            "stdout": "NVIDIA nsys profiling completed successfully",
            "stderr": "",
            "summary": {
                "profile_files": nvidia_files,
                "summary_file": nvidia_files[0] if nvidia_files else None,
                "profile_type": "nsys",
                "message": "NVIDIA nsys profiling completed. Generated profile data.",
            },
        }

    def run_ncompute(
        self,
        droplet_info: Dict[str, Any],
        command: str,
        local_output_dir: Path,
        extra_flags: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run ncu (nsight-compute) on the droplet."""
        ssh_manager = SSHManager()
        self._ensure_nvidia_profilers(droplet_info)

        remote_profile_dir = "/tmp/chisel-ncompute"

        # Combine setup and profiling into a single atomic command to ensure cd works properly
        def make_full_cmd(remote_profile_dir: str, extra_flags: str):
            return f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir} && cd {remote_profile_dir} && ncu {extra_flags or '--set full --force-overwrite'} -o nvidia_ncompute_profile -- {command}"

        full_cmd = make_full_cmd(remote_profile_dir, extra_flags or "--set full --force-overwrite")
        console.print(f"[cyan]Running NVIDIA ncu with flags '{full_cmd}'[/cyan]")
        ncu_exit_code = ssh_manager.run(full_cmd, droplet_info["gpu_type"])
        if ncu_exit_code != 0:
            raise RuntimeError(f"ncu profiling failed with exit code {ncu_exit_code}")

        nvidia_files = self._download_results(droplet_info, remote_profile_dir, local_output_dir)

        self._cleanup_nvidia_remote(droplet_info, remote_profile_dir)

        return {
            "local_output_dir": local_output_dir,
            "stdout": "NVIDIA ncu profiling completed successfully",
            "stderr": "",
            "summary": {
                "profile_files": nvidia_files,
                "summary_file": nvidia_files[0] if nvidia_files else None,
                "profile_type": "ncompute",
                "message": "NVIDIA ncu profiling completed. Generated profile data.",
            },
        }

    def _ensure_droplet(self, gpu_type: str) -> Dict[str, Any]:
        """Ensure a droplet exists for the given GPU type."""
        droplet_info = self.state.get_droplet(gpu_type)

        if droplet_info and self._is_droplet_alive(droplet_info):
            console.print(f"[green]Using existing droplet: {droplet_info['name']}[/green]")
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

    def _is_droplet_alive(self, droplet_info: Dict[str, Any]) -> bool:
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

    def _get_target_info(self, target: str) -> TargetInfo:
        """Analyze the target to determine if it's a file or command."""
        target_path = Path(target)
        extension = target_path.suffix.lower()

        compiler_map = {
            ".cpp": "hipcc",
            ".hip": "hipcc",
            ".cu": "nvcc",
            ".c": "gcc",
            ".py": "python",
        }

        is_source_extension = extension in compiler_map
        file_exists = target_path.exists() and target_path.is_file()
        if file_exists or is_source_extension:
            return TargetInfo(
                raw_target=target,
                is_source_file=True,
                file_path=target_path,
                file_extension=extension,
                compiler=compiler_map.get(extension, "gcc"),
            )

        return TargetInfo(raw_target=target, is_source_file=False)

    def _sync_file(self, droplet_info: Dict[str, Any], source_file: Path, remote_dir: str):
        """Sync a file to the droplet with proper temp directory setup."""
        ssh_manager = SSHManager()
        success = ssh_manager.sync(str(source_file), f"{remote_dir}/", droplet_info["gpu_type"])
        if not success:
            raise RuntimeError(
                f"Failed to sync {source_file} to {remote_dir}. Ensure the file exists and is accessible."
            )

        chmod_cmd = f"chmod +x {remote_dir}/{source_file.name}"
        exit_code = ssh_manager.run(chmod_cmd, droplet_info["gpu_type"])
        if exit_code != 0:
            console.print("[yellow]Warning: Failed to make file executable[/yellow]")
        console.print(f"[green]✓ File synced to {remote_dir} on remote server[/green]")

        return remote_dir

    def _parse_amd_results(self, output_dir: Path) -> Dict[str, Any]:
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

    def _ensure_nvidia_profilers(self, droplet_info: Dict[str, Any]):
        """Ensure both nsight-compute and nsight-systems are installed on the droplet."""
        ssh_manager = SSHManager()

        try:
            # Check if both profilers are already available
            check_cmd = "which ncu && ncu --version && which nsys && nsys --version"
            exit_code = ssh_manager.run(check_cmd, droplet_info["gpu_type"])

            if exit_code == 0:
                console.print("[green]✓ NVIDIA profilers (ncu + nsys) already available[/green]")
                return

            console.print(
                "[yellow]Installing NVIDIA profilers (nsight-compute + nsight-systems)...[/yellow]"
            )

            # Install both profilers with timeout
            install_cmd = """
            timeout 600 bash -c '
            apt-get update -y && 
            apt-get install -y nvidia-nsight-compute nvidia-nsight-systems
            '
            """

            exit_code = ssh_manager.run(install_cmd, droplet_info["gpu_type"])

            if exit_code != 0:
                raise RuntimeError(
                    "Failed to install NVIDIA profilers. This may be due to package repository issues or network connectivity."
                )

            # Verify both installations
            verify_ncu = ssh_manager.run("which ncu && ncu --version", droplet_info["gpu_type"])
            verify_nsys = ssh_manager.run("which nsys && nsys --version", droplet_info["gpu_type"])

            if verify_ncu != 0:
                raise RuntimeError(
                    "nsight-compute installation verification failed. The ncu command is not available after installation."
                )

            if verify_nsys != 0:
                raise RuntimeError(
                    "nsight-systems installation verification failed. The nsys command is not available after installation."
                )

            console.print("[green]✓ NVIDIA profilers installed successfully (ncu + nsys)[/green]")

        except Exception as e:
            raise RuntimeError(f"Failed to setup NVIDIA profilers: {e}")

    def _ensure_pytorch(self, droplet_info: Dict[str, Any]):
        """Ensure PyTorch with CUDA support is installed on the NVIDIA droplet."""
        ssh_manager = SSHManager()

        try:
            # Check if PyTorch is already available with CUDA
            check_cmd = "python3 -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\" 2>/dev/null"
            exit_code = ssh_manager.run(check_cmd, droplet_info["gpu_type"])

            if exit_code == 0:
                console.print("[green]✓ PyTorch with CUDA already available[/green]")
                return

            console.print("[yellow]Installing PyTorch with CUDA support...[/yellow]")

            # Install pip if not available
            install_pip_cmd = "apt update -y && apt install -y python3-pip"
            exit_code = ssh_manager.run(install_pip_cmd, droplet_info["gpu_type"])

            if exit_code != 0:
                raise RuntimeError("Failed to install pip")

            # Install PyTorch with CUDA support
            install_pytorch_cmd = (
                "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            )
            exit_code = ssh_manager.run(install_pytorch_cmd, droplet_info["gpu_type"])

            if exit_code != 0:
                raise RuntimeError("Failed to install PyTorch")

            # Verify PyTorch CUDA detection
            verify_cmd = "python3 -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')\""
            exit_code = ssh_manager.run(verify_cmd, droplet_info["gpu_type"])

            if exit_code != 0:
                raise RuntimeError("PyTorch installation verification failed")

            console.print("[green]✓ PyTorch with CUDA installed successfully[/green]")

        except Exception as e:
            raise RuntimeError(f"Failed to setup PyTorch: {e}")

    def _ensure_rocprofv3(self, droplet_info: Dict[str, Any]):
        """Ensure rocprofv3 and dependencies are installed on the AMD droplet."""
        ssh_manager = SSHManager()

        try:
            # Check if rocprofv3 is already available
            check_cmd = "which rocprofv3 && echo 'rocprofv3 available'"
            exit_code = ssh_manager.run(check_cmd, droplet_info["gpu_type"])

            if exit_code == 0:
                console.print("[green]✓ rocprofv3 already available[/green]")
                return

            console.print("[yellow]Installing rocprofv3 and dependencies...[/yellow]")

            # Install build dependencies and build tools
            setup_cmd = """
            timeout 1800 bash -c '
            apt-get update -y && 
            apt-get install -y git cmake build-essential python3 python3-pip wget
            '
            """

            exit_code = ssh_manager.run(setup_cmd, droplet_info["gpu_type"])
            if exit_code != 0:
                raise RuntimeError("Failed to install build dependencies")

            # Build aqlprofile from mainline
            build_aqlprofile_cmd = """
            cd /tmp && 
            git clone https://github.com/ROCm/aqlprofile.git && 
            cd aqlprofile && 
            mkdir build && cd build && 
            cmake .. && make -j$(nproc) && make install
            """

            console.print("[cyan]Building aqlprofile...[/cyan]")
            exit_code = ssh_manager.run(build_aqlprofile_cmd, droplet_info["gpu_type"])
            if exit_code != 0:
                raise RuntimeError("Failed to build aqlprofile")

            # Build rocprofiler-sdk from mainline
            build_rocprofiler_cmd = """
            cd /tmp && 
            git clone https://github.com/ROCm/rocprofiler-sdk.git && 
            cd rocprofiler-sdk && 
            mkdir build && cd build && 
            cmake .. && make -j$(nproc) && make install
            """

            console.print("[cyan]Building rocprofiler-sdk...[/cyan]")
            exit_code = ssh_manager.run(build_rocprofiler_cmd, droplet_info["gpu_type"])
            if exit_code != 0:
                raise RuntimeError("Failed to build rocprofiler-sdk")

            # Download rocprof-trace-decoder binary
            download_decoder_cmd = """
            cd /tmp && 
            wget -O /opt/rocm/lib/rocprof-trace-decoder https://github.com/ROCm/rocprof-trace-decoder/releases/latest/download/rocprof-trace-decoder && 
            chmod +x /opt/rocm/lib/rocprof-trace-decoder &&
            ln -sf /opt/rocm/lib/rocprof-trace-decoder /opt/rocm/lib/libatt_decoder_trace.so
            """

            console.print("[cyan]Installing rocprof-trace-decoder...[/cyan]")
            exit_code = ssh_manager.run(download_decoder_cmd, droplet_info["gpu_type"])
            if exit_code != 0:
                raise RuntimeError("Failed to install rocprof-trace-decoder")

            # Set up environment
            env_setup_cmd = """
            echo 'export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/' >> /root/.bashrc &&
            export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/
            """

            exit_code = ssh_manager.run(env_setup_cmd, droplet_info["gpu_type"])
            if exit_code != 0:
                raise RuntimeError("Failed to set up environment")

            # Verify installation
            verify_cmd = "export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ && which rocprofv3 && rocprofv3 --help"
            exit_code = ssh_manager.run(verify_cmd, droplet_info["gpu_type"])

            if exit_code != 0:
                raise RuntimeError("rocprofv3 installation verification failed")

            console.print("[green]✓ rocprofv3 and dependencies installed successfully[/green]")

        except Exception as e:
            raise RuntimeError(f"Failed to setup rocprofv3: {e}")

    def _download_results(
        self,
        droplet_info: Dict[str, Any],
        remote_dir: str,
        local_output_dir: Path,
    ) -> list:
        import subprocess

        ssh_manager = SSHManager()
        ip = droplet_info["ip"]
        console.print("[cyan]Downloading profiling results...[/cyan]")

        # Download all files from remote directory to local directory
        scp_cmd = [
            "scp",
            "-r",  # Recursive to download entire directory contents
            "-o",
            "StrictHostKeyChecking=no",
            f"root@{ip}:{remote_dir}/*",  # Download all files from remote directory
            str(local_output_dir),
        ]

        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                console.print(
                    f"[yellow]Warning: Failed to download profiling results: {result.stderr}[/yellow]"
                )
                return []

            # Flatten any subdirectories - move all files to the top level
            downloaded_files = []

            # Walk through all files and subdirectories
            all_files = []
            for item in local_output_dir.rglob("*"):
                if item.is_file():
                    all_files.append(item)

            # Move all files to the top level and clean up names
            for file_path in all_files:
                original_name = file_path.name
                # Remove numeric session ID prefixes (e.g., "40396_agent_info.csv" -> "agent_info.csv")
                import re

                clean_name = re.sub(r"^\d+_", "", original_name)

                # Target path in the top level directory
                target_path = local_output_dir / clean_name

                # If file is not already in the top level, move it there
                if file_path.parent != local_output_dir:
                    # Handle name conflicts by adding a counter if needed
                    counter = 1
                    while target_path.exists():
                        name_parts = clean_name.rsplit(".", 1)
                        if len(name_parts) == 2:
                            target_path = (
                                local_output_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
                            )
                        else:
                            target_path = local_output_dir / f"{clean_name}_{counter}"
                        counter += 1

                    file_path.rename(target_path)
                    console.print(
                        f"[green]✓ Downloaded: {original_name} -> {target_path.name}[/green]"
                    )
                    downloaded_files.append(target_path.name)
                else:
                    # File is already in top level, just rename if needed
                    if clean_name != original_name:
                        # Handle name conflicts
                        counter = 1
                        while target_path.exists() and target_path != file_path:
                            name_parts = clean_name.rsplit(".", 1)
                            if len(name_parts) == 2:
                                target_path = (
                                    local_output_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
                                )
                            else:
                                target_path = local_output_dir / f"{clean_name}_{counter}"
                            counter += 1

                        if target_path != file_path:
                            file_path.rename(target_path)
                        console.print(
                            f"[green]✓ Downloaded: {original_name} -> {target_path.name}[/green]"
                        )
                        downloaded_files.append(target_path.name)
                    else:
                        console.print(f"[green]✓ Downloaded: {original_name}[/green]")
                        downloaded_files.append(original_name)

            # Remove any empty subdirectories
            for item in local_output_dir.iterdir():
                if item.is_dir():
                    try:
                        item.rmdir()  # Only removes if empty
                        console.print(f"[green]✓ Removed empty directory: {item.name}[/green]")
                    except OSError:
                        # Directory not empty, leave it
                        pass

            if not downloaded_files:
                console.print("[yellow]Warning: No files were downloaded[/yellow]")
                return []

            console.print(
                f"[green]✓ Profiling results downloaded ({len(downloaded_files)} files)[/green]"
            )
            return downloaded_files

        except subprocess.TimeoutExpired:
            console.print("[yellow]Warning: Download timed out[/yellow]")
            return []
        except Exception as e:
            console.print(f"[yellow]Warning: Unexpected error during download: {e}[/yellow]")
            return []

    def _cleanup_amd_remote(self, droplet_info: Dict[str, Any], remote_dir: str):
        """Clean up remote AMD profiling files."""
        ssh_manager = SSHManager()

        cleanup_cmd = f"rm -rf {remote_dir}"
        ssh_manager.run(cleanup_cmd, droplet_info["gpu_type"])

        console.print("[green]✓ Remote cleanup completed[/green]")

    def _cleanup_nvidia_remote(self, droplet_info: Dict[str, Any], remote_dir: str):
        """Clean up remote NVIDIA profiling files."""
        ssh_manager = SSHManager()

        cleanup_cmd = f"rm -rf {remote_dir}"
        ssh_manager.run(cleanup_cmd, droplet_info["gpu_type"])

        console.print("[green]✓ Remote cleanup completed[/green]")
