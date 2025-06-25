"""Profile manager for orchestrating GPU profiling workflows."""

# TODO: Have the name of profile output be <target>-<vendor>-<gpu>-<time>-<date>

import time
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import os

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

            # Show AMD rocprofv3 profiling results
            if "att_files" in self.summary:
                rocprof_count = len(self.summary.get("att_files", []))
                console.print(
                    f"\n[cyan]AMD rocprofv3 profile files generated:[/cyan] {rocprof_count} files"
                )

                # Show rocprofv3 files
                for rocprof_file in self.summary.get("att_files", []):
                    console.print(f"  • {rocprof_file}")

                # Show performance counter info if available
                if self.summary.get("pmc_counters"):
                    console.print(
                        f"\n[cyan]Performance counters collected:[/cyan] {self.summary.get('pmc_counters')}"
                    )
                    console.print("  • counter_collection.csv: Performance counter data")

                # Usage instructions
                console.print("\n[cyan]Analysis tools:[/cyan]")
                console.print("  • Open CSV files for detailed trace analysis")
                console.print("  • kernel_trace.csv: GPU kernel execution data")
                console.print("  • hip_api_trace.csv: HIP API call traces")
                console.print("  • memory_allocation_trace.csv: Memory operations")

            # Show NVIDIA profiling results
            if "profile_files" in self.summary:
                csv_count = len(self.summary.get("csv_files", []))

                console.print(f"\n[cyan]Profile files generated:[/cyan] {csv_count} CSV files")

                # Show CSV files (only output format)
                if csv_count > 0:
                    console.print("[cyan]GPU kernel trace (CSV):[/cyan]")
                    for csv_file in self.summary.get("csv_files", []):
                        console.print(f"  • {csv_file}")

                # Usage instructions
                console.print("\n[cyan]Analysis tools:[/cyan]")
                if csv_count > 0:
                    console.print("  • View CSV files for kernel execution details")
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

        # We'll use a separate state file for the new profiling system
        self.state = ProfilingState()

    def profile(
        self,
        vendor: str,
        target: str,
        pmc_counters: Optional[str] = None,
        gpu_type: Optional[str] = None,
    ) -> ProfilingResults:
        """
        Execute a complete profiling workflow.

        Args:
            vendor: Either "nvidia" or "amd"
            target: File path or command to profile
            pmc_counters: Comma-separated performance counters for AMD (optional)
            gpu_type: GPU type override - "h100" or "l40s" for NVIDIA (optional)

        Returns:
            ProfilingResults with profiling data and summary
        """
        start_time = time.time()

        # Map vendor to GPU type
        if vendor == "nvidia":
            # Default to H100, allow override to L40S
            resolved_gpu_type = f"nvidia-{gpu_type}" if gpu_type else "nvidia-h100"
        else:
            resolved_gpu_type = "amd-mi300x"

        try:
            # 1. Ensure droplet exists
            console.print(f"[cyan]Ensuring {vendor.upper()} droplet is ready...[/cyan]")
            droplet_info = self._ensure_droplet(resolved_gpu_type)

            # 2. Analyze the target
            target_info = self._analyze_target(target)

            # 3. Prepare the command
            if target_info.is_source_file and target_info.file_path:
                console.print(f"[cyan]Syncing {target_info.file_path.name}...[/cyan]")
                self._sync_file(droplet_info, target_info.file_path)
                command = self._build_command(vendor, target_info)
            else:
                command = target

            # 4. Run profiling
            console.print("[cyan]Running profiler...[/cyan]")
            profile_output = self._run_profiler(
                droplet_info, vendor, command, pmc_counters, target_info
            )

            # 5. Calculate cost
            elapsed_hours = (time.time() - start_time) / 3600
            hourly_rate = 4.89 if vendor == "nvidia" else 1.99
            cost_estimate = elapsed_hours * hourly_rate

            # 6. Update last activity
            self.state.update_activity(resolved_gpu_type)

            return ProfilingResults(
                success=True,
                output_dir=profile_output["output_dir"],
                stdout=profile_output["stdout"],
                stderr=profile_output["stderr"],
                summary=profile_output["summary"],
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

    def _ensure_droplet(self, gpu_type: str) -> Dict[str, Any]:
        """Ensure a droplet exists for the given GPU type."""
        # Check if we have an active droplet
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

    def _analyze_target(self, target: str) -> TargetInfo:
        """Analyze the target to determine if it's a file or command."""
        target_path = Path(target)
        extension = target_path.suffix.lower()

        # Determine compiler based on extension
        compiler_map = {
            ".cpp": "hipcc",
            ".hip": "hipcc",
            ".cu": "nvcc",
            ".c": "gcc",
            ".py": "python",
        }

        # Check if it's a source file by extension or if it exists as a file
        # This handles cases where chisel is called as a library with relative paths
        is_source_extension = extension in compiler_map
        file_exists = target_path.exists() and target_path.is_file()

        if file_exists or is_source_extension:
            return TargetInfo(
                raw_target=target,
                is_source_file=True,
                file_path=target_path,  # Don't resolve() for relative paths when called as library
                file_extension=extension,
                compiler=compiler_map.get(extension, "gcc"),
            )

        # It's a command
        return TargetInfo(raw_target=target, is_source_file=False)

    def _sync_file(self, droplet_info: Dict[str, Any], file_path: Path):
        """Sync a file to the droplet."""
        ssh_manager = SSHManager()

        # For the new system, we'll sync to /tmp for simplicity
        # Use the original path string to handle relative paths when called as library
        success = ssh_manager.sync(str(file_path), "/tmp/", droplet_info["gpu_type"])

        if not success:
            raise RuntimeError(
                f"Failed to sync {file_path}. Ensure the file exists and is accessible."
            )

    def _build_command(self, vendor: str, target_info: TargetInfo) -> str:
        """Build the compilation and execution command."""
        if not target_info.file_path:
            return target_info.raw_target

        remote_source = f"/tmp/{target_info.file_path.name}"
        binary_name = target_info.file_path.stem
        remote_binary = f"/tmp/{binary_name}"

        # Handle Python files - no compilation needed
        if target_info.compiler == "python":
            return f"python3 {remote_source}"

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
        self,
        droplet_info: Dict[str, Any],
        vendor: str,
        command: str,
        pmc_counters: Optional[str] = None,
        target_info: Optional[TargetInfo] = None,
    ) -> Dict[str, Any]:
        """Run the profiler on the droplet."""
        ssh_manager = SSHManager()

        # Create output directory with simple timestamp naming
        now = datetime.now()
        timestamp = now.strftime("%H%M%S-%Y%m%d")
        output_dir = Path(f"./{CHISEL_PROFILING_DIR_NAME}/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

        if vendor == "amd":
            try:
                return self._run_amd_profiler(droplet_info, command, output_dir, pmc_counters)
            except Exception as e:
                console.print(f"[yellow]AMD rocprofv3 profiling failed: {e}[/yellow]")
                console.print("[yellow]Falling back to legacy rocprof...[/yellow]")
                ssh_manager.profile(
                    command,
                    droplet_info["gpu_type"],
                    trace="hip,hsa",
                    output_dir=str(output_dir),
                    open_result=False,
                )
                summary = self._parse_amd_results(output_dir)
                return {
                    "output_dir": output_dir,
                    "stdout": "",
                    "stderr": "",
                    "summary": summary,
                }
        else:
            try:
                return self._run_nvidia_profiler(droplet_info, command, output_dir)
            except Exception as e:
                console.print(f"[yellow]NVIDIA profiling failed: {e}[/yellow]")
                console.print("[yellow]Falling back to basic execution...[/yellow]")
                exit_code = ssh_manager.run(command, droplet_info["gpu_type"])
                return {
                    "output_dir": output_dir,
                    "stdout": f"Command executed with exit code: {exit_code}",
                    "stderr": str(e),
                    "summary": {},
                }

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

    def _run_nvidia_profiler(
        self, droplet_info: Dict[str, Any], command: str, output_dir: Path
    ) -> Dict[str, Any]:
        """Run NVIDIA profilers (nsight-compute + nsight-systems) on the droplet."""
        ssh_manager = SSHManager()

        # Ensure NVIDIA profilers are available
        self._ensure_nvidia_profilers(droplet_info)

        # If profiling a Python script, ensure PyTorch is available
        if command.startswith("python3"):
            self._ensure_pytorch(droplet_info)

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
            # Execute compilation first, then profile with both tools
            ncu_cmd = f"{compile_part} && ncu --set full --target-processes all --export {profile_filename}_%p.ncu-rep {execute_part}"
            nsys_cmd = f"nsys profile --output={profile_filename}.nsys-rep {execute_part}"
        else:
            # Just profile the single command with both tools
            ncu_cmd = f"ncu --set full --target-processes all --export {profile_filename}_%p.ncu-rep {command}"
            nsys_cmd = f"nsys profile --output={profile_filename}.nsys-rep {command}"

        # Run both profilers - ncu first (more likely to fail), then nsys
        full_cmd = f"{profile_setup} && {ncu_cmd} && {nsys_cmd}"

        console.print(f"[cyan]Running NVIDIA profilers (ncu + nsys): {command}[/cyan]")

        # Execute profiling command - try both profilers with graceful degradation
        exit_code = ssh_manager.run(full_cmd, droplet_info["gpu_type"])

        # If both profilers failed, try them individually for better error reporting
        if exit_code != 0:
            console.print("[yellow]Both profilers failed, trying individually...[/yellow]")

            # Try ncu alone
            ncu_only_cmd = f"{profile_setup} && {ncu_cmd}"
            ncu_exit = ssh_manager.run(ncu_only_cmd, droplet_info["gpu_type"])

            # Try nsys alone
            nsys_only_cmd = f"{profile_setup} && {nsys_cmd}"
            nsys_exit = ssh_manager.run(nsys_only_cmd, droplet_info["gpu_type"])

            if ncu_exit != 0 and nsys_exit != 0:
                raise RuntimeError(
                    f"Both NVIDIA profilers failed: ncu={ncu_exit}, nsys={nsys_exit}"
                )
            elif ncu_exit != 0:
                console.print("[yellow]ncu profiling failed, but nsys succeeded[/yellow]")
            elif nsys_exit != 0:
                console.print("[yellow]nsys profiling failed, but ncu succeeded[/yellow]")

        # Download results without parsing - let users analyze .ncu-rep files with proper tools
        profile_files = self._download_nvidia_results(droplet_info, remote_profile_dir, output_dir)

        # Create basic summary with CSV files only
        csv_files = [f for f in profile_files if f.endswith(".csv")]

        summary = {
            "profile_files": profile_files,
            "csv_files": csv_files,
            "message": f"NVIDIA profiling completed. Generated {len(csv_files)} CSV files.",
        }

        # Cleanup remote files
        self._cleanup_nvidia_remote(droplet_info, remote_profile_dir)

        return {
            "output_dir": output_dir,
            "stdout": "NVIDIA profiling completed successfully",
            "stderr": "",
            "summary": summary,
        }

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

    def _run_amd_profiler(
        self,
        droplet_info: Dict[str, Any],
        command: str,
        output_dir: Path,
        pmc_counters: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run AMD rocprofv3 profiler with ATT traces on the droplet."""
        ssh_manager = SSHManager()

        # Ensure rocprofv3 is available
        self._ensure_rocprofv3(droplet_info)

        # Setup remote profiling environment
        remote_profile_dir = "/tmp/chisel_amd_profile"
        profile_dirname = f"att_trace_{int(time.time())}"

        # Build rocprofv3 profiling command
        profile_setup = f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir} && cd {remote_profile_dir}"

        # Build rocprofv3 command with optional PMC counters
        pmc_args = ""
        if pmc_counters:
            # Validate and format counters
            counters = [c.strip() for c in pmc_counters.split(",")]
            if len(counters) > 8:
                console.print(
                    f"[yellow]Warning: {len(counters)} counters requested, but hardware typically supports max 7-8. Some may fail.[/yellow]"
                )
            pmc_args = f"--pmc {','.join(counters)}"

        # For AMD, we need to separate compilation from profiling
        if " && " in command:
            # Split compilation and execution
            compile_part, execute_part = command.split(" && ", 1)
            # Execute compilation first, then profile with rocprofv3
            rocprof_cmd = f"{compile_part} && export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ && rocprofv3 --sys-trace --stats {pmc_args} -d {profile_dirname} -- {execute_part}"
        else:
            # Just profile the single command
            rocprof_cmd = f"export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ && rocprofv3 --sys-trace --stats {pmc_args} -d {profile_dirname} -- {command}"

        # Full profiling command
        full_cmd = f"{profile_setup} && {rocprof_cmd}"

        console.print(f"[cyan]Running AMD rocprofv3 with ATT traces: {command}[/cyan]")

        # Execute profiling command
        exit_code = ssh_manager.run(full_cmd, droplet_info["gpu_type"])

        if exit_code != 0:
            raise RuntimeError(f"rocprofv3 profiling failed with exit code {exit_code}")

        # Download results
        rocprof_files = self._download_amd_att_results(
            droplet_info, remote_profile_dir, profile_dirname, output_dir
        )

        # Create summary
        summary = {
            "att_files": rocprof_files,  # Keep same key for display compatibility
            "profile_type": "rocprofv3",
            "message": f"AMD rocprofv3 profiling completed. Generated {len(rocprof_files)} output files.",
            "pmc_counters": pmc_counters,  # Include counter info for display
        }

        # Cleanup remote files
        self._cleanup_amd_remote(droplet_info, remote_profile_dir)

        return {
            "output_dir": output_dir,
            "stdout": "AMD rocprofv3 profiling completed successfully",
            "stderr": "",
            "summary": summary,
        }

    def _download_amd_att_results(
        self,
        droplet_info: Dict[str, Any],
        remote_dir: str,
        profile_dirname: str,
        local_output_dir: Path,
    ) -> list:
        import subprocess
        import tarfile

        ssh_manager = SSHManager()
        ip = droplet_info["ip"]
        console.print("[cyan]Filtering and downloading AMD profiling results...[/cyan]")
        filter_cmd = f"""cd {remote_dir} && \
reports=(kernel_stats memory_copy_stats)\n\nfor csv in $(find {profile_dirname} -type f -name '*.csv'); do\n    keep=false\n    for r in \"${{reports[@]}}\"; do\n        [[ \"$csv\" == *\"${{r}}.csv\" ]] && keep=true\n    done\n    $keep || rm -f \"$csv\"\ndone"""
        ssh_manager.run(filter_cmd, droplet_info["gpu_type"])
        archive_cmd = f"cd {remote_dir} && tar -czf amd_att_profile.tgz {profile_dirname}/ 2>/dev/null || echo 'No CSV files found'"
        exit_code = ssh_manager.run(archive_cmd, droplet_info["gpu_type"])
        if exit_code != 0:
            console.print(
                "[yellow]Warning: No ATT trace files found or archive creation failed[/yellow]"
            )
            return []
        local_archive_path = local_output_dir / "amd_att_profile.tgz"
        scp_cmd = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            f"root@{ip}:{remote_dir}/amd_att_profile.tgz",
            str(local_archive_path),
        ]
        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                console.print(
                    f"[yellow]Warning: Failed to download AMD ATT results: {result.stderr}[/yellow]"
                )
                return []
            if not local_archive_path.exists() or local_archive_path.stat().st_size == 0:
                console.print("[yellow]Warning: Downloaded archive is empty or missing[/yellow]")
                return []
            # Extract archive into a subdirectory as before
            amd_results_dir = local_output_dir / "amd_att_profile"
            amd_results_dir.mkdir(exist_ok=True)
            with tarfile.open(local_archive_path, "r:gz") as tar:
                tar.extractall(amd_results_dir)
            csv_files = list(amd_results_dir.rglob("*_kernel_stats.csv")) + list(
                amd_results_dir.rglob("*_memory_copy_stats.csv")
            )
            if not csv_files:
                console.print(
                    "[yellow]Warning: No essential CSV files found in extracted archive[/yellow]"
                )
                return []
            else:
                console.print(
                    f"[green]✓ AMD profiling results saved to {amd_results_dir} ({len(csv_files)} essential CSV files)[/green]"
                )
                return [str(f.relative_to(amd_results_dir)) for f in csv_files]
        except subprocess.TimeoutExpired:
            console.print("[yellow]Warning: Download timed out[/yellow]")
            return []
        except tarfile.TarError as e:
            console.print(f"[yellow]Warning: Failed to extract archive: {e}[/yellow]")
            return []
        except Exception as e:
            console.print(f"[yellow]Warning: Unexpected error during download: {e}[/yellow]")
            return []
        finally:
            if local_archive_path.exists():
                local_archive_path.unlink()

    def _cleanup_amd_remote(self, droplet_info: Dict[str, Any], remote_dir: str):
        """Clean up remote AMD profiling files."""
        ssh_manager = SSHManager()

        cleanup_cmd = f"rm -rf {remote_dir}"
        ssh_manager.run(cleanup_cmd, droplet_info["gpu_type"])

        console.print("[green]✓ Remote cleanup completed[/green]")

    def _download_nvidia_results(
        self, droplet_info: Dict[str, Any], remote_dir: str, local_output_dir: Path
    ) -> list:
        import subprocess
        import tarfile

        ssh_manager = SSHManager()
        ip = droplet_info["ip"]
        console.print("[cyan]Converting profiles to CSV format...[/cyan]")
        convert_cmd = f"""cd {remote_dir} && \
reports=(cuda_gpu_kern_sum cuda_gpu_mem_time_sum)\n\nfor nsys_file in *.nsys-rep; do\n    [ -f \"$nsys_file\" ] || continue\n    base=${{nsys_file%.nsys-rep}}\n    for rep in \"${{reports[@]}}\"; do\n        nsys stats -r \"$rep\" --format csv \"$nsys_file\" \
            > \"${{base}}_${{rep}}.csv\" 2>/dev/null || true\n    done\ndone"""
        ssh_manager.run(convert_cmd, droplet_info["gpu_type"])
        console.print("[cyan]Downloading NVIDIA profiling results...[/cyan]")
        archive_cmd = f"cd {remote_dir} && tar -czf nvidia_profile.tgz *.csv 2>/dev/null || echo 'No CSV files found'"
        exit_code = ssh_manager.run(archive_cmd, droplet_info["gpu_type"])
        if exit_code != 0:
            console.print("[yellow]Warning: No CSV files found or archive creation failed[/yellow]")
            return []
        local_archive_path = local_output_dir / "nvidia_profile.tgz"
        scp_cmd = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            f"root@{ip}:{remote_dir}/nvidia_profile.tgz",
            str(local_archive_path),
        ]
        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                console.print(
                    f"[yellow]Warning: Failed to download NVIDIA profile results: {result.stderr}[/yellow]"
                )
                return []
            if not local_archive_path.exists() or local_archive_path.stat().st_size == 0:
                console.print("[yellow]Warning: Downloaded archive is empty or missing[/yellow]")
                return []
            # Extract archive into a subdirectory as before
            nvidia_results_dir = local_output_dir / "nvidia_profile"
            nvidia_results_dir.mkdir(exist_ok=True)
            with tarfile.open(local_archive_path, "r:gz") as tar:
                tar.extractall(nvidia_results_dir)
            csv_files = list(nvidia_results_dir.glob("*.csv"))
            if not csv_files:
                console.print("[yellow]Warning: No CSV files found in extracted archive[/yellow]")
                return []
            csv_file_names = [f.name for f in csv_files]
            console.print(
                f"[green]✓ NVIDIA profile results saved to {nvidia_results_dir} ({len(csv_files)} CSV files)[/green]"
            )
            return csv_file_names
        except subprocess.TimeoutExpired:
            console.print("[yellow]Warning: Download timed out[/yellow]")
            return []
        except tarfile.TarError as e:
            console.print(f"[yellow]Warning: Failed to extract archive: {e}[/yellow]")
            return []
        except Exception as e:
            console.print(f"[yellow]Warning: Unexpected error during download: {e}[/yellow]")
            return []
        finally:
            if local_archive_path.exists():
                local_archive_path.unlink()

    def _cleanup_nvidia_remote(self, droplet_info: Dict[str, Any], remote_dir: str):
        """Clean up remote NVIDIA profiling files."""
        ssh_manager = SSHManager()

        cleanup_cmd = f"rm -rf {remote_dir}"
        ssh_manager.run(cleanup_cmd, droplet_info["gpu_type"])

        console.print("[green]✓ Remote cleanup completed[/green]")
