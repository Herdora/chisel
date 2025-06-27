"""Profile manager for orchestrating GPU profiling workflows."""

# TODO: Have the name of profile output be <target>-<vendor>-<gpu>-<time>-<date>

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from chisel.core.droplet_service import DropletService, Droplet
from .types.gpu_profiles import GPUType

console = Console()

CHISEL_PROFILING_DIR_NAME = "chisel-results"
ROCPROFV3_DIR_NAME = "chisel-rocprofv3"
NSYS_DIR_NAME = "chisel-nsys"
NCOMPUTE_DIR_NAME = "chisel-ncompute"
MNT_SHARE_DIR = "/mnt/share"


@dataclass
class TargetInfo:
    """Information about the profiling target."""

    raw_target: str
    is_source_file: bool
    file_path: Optional[Path] = None
    file_extension: Optional[str] = None
    compiler: Optional[str] = None


@dataclass
class ProfilerResult:
    """Result from an individual profiler run."""

    local_output_dir: Path
    stdout: str
    stderr: str
    profile_files: List[str]
    summary_file: Optional[str]
    profile_type: str
    message: str


@dataclass
class ProfilingResults:
    """Result of a profiling operation."""

    success: bool
    output_dir: Path
    stdout: str
    stderr: str
    summary: Dict[str, Any]

    def display_summary(self):
        """Display a summary of the profiling results."""
        if self.success:
            console.print("\n[green]✓ Profiling completed successfully[/green]")
            console.print(f"[cyan]Results saved to:[/cyan] {self.output_dir}")

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

    def __init__(self, digital_ocean_token: Optional[str] = None):
        if not digital_ocean_token:
            raise RuntimeError("No API token configured. Run 'chisel configure' first.")

        self.droplet_service = DropletService(digital_ocean_token)

    def profile(
        self,
        command_to_profile: str,
        gpu_type: GPUType,
        files_to_sync: List[str] = [],
        output_dir: Path = Path("./chisel-results"),
        rocprofv3_flag: Optional[str] = None,
        rocprof_compute_flag: Optional[str] = None,
        nsys_flag: Optional[str] = None,
        ncompute_flag: Optional[str] = None,
    ) -> ProfilingResults:
        """
        Execute a complete profiling workflow.

        Args:
            command_to_profile: The command that will be profiled on the remote server.
            gpu_type: The GPU type to use for profiling.
            files_to_sync: List of files to sync to the remote server.
            output_dir: Custom output directory for results.
            rocprofv3_flag: Full command to run with rocprofv3 (AMD).
            rocprof_compute_flag: Full command to run with rocprof-compute (AMD)
            nsys_flag: Full command to run with nsys (NVIDIA)
            ncompute_flag: Full command to run with ncu (NVIDIA)

        Returns:
            ProfilingResults with profiling data and summary
        """

        try:
            console.print(
                f"[cyan]Starting profiling for {command_to_profile} on {gpu_type.value}[/cyan]"
            )
            droplet_info = self.droplet_service.get_or_create_droplet_by_type(gpu_type)

            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                console.print(
                    f"[yellow]Overwriting existing output directory: {output_dir}[/yellow]"
                )
                output_dir.rmdir()
                output_dir.mkdir(parents=True, exist_ok=True)

            all_results = []
            if rocprofv3_flag:
                result = self.run_rocprofv3(
                    command_to_profile, gpu_type, files_to_sync, output_dir, rocprofv3_flag
                )
                all_results.append(result)
            if rocprof_compute_flag:
                result = self.run_rocprof_compute(
                    command_to_profile,
                    gpu_type,
                    files_to_sync,
                    output_dir,
                    rocprof_compute_flag,
                )
                all_results.append(result)
            if nsys_flag:
                result = self.run_nsys(
                    command_to_profile, gpu_type, files_to_sync, output_dir, nsys_flag
                )
                all_results.append(result)
            if ncompute_flag:
                result = self.run_ncompute(
                    command_to_profile, gpu_type, files_to_sync, output_dir, ncompute_flag
                )
                all_results.append(result)

            return ProfilingResults(
                success=True,
                output_dir=output_dir,
                stdout="",
                stderr="",
                summary={
                    "profile_files": [result.local_output_dir for result in all_results],
                    "summary_file": all_results[0].summary_file if all_results else None,
                    "profile_type": all_results[0].profile_type if all_results else "unknown",
                    "message": "Profiling completed. Generated profile data.",
                },
            )

        except Exception as e:
            console.print(f"[red]Error during profiling: {e}[/red]")
            return ProfilingResults(
                success=False,
                output_dir=Path(f"./{CHISEL_PROFILING_DIR_NAME}/failed"),
                stdout="",
                stderr=str(e),
                summary={},
            )

    def run_rocprofv3(
        self,
        command_to_profile: str,
        gpu_type: GPUType,
        files_to_sync: List[str],
        output_dir: Path,
        rocprofv3_flags: str,
    ) -> ProfilerResult:
        """Run rocprofv3 on the droplet."""
        droplet_with_gpu: Droplet = self.droplet_service.get_or_create_droplet_by_type(gpu_type)
        self._ensure_rocprofv3(droplet_with_gpu)

        remote_profile_dir = f"{MNT_SHARE_DIR}/{ROCPROFV3_DIR_NAME}"

        for file in files_to_sync:
            self._sync_file(droplet_with_gpu, Path(file), remote_profile_dir)
            if file.endswith(".py"):
                self._ensure_pytorch_rocm(droplet_with_gpu)

        RESET_CMD = f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"
        result = droplet_with_gpu.run_container_command(RESET_CMD)
        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to reset remote directory: {result['exit_code']}")

        CD_CMD = f"cd {remote_profile_dir}"
        PROFILE_CMD = f"rocprofv3 -S --summary-output-file amd_profile_summary.txt {rocprofv3_flags} -- {command_to_profile}"
        FULL_CMD = f"{CD_CMD} && {PROFILE_CMD}"
        console.print(f"[cyan]Running AMD rocprofv3 with command: {FULL_CMD}[/cyan]")
        rocprof_result = droplet_with_gpu.run_container_command(FULL_CMD, timeout=600)
        if rocprof_result["exit_code"] != 0:
            raise RuntimeError(
                f"rocprofv3 profiling failed with exit code {rocprof_result['exit_code']}"
            )

        rocprof_files = self._download_results(
            droplet_with_gpu, remote_profile_dir, output_dir
        )  # TODO: Make this 'feel' better
        self._cleanup_amd_remote(droplet_with_gpu, remote_profile_dir)

        return ProfilerResult(
            local_output_dir=output_dir,
            stdout="AMD rocprofv3 profiling completed successfully",
            stderr="",
            profile_files=rocprof_files,
            summary_file=rocprof_files[0] if rocprof_files else None,
            profile_type="rocprofv3",
            message="AMD rocprofv3 profiling completed. Generated profile summary.",
        )

    def run_rocprof_compute(
        self,
        command_to_profile: str,
        gpu_type: GPUType,
        files_to_sync: List[str],
        output_dir: Path,
        rocprof_compute_flags: str,
    ) -> ProfilerResult:
        """Run rocprof-compute on the droplet."""
        # TODO: Implement rocprof-compute when ready

        console.print("[yellow]rocprof-compute support not yet implemented[/yellow]")
        raise RuntimeError("rocprof-compute is not yet supported")

    def run_nsys(
        self,
        command_to_profile: str,
        gpu_type: GPUType,
        files_to_sync: List[str],
        output_dir: Path,
        nsys_flags: str,
    ) -> ProfilerResult:
        """Run nsys on the droplet."""
        droplet_with_gpu: Droplet = self.droplet_service.get_or_create_droplet_by_type(gpu_type)
        self._ensure_nvidia_profilers(droplet_with_gpu)

        remote_profile_dir = f"{MNT_SHARE_DIR}/{NSYS_DIR_NAME}"

        for file in files_to_sync:
            self._sync_file(droplet_with_gpu, Path(file), remote_profile_dir)
            if file.endswith(".py"):
                self._ensure_pytorch(droplet_with_gpu)

        RESET_CMD = f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"
        result = droplet_with_gpu.run_container_command(RESET_CMD)
        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to reset remote directory: {result['exit_code']}")

        CD_CMD = f"cd {remote_profile_dir}"
        PROFILE_CMD = f"nsys profile {nsys_flags} -o nvidia_profile -- {command_to_profile}"
        FULL_CMD = f"{CD_CMD} && {PROFILE_CMD}"
        console.print(f"[cyan]Running NVIDIA nsys with command: {FULL_CMD}[/cyan]")
        nsys_result = droplet_with_gpu.run_container_command(FULL_CMD, timeout=600)
        if nsys_result["exit_code"] != 0:
            raise RuntimeError(f"nsys profiling failed with exit code {nsys_result['exit_code']}")

        nvidia_files = self._download_results(droplet_with_gpu, remote_profile_dir, output_dir)
        self._cleanup_nvidia_remote(droplet_with_gpu, remote_profile_dir)

        return ProfilerResult(
            local_output_dir=output_dir,
            stdout="NVIDIA nsys profiling completed successfully",
            stderr="",
            profile_files=nvidia_files,
            summary_file=nvidia_files[0] if nvidia_files else None,
            profile_type="nsys",
            message="NVIDIA nsys profiling completed. Generated profile data.",
        )

    def run_ncompute(
        self,
        command_to_profile: str,
        gpu_type: GPUType,
        files_to_sync: List[str],
        output_dir: Path,
        ncompute_flags: str,
    ) -> ProfilerResult:
        """Run ncu (nsight-compute) on the droplet."""
        droplet_with_gpu: Droplet = self.droplet_service.get_or_create_droplet_by_type(gpu_type)
        self._ensure_nvidia_profilers(droplet_with_gpu)

        remote_profile_dir = f"{MNT_SHARE_DIR}/{NCOMPUTE_DIR_NAME}"

        for file in files_to_sync:
            self._sync_file(droplet_with_gpu, Path(file), remote_profile_dir)
            if file.endswith(".py"):
                self._ensure_pytorch(droplet_with_gpu)

        RESET_CMD = f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"
        result = droplet_with_gpu.run_container_command(RESET_CMD)
        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to reset remote directory: {result['exit_code']}")

        CD_CMD = f"cd {remote_profile_dir}"
        PROFILE_CMD = f"ncu {ncompute_flags} -o nvidia_ncompute_profile -- {command_to_profile}"
        FULL_CMD = f"{CD_CMD} && {PROFILE_CMD}"
        console.print(f"[cyan]Running NVIDIA ncu with command: {FULL_CMD}[/cyan]")
        ncu_result = droplet_with_gpu.run_container_command(FULL_CMD, timeout=600)
        if ncu_result["exit_code"] != 0:
            raise RuntimeError(f"ncu profiling failed with exit code {ncu_result['exit_code']}")

        nvidia_files = self._download_results(droplet_with_gpu, remote_profile_dir, output_dir)
        self._cleanup_nvidia_remote(droplet_with_gpu, remote_profile_dir)

        return ProfilerResult(
            local_output_dir=output_dir,
            stdout="NVIDIA ncu profiling completed successfully",
            stderr="",
            profile_files=nvidia_files,
            summary_file=nvidia_files[0] if nvidia_files else None,
            profile_type="ncompute",
            message="NVIDIA ncu profiling completed. Generated profile data.",
        )

    def _get_target_info(self, target: str) -> TargetInfo:
        """Analyze the target to determine if it's a file or command."""
        target_path = Path(target)
        extension = target_path.suffix.lower()

        compiler_map = {
            ".cpp": "hipcc",
            ".hip": "hipcc",
            ".cu": "nvcc",
            ".c": "gcc",
            ".py": "python3",
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

    def _sync_file(self, droplet_info: Droplet, source_file: Path, remote_dir: str):
        """Sync a file to the droplet with proper temp directory setup."""
        success = droplet_info.sync_file(str(source_file), f"{remote_dir}/")
        if not success:
            raise RuntimeError(
                f"Failed to sync {source_file} to {remote_dir}. Ensure the file exists and is accessible."
            )

        # Make file executable inside the container (not on host)
        chmod_cmd = f"chmod +x {remote_dir}/{source_file.name}"
        result = droplet_info.run_container_command(chmod_cmd)
        if result["exit_code"] != 0:
            console.print("[yellow]Warning: Failed to make file executable in container[/yellow]")
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

    def _ensure_nvidia_profilers(self, droplet_info: Droplet):
        """Ensure both nsight-compute and nsight-systems are installed on the droplet."""
        try:
            # First check if the container exists and is working
            check_cmd = "which ncu && ncu --version && which nsys && nsys --version"
            result = droplet_info.run_container_command(check_cmd)

            if result["exit_code"] == 0:
                console.print("[green]✓ NVIDIA profilers (ncu + nsys) already available[/green]")
                return

            # If container command failed, check if it's a container issue
            console.print("[yellow]Profilers not found, checking container status...[/yellow]")

            # Debug the container status
            self.debug_droplet_container_status(droplet_info)

            # Try to fix the container
            if not self.fix_droplet_container(droplet_info):
                raise RuntimeError("Failed to fix droplet container setup")

            # Now try installing profilers in the working container
            console.print(
                "[yellow]Installing NVIDIA profilers (nsight-compute + nsight-systems)...[/yellow]"
            )

            # First, try installing from snap (most reliable)
            snap_install_cmd = """
            timeout 900 bash -c '
            apt-get update -y && 
            apt-get install -y snapd && 
            snap install nsight-compute nsight-systems && 
            ln -sf /snap/nsight-compute/current/bin/ncu /usr/local/bin/ncu && 
            ln -sf /snap/nsight-systems/current/bin/nsys /usr/local/bin/nsys &&
            echo "✓ Installed via snap"
            '
            """

            console.print("[cyan]Trying snap installation...[/cyan]")
            snap_result = droplet_info.run_container_command(snap_install_cmd, timeout=1000)

            if snap_result["exit_code"] == 0:
                console.print("[green]✓ NVIDIA profilers installed via snap[/green]")
            else:
                console.print(
                    "[yellow]Snap installation failed, trying alternative methods...[/yellow]"
                )

                # Try installing from NVIDIA's CUDA repositories
                cuda_repo_install_cmd = """
                timeout 900 bash -c '
                apt-get update -y && 
                apt-get install -y wget gnupg && 
                wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - && 
                echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && 
                apt-get update -y && 
                apt-get install -y nsight-compute nsight-systems-cli && 
                echo "✓ Installed via NVIDIA CUDA repository"
                '
                """

                console.print("[cyan]Trying NVIDIA CUDA repository installation...[/cyan]")
                cuda_result = droplet_info.run_container_command(
                    cuda_repo_install_cmd, timeout=1000
                )

                if cuda_result["exit_code"] != 0:
                    console.print(
                        "[yellow]CUDA repository installation failed, trying direct download...[/yellow]"
                    )

                    # Last resort: Download and install manually
                    direct_install_cmd = """
                    timeout 1200 bash -c '
                    cd /tmp && 
                    wget -q https://developer.nvidia.com/downloads/assets/tools/secure/nsight-compute/2023_3_1/nsight-compute-linux-2023.3.1.4-33567449.run && 
                    wget -q https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2023_4_1/nsight-systems-linux-public-2023.4.1.97-33513133.run && 
                    chmod +x nsight-compute-linux-*.run && 
                    chmod +x nsight-systems-linux-*.run && 
                    ./nsight-compute-linux-*.run --silent --toolkit --installpath=/opt/nvidia/nsight-compute && 
                    ./nsight-systems-linux-*.run --silent --accept && 
                    ln -sf /opt/nvidia/nsight-compute/ncu /usr/local/bin/ncu && 
                    ln -sf /opt/nvidia/nsight-systems/bin/nsys /usr/local/bin/nsys && 
                    echo "✓ Installed via direct download"
                    '
                    """

                    console.print("[cyan]Trying direct download installation...[/cyan]")
                    direct_result = droplet_info.run_container_command(
                        direct_install_cmd, timeout=1300
                    )

                    if direct_result["exit_code"] != 0:
                        # Show detailed error information for debugging
                        console.print(
                            "[yellow]All installation methods failed. Checking what's available...[/yellow]"
                        )

                        debug_cmd = """
                        echo "=== Available packages ===" && 
                        apt search nsight 2>/dev/null | head -10 && 
                        echo "=== CUDA toolkit ===" && 
                        ls -la /usr/local/cuda*/bin/ncu* 2>/dev/null || echo "No CUDA toolkit found" && 
                        echo "=== System info ===" && 
                        lsb_release -a && 
                        uname -a
                        """

                        debug_result = droplet_info.run_container_command(debug_cmd)
                        console.print("[cyan]Debug information:[/cyan]")
                        console.print(debug_result.get("stdout", "No debug output"))
                        console.print("[red]Debug stderr:[/red]")
                        console.print(debug_result.get("stderr", "No debug stderr"))

                        raise RuntimeError(
                            "Failed to install NVIDIA profilers using all available methods. "
                            "The droplet may not have the required NVIDIA drivers or repositories configured."
                        )

            # Verify both installations work
            verify_ncu_result = droplet_info.run_container_command("which ncu && ncu --version")
            verify_nsys_result = droplet_info.run_container_command("which nsys && nsys --version")

            if verify_ncu_result["exit_code"] != 0:
                raise RuntimeError(
                    "nsight-compute installation verification failed. The ncu command is not available after installation."
                )

            if verify_nsys_result["exit_code"] != 0:
                raise RuntimeError(
                    "nsight-systems installation verification failed. The nsys command is not available after installation."
                )

            console.print(
                "[green]✓ NVIDIA profilers installed and verified successfully (ncu + nsys)[/green]"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to setup NVIDIA profilers: {e}")

    def _ensure_pytorch(self, droplet_info: Droplet):
        """Check that PyTorch with CUDA support is available (should be pre-installed in container)."""

        try:
            # Check if PyTorch is available (should be pre-installed in Docker container)
            check_cmd = "python -c \"import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')\""
            result = droplet_info.run_container_command(check_cmd)

            if result["exit_code"] == 0:
                console.print("[green]✓ PyTorch with CUDA already available in container[/green]")
                console.print(f"[cyan]PyTorch info: {result['stdout'].strip()}[/cyan]")
                return

            console.print(
                "[yellow]PyTorch not detected in container - checking if container needs restart[/yellow]"
            )

            # Try restarting the container in case it's not running
            restart_cmd = "docker restart ml && sleep 5"
            droplet_info.run_command(restart_cmd, timeout=30)

            # Try again
            result = droplet_info.run_container_command(check_cmd)
            if result["exit_code"] == 0:
                console.print("[green]✓ PyTorch available after container restart[/green]")
                console.print(f"[cyan]PyTorch info: {result['stdout'].strip()}[/cyan]")
                return

            console.print("[red]Warning: PyTorch not available in container[/red]")
            console.print(
                "[yellow]Container may still be starting up - profiling may work anyway[/yellow]"
            )

        except Exception as e:
            console.print(f"[yellow]Warning: Could not verify PyTorch: {e}[/yellow]")
            console.print("[yellow]Continuing anyway - container may still be starting[/yellow]")

    def _ensure_rocprofv3(self, droplet_info: Droplet):
        """Ensure rocprofv3 and dependencies are installed on the AMD droplet."""
        try:
            # Check if rocprofv3 is already available
            check_cmd = "which rocprofv3 && echo 'rocprofv3 available'"
            result = droplet_info.run_container_command(check_cmd)

            if result["exit_code"] == 0:
                console.print("[green]✓ rocprofv3 already available[/green]")
                return

            # If container command failed, check if it's a container issue
            console.print("[yellow]rocprofv3 not found, checking container status...[/yellow]")

            # Debug the container status
            self.debug_droplet_container_status(droplet_info)

            # Try to fix the container
            if not self.fix_droplet_container(droplet_info):
                raise RuntimeError("Failed to fix droplet container setup")

            # First validate that ROCm is properly installed
            self._validate_rocm_installation(droplet_info)

            console.print("[yellow]Installing rocprofv3 and dependencies...[/yellow]")

            # Install build dependencies and build tools with verbose output
            setup_cmd = """
            timeout 1800 bash -c '
            apt-get update -y && 
            apt-get install -y git cmake build-essential python3 python3-pip wget && \\
            echo "✓ Build dependencies installed"
            '
            """

            console.print("[cyan]Installing build dependencies...[/cyan]")
            setup_result = droplet_info.run_container_command(setup_cmd, timeout=1900)
            if setup_result["exit_code"] != 0:
                console.print(
                    f"[red]Failed to install build dependencies: {setup_result.get('stderr', '')}[/red]"
                )
                raise RuntimeError("Failed to install build dependencies")

            # Build aqlprofile from mainline
            build_aqlprofile_cmd = """
            cd /tmp && 
            git clone https://github.com/ROCm/aqlprofile.git && 
            cd aqlprofile && 
            mkdir build && cd build && 
            cmake .. && make -j$(nproc) && make install && \\
            echo "✓ aqlprofile built and installed"
            """

            console.print("[cyan]Building aqlprofile...[/cyan]")
            aql_result = droplet_info.run_container_command(build_aqlprofile_cmd, timeout=1200)
            if aql_result["exit_code"] != 0:
                console.print(
                    f"[red]Failed to build aqlprofile: {aql_result.get('stderr', '')}[/red]"
                )
                raise RuntimeError("Failed to build aqlprofile")

            # Build rocprofiler-sdk from mainline
            build_rocprofiler_cmd = """
            cd /tmp && 
            git clone https://github.com/ROCm/rocprofiler-sdk.git && 
            cd rocprofiler-sdk && 
            mkdir build && cd build && 
            cmake .. && make -j$(nproc) && make install && \\
            echo "✓ rocprofiler-sdk built and installed"
            """

            console.print("[cyan]Building rocprofiler-sdk...[/cyan]")
            profiler_result = droplet_info.run_container_command(
                build_rocprofiler_cmd, timeout=1200
            )
            if profiler_result["exit_code"] != 0:
                console.print(
                    f"[red]Failed to build rocprofiler-sdk: {profiler_result.get('stderr', '')}[/red]"
                )
                raise RuntimeError("Failed to build rocprofiler-sdk")

            # Download rocprof-trace-decoder binary
            download_decoder_cmd = """
            cd /tmp && 
            wget -O /opt/rocm/lib/rocprof-trace-decoder https://github.com/ROCm/rocprof-trace-decoder/releases/latest/download/rocprof-trace-decoder && 
            chmod +x /opt/rocm/lib/rocprof-trace-decoder &&
            ln -sf /opt/rocm/lib/rocprof-trace-decoder /opt/rocm/lib/libatt_decoder_trace.so && \\
            echo "✓ rocprof-trace-decoder installed"
            """

            console.print("[cyan]Installing rocprof-trace-decoder...[/cyan]")
            decoder_result = droplet_info.run_container_command(download_decoder_cmd, timeout=300)
            if decoder_result["exit_code"] != 0:
                console.print(
                    f"[red]Failed to install rocprof-trace-decoder: {decoder_result.get('stderr', '')}[/red]"
                )
                raise RuntimeError("Failed to install rocprof-trace-decoder")

            # Set up environment
            env_setup_cmd = """
            echo 'export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/' >> /root/.bashrc &&
            export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ && \\
            echo "✓ Environment variables set"
            """

            env_result = droplet_info.run_container_command(env_setup_cmd)
            if env_result["exit_code"] != 0:
                console.print(
                    f"[red]Failed to set up environment: {env_result.get('stderr', '')}[/red]"
                )
                raise RuntimeError("Failed to set up environment")

            # Verify installation with detailed output
            verify_cmd = "export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ && which rocprofv3 && rocprofv3 --help | head -10"
            verify_result = droplet_info.run_container_command(verify_cmd)

            if verify_result["exit_code"] != 0:
                console.print(
                    f"[red]rocprofv3 verification failed: {verify_result.get('stderr', '')}[/red]"
                )
                raise RuntimeError("rocprofv3 installation verification failed")

            console.print("[green]✓ rocprofv3 and dependencies installed successfully[/green]")
            console.print(f"[cyan]rocprofv3 help preview:[/cyan]")
            console.print(verify_result.get("stdout", "").strip())

        except Exception as e:
            raise RuntimeError(f"Failed to setup rocprofv3: {e}")

    def _validate_rocm_installation(self, droplet_info: Droplet):
        """Validate that ROCm is properly installed and working."""
        try:
            console.print("[cyan]Validating ROCm installation...[/cyan]")

            # Check if rocminfo is available and working
            rocminfo_cmd = "/opt/rocm/bin/rocminfo"
            result = droplet_info.run_container_command(rocminfo_cmd)

            if result["exit_code"] != 0:
                console.print(
                    "[yellow]rocminfo not found or failed, checking ROCm installation...[/yellow]"
                )

                # Check if ROCm directory exists
                rocm_check_cmd = "ls -la /opt/rocm/ && echo 'ROCm directory exists'"
                rocm_result = droplet_info.run_container_command(rocm_check_cmd)

                if rocm_result["exit_code"] != 0:
                    raise RuntimeError("ROCm installation not found at /opt/rocm/")

                # Try alternative rocminfo locations
                alt_rocminfo_cmd = "which rocminfo || find /opt -name rocminfo 2>/dev/null || echo 'rocminfo not found'"
                alt_result = droplet_info.run_container_command(alt_rocminfo_cmd)

                if "rocminfo not found" in alt_result.get("stdout", ""):
                    console.print(
                        "[yellow]Warning: rocminfo not available, but ROCm directory exists[/yellow]"
                    )
                    console.print("[yellow]This may still work for profiling[/yellow]")
                else:
                    console.print(
                        f"[green]Found rocminfo at: {alt_result.get('stdout', '').strip()}[/green]"
                    )
            else:
                # rocminfo worked, show some output
                console.print("[green]✓ ROCm installation validated[/green]")
                stdout = result.get("stdout", "")
                # Show first few lines of rocminfo output
                lines = stdout.split("\n")[:5]
                for line in lines:
                    if line.strip():
                        console.print(f"[cyan]  {line.strip()}[/cyan]")

                # Check for GPU devices
                if "GPU" in stdout or "gfx" in stdout:
                    console.print("[green]✓ GPU device(s) detected by ROCm[/green]")
                else:
                    console.print(
                        "[yellow]Warning: No GPU devices detected in rocminfo output[/yellow]"
                    )

        except Exception as e:
            console.print(f"[yellow]Warning: ROCm validation failed: {e}[/yellow]")
            console.print(
                "[yellow]Continuing anyway - this may still work in the container environment[/yellow]"
            )

    def _debug_rocm_packages(self, droplet_info: Droplet):
        """Show debug information about ROCm packages for troubleshooting."""
        try:
            console.print("[cyan]ROCm Package Debug Information:[/cyan]")

            # Check which ROCm packages are installed
            package_check_cmd = """
            echo "=== Installed ROCm packages ===" && \\
            dpkg -l | grep -i rocm | head -10 && \\
            echo "=== ROCm library files ===" && \\
            ls -la /opt/rocm/lib/ | head -10 && \\
            echo "=== ROCm binary files ===" && \\
            ls -la /opt/rocm/bin/ | head -10 && \\
            echo "=== Environment variables ===" && \\
            env | grep -i rocm
            """

            result = droplet_info.run_container_command(package_check_cmd)

            if result["exit_code"] == 0:
                console.print(f"[cyan]Debug output:[/cyan]")
                for line in result.get("stdout", "").split("\n"):
                    if line.strip():
                        console.print(f"  {line.strip()}")
            else:
                console.print("[yellow]Could not gather ROCm debug information[/yellow]")

        except Exception as e:
            console.print(f"[yellow]Debug information gathering failed: {e}[/yellow]")

    def _download_results(
        self,
        droplet_info: Droplet,
        remote_dir: str,
        local_output_dir: Path,
    ) -> list:
        import subprocess

        ip = droplet_info.ip
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

    def _cleanup_amd_remote(self, droplet_info: Droplet, remote_dir: str):
        """Clean up remote AMD profiling files."""
        cleanup_cmd = f"rm -rf {remote_dir}"
        droplet_info.run_container_command(cleanup_cmd)
        console.print("[green]✓ Remote cleanup completed[/green]")

    def _cleanup_nvidia_remote(self, droplet_info: Droplet, remote_dir: str):
        """Clean up remote NVIDIA profiling files."""
        cleanup_cmd = f"rm -rf {remote_dir}"
        droplet_info.run_container_command(cleanup_cmd)
        console.print("[green]✓ Remote cleanup completed[/green]")

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
                                    "total_time": float(row["TotalDurationNs"]) / 1_000_000,
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

    def _ensure_pytorch_rocm(self, droplet_info: Droplet):
        """Check that PyTorch with ROCm support is available (should be pre-installed in container)."""

        try:
            # Check if PyTorch is available (should be pre-installed in Docker container)
            check_cmd = "python -c \"import torch; print(f'PyTorch {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')\""
            result = droplet_info.run_container_command(check_cmd)

            if result["exit_code"] == 0:
                console.print("[green]✓ PyTorch with ROCm already available in container[/green]")
                console.print(f"[cyan]PyTorch info: {result['stdout'].strip()}[/cyan]")
                return

            console.print(
                "[yellow]PyTorch not detected in container - checking if container needs restart[/yellow]"
            )

            # Try restarting the container in case it's not running
            restart_cmd = "docker restart ml && sleep 5"
            droplet_info.run_command(restart_cmd, timeout=30)

            # Try again
            result = droplet_info.run_container_command(check_cmd)
            if result["exit_code"] == 0:
                console.print("[green]✓ PyTorch available after container restart[/green]")
                console.print(f"[cyan]PyTorch info: {result['stdout'].strip()}[/cyan]")
                return

            console.print(
                "[red]Warning: PyTorch not available in container[/red]"
            )  # TODO: fix this env issue
            console.print(
                "[yellow]Container may still be starting up - profiling may work anyway[/yellow]"
            )

        except Exception as e:
            console.print(f"[yellow]Warning: Could not verify PyTorch: {e}[/yellow]")
            console.print("[yellow]Continuing anyway - container may still be starting[/yellow]")

    def debug_droplet_container_status(self, droplet_info: Droplet):
        """Debug the Docker container status on the droplet."""
        try:
            console.print("[cyan]🔍 Debugging droplet container status...[/cyan]")

            # Check if Docker is running
            docker_status_cmd = "systemctl status docker --no-pager -l"
            result = droplet_info.run_command(docker_status_cmd)
            console.print("[cyan]Docker service status:[/cyan]")
            console.print(result.get("stdout", "No output"))

            # Check Docker containers
            containers_cmd = "docker ps -a"
            result = droplet_info.run_command(containers_cmd)
            console.print("[cyan]Docker containers:[/cyan]")
            console.print(result.get("stdout", "No containers"))

            # Check Docker images
            images_cmd = "docker images"
            result = droplet_info.run_command(images_cmd)
            console.print("[cyan]Docker images:[/cyan]")
            console.print(result.get("stdout", "No images"))

            # Check if ml container exists but is stopped
            ml_status_cmd = "docker inspect ml 2>/dev/null || echo 'Container ml does not exist'"
            result = droplet_info.run_command(ml_status_cmd)
            console.print("[cyan]ML container status:[/cyan]")
            console.print(result.get("stdout", "No ml container info"))

            # Check cloud-init logs
            cloud_init_cmd = "cloud-init status && tail -50 /var/log/cloud-init-output.log"
            result = droplet_info.run_command(cloud_init_cmd)
            console.print("[cyan]Cloud-init status and logs:[/cyan]")
            console.print(result.get("stdout", "No cloud-init logs"))

            # Check GPU devices
            gpu_cmd = "ls -la /dev/kfd /dev/dri/ 2>/dev/null || echo 'No GPU devices found'"
            result = droplet_info.run_command(gpu_cmd)
            console.print("[cyan]GPU devices:[/cyan]")
            console.print(result.get("stdout", "No GPU info"))

        except Exception as e:
            console.print(f"[red]Debug failed: {e}[/red]")

    def wait_for_cloud_init(self, droplet_info: Droplet, timeout: int = 600):
        """Wait for cloud-init to complete setup."""
        import time

        console.print("[cyan]⏳ Waiting for cloud-init to complete initial setup...[/cyan]")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check cloud-init status
            result = droplet_info.run_command("cloud-init status")
            status = result.get("stdout", "").strip()

            if "status: done" in status:
                console.print("[green]✓ Cloud-init setup completed[/green]")
                return True
            elif "status: error" in status:
                console.print(
                    "[yellow]⚠️ Cloud-init completed with errors, but continuing...[/yellow]"
                )
                return True
            elif "status: running" in status:
                console.print(
                    f"[cyan]Cloud-init still running... ({int(time.time() - start_time)}s elapsed)[/cyan]"
                )
                time.sleep(10)
            else:
                console.print(f"[yellow]Unknown cloud-init status: {status}[/yellow]")
                time.sleep(10)

        console.print("[yellow]⚠️ Cloud-init timeout, but continuing anyway...[/yellow]")
        return False

    def fix_droplet_container(self, droplet_info: Droplet):
        """Try to fix the droplet container setup."""
        try:
            console.print("[yellow]🔧 Attempting to fix droplet container setup...[/yellow]")

            # Wait for cloud-init to finish if it's still running
            self.wait_for_cloud_init(droplet_info)

            # First, try to start the existing ml container if it exists but is stopped
            start_existing_cmd = "docker start ml"
            result = droplet_info.run_command(start_existing_cmd)

            if result.get("exit_code") == 0:
                console.print("[green]✓ Started existing ml container[/green]")
                return True

            # If that fails, try to create the container from scratch
            console.print("[yellow]Creating new ml container...[/yellow]")

            # Pull the image if it doesn't exist
            pull_cmd = "docker pull rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0"
            result = droplet_info.run_command(pull_cmd, timeout=600)

            if result.get("exit_code") != 0:
                console.print("[red]Failed to pull Docker image[/red]")
                return False

            # Create and start the ml container
            create_container_cmd = """
            docker run -dit \\
              --name ml \\
              --restart=always \\
              --network host \\
              --ipc=host \\
              --device=/dev/kfd \\
              --device=/dev/dri \\
              --group-add video \\
              --cap-add=SYS_PTRACE \\
              --security-opt seccomp=unconfined \\
              -v /mnt/share:/workspace \\
              rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0 bash
            """

            result = droplet_info.run_command(create_container_cmd, timeout=120)

            if result.get("exit_code") != 0:
                console.print(
                    f"[red]Failed to create ml container: {result.get('stderr', '')}[/red]"
                )
                return False

            # Set up the container environment
            setup_cmd = """
            docker exec ml bash -c "
              apt-get update && \\
              apt-get install -y rocprofiler-dev roctracer-dev rocm-dev rocm-profiler wget gnupg build-essential || echo 'Some packages may not be available' && \\
              python -m venv /opt/venv && \\
              source /opt/venv/bin/activate && \\
              pip install --upgrade pip setuptools && \\
              echo 'source /opt/venv/bin/activate' >> /root/.bashrc && \\
              echo 'export ROCM_PATH=/opt/rocm' >> /root/.bashrc && \\
              echo 'export PATH=$PATH:/opt/rocm/bin' >> /root/.bashrc && \\
              echo 'export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/' >> /root/.bashrc && \\
              mkdir -p /mnt/share
            "
            """

            result = droplet_info.run_command(setup_cmd, timeout=300)

            if result.get("exit_code") == 0:
                console.print("[green]✓ Container setup completed successfully[/green]")
                return True
            else:
                console.print(
                    f"[yellow]Container setup had some issues but may still work: {result.get('stderr', '')}[/yellow]"
                )
                return True  # Continue anyway, basic container should work

        except Exception as e:
            console.print(f"[red]Failed to fix container: {e}[/red]")
            return False
