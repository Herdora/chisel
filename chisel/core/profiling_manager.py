"""Profile manager for orchestrating GPU profiling workflows."""

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from chisel.core.chisel_client import ChiselClient
from chisel.core.droplet_service import Droplet, DropletService

from .types.gpu_profiles import GPUType

console = Console()

CHISEL_PROFILING_DIR_NAME = "chisel-results"
ROCPROFV3_DIR_NAME = "chisel-rocprofv3"
NSYS_DIR_NAME = "chisel-nsys"
NCOMPUTE_DIR_NAME = "chisel-ncompute"
WORKSPACE_DIR = "/workspace"


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
    success: bool
    output_dir: Path
    stdout: str
    stderr: str
    summary: Dict[str, Any]

    def display_summary(self):
        if self.success:
            console.print("\n[green]✓ Profiling completed successfully[/green]")
            console.print(f"[cyan]Results saved to:[/cyan] {self.output_dir}")

            if "top_kernels" in self.summary:
                console.print("\n[cyan]Top GPU Kernels:[/cyan]")
                for i, kernel in enumerate(self.summary["top_kernels"][:5], 1):
                    console.print(
                        f"  {i}. {kernel['name'][:50]:<50} {kernel['time_ms']:8.3f} ms"
                    )

            if "profile_files" in self.summary:
                summary_file = self.summary.get("summary_file")
                profile_type = self.summary.get("profile_type", "nvidia")

                if summary_file:
                    vendor_name = (
                        "AMD rocprofv3"
                        if profile_type == "rocprofv3"
                        else "NVIDIA"
                    )
                    console.print(
                        f"\n[cyan]{vendor_name} profile summary generated:[/cyan] {summary_file}"
                    )

                    console.print("\n[cyan]Analysis tools:[/cyan]")
                    console.print(
                        "  • View text summary for human-readable kernel analysis"
                    )
                else:
                    console.print(
                        "\n[cyan]Profile files generated:[/cyan] 0 files"
                    )
        else:
            console.print("\n[red]✗ Profiling failed[/red]")
            if self.stderr:
                console.print(f"[red]Error:[/red] {self.stderr}")


class ProfilingManager:
    def __init__(
        self,
        digital_ocean_token: Optional[str] = None,
        chisel_client: Optional[ChiselClient] = None,
    ):
        if not digital_ocean_token:
            raise RuntimeError(
                "No API token configured. Run 'chisel configure' first."
            )

        self.droplet_service = DropletService(
            digital_ocean_token, chisel_client
        )
        self.chisel_client = chisel_client

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
        try:
            console.print(  # TODO: implment logging functions
                f"[cyan]Starting profiling for {command_to_profile} on {gpu_type.value}[/cyan]"
            )

            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                console.print(
                    f"[yellow]Overwriting existing output directory: {output_dir}[/yellow]"
                )
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

            all_results = []
            if rocprofv3_flag:
                result = self.run_rocprofv3(
                    command_to_profile,
                    gpu_type,
                    files_to_sync,
                    output_dir,
                    rocprofv3_flag,
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
                    command_to_profile,
                    gpu_type,
                    files_to_sync,
                    output_dir,
                    nsys_flag,
                )
                all_results.append(result)
            if ncompute_flag:
                result = self.run_ncompute(
                    command_to_profile,
                    gpu_type,
                    files_to_sync,
                    output_dir,
                    ncompute_flag,
                )
                all_results.append(result)

            return ProfilingResults(
                success=True,
                output_dir=output_dir,
                stdout="",
                stderr="",
                summary={
                    "profile_files": [
                        result.local_output_dir for result in all_results
                    ],
                    "summary_file": all_results[0].summary_file
                    if all_results
                    else None,
                    "profile_type": all_results[0].profile_type
                    if all_results
                    else "unknown",
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
        droplet_with_gpu: Droplet = (
            self.droplet_service.get_or_create_droplet_by_type(gpu_type)
        )
        self._ensure_rocprofv3(droplet_with_gpu)

        remote_profile_dir = f"{WORKSPACE_DIR}/{ROCPROFV3_DIR_NAME}"

        RESET_CMD = (
            f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"
        )
        result = droplet_with_gpu.run_container_command(RESET_CMD)
        if result["exit_code"] != 0:
            raise RuntimeError(
                f"Failed to reset remote directory: {result['exit_code']}"
            )

        for file in files_to_sync:
            self._sync_file(droplet_with_gpu, Path(file), remote_profile_dir)
            if file.endswith(".py"):
                self._ensure_pytorch_rocm(droplet_with_gpu)

        adjusted_command = command_to_profile
        if files_to_sync:
            for file in files_to_sync:
                file_path = Path(file)
                if command_to_profile == file or command_to_profile == str(
                    file_path
                ):
                    adjusted_command = f"./{file_path.name}"
                    console.print(
                        f"[cyan]Adjusted command path: {command_to_profile} → {adjusted_command}[/cyan]"
                    )
                    break

        CD_CMD = f"cd {remote_profile_dir}"
        PROFILE_CMD = f"rocprofv3 -S --summary-output-file amd_profile_summary {rocprofv3_flags} -- {adjusted_command}"
        FULL_CMD = f"{CD_CMD} && {PROFILE_CMD}"
        console.print(
            f"[cyan]Running AMD rocprofv3 with command: {FULL_CMD}[/cyan]"
        )
        rocprof_result = droplet_with_gpu.run_container_command(
            FULL_CMD, timeout=600
        )
        if rocprof_result["exit_code"] != 0:
            console.print(
                f"[red]rocprofv3 command failed with exit code {rocprof_result['exit_code']}[/red]"
            )
            console.print(
                f"[red]stdout: {rocprof_result.get('stdout', 'No stdout')}[/red]"
            )
            console.print(
                f"[red]stderr: {rocprof_result.get('stderr', 'No stderr')}[/red]"
            )
            raise RuntimeError(
                f"rocprofv3 profiling failed with exit code {rocprof_result['exit_code']}: {rocprof_result.get('stderr', 'No error details')}"
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
        # TODO: Implement rocprof-compute when ready

        console.print(
            "[yellow]rocprof-compute support not yet implemented[/yellow]"
        )
        raise RuntimeError("rocprof-compute is not yet supported")

    def run_nsys(
        self,
        command_to_profile: str,
        gpu_type: GPUType,
        files_to_sync: List[str],
        output_dir: Path,
        nsys_flags: str,
    ) -> ProfilerResult:
        droplet_with_gpu: Droplet = (
            self.droplet_service.get_or_create_droplet_by_type(gpu_type)
        )
        self._ensure_nvidia_profilers(droplet_with_gpu)

        remote_profile_dir = f"{WORKSPACE_DIR}/{NSYS_DIR_NAME}"

        RESET_CMD = (
            f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"
        )
        result = droplet_with_gpu.run_container_command(RESET_CMD)
        if result["exit_code"] != 0:
            raise RuntimeError(
                f"Failed to reset remote directory: {result['exit_code']}"
            )

        for file in files_to_sync:
            self._sync_file(droplet_with_gpu, Path(file), remote_profile_dir)
            if file.endswith(".py"):
                self._ensure_pytorch(droplet_with_gpu)

        adjusted_command = command_to_profile
        if files_to_sync:
            for file in files_to_sync:
                file_path = Path(file)
                if command_to_profile == file or command_to_profile == str(
                    file_path
                ):
                    adjusted_command = f"./{file_path.name}"
                    console.print(
                        f"[cyan]Adjusted command path: {command_to_profile} → {adjusted_command}[/cyan]"
                    )
                    break

        CD_CMD = f"cd {remote_profile_dir}"
        PROFILE_CMD = (
            f"nsys profile {nsys_flags} -o nvidia_profile {adjusted_command}"
        )
        FULL_CMD = f"{CD_CMD} && {PROFILE_CMD}"
        console.print(
            f"[cyan]Running NVIDIA nsys with command: {FULL_CMD}[/cyan]"
        )
        nsys_result = droplet_with_gpu.run_container_command(
            FULL_CMD, timeout=600
        )
        if nsys_result["exit_code"] != 0:
            console.print(
                f"[red]nsys command failed with exit code {nsys_result['exit_code']}[/red]"
            )
            console.print(
                f"[red]stdout: {nsys_result.get('stdout', 'No stdout')}[/red]"
            )
            console.print(
                f"[red]stderr: {nsys_result.get('stderr', 'No stderr')}[/red]"
            )
            raise RuntimeError(
                f"nsys profiling failed with exit code {nsys_result['exit_code']}: {nsys_result.get('stderr', 'No error details')}"
            )

        nvidia_files = self._download_results(
            droplet_with_gpu, remote_profile_dir, output_dir
        )
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
        droplet_with_gpu: Droplet = (
            self.droplet_service.get_or_create_droplet_by_type(gpu_type)
        )
        self._ensure_nvidia_profilers(droplet_with_gpu)

        remote_profile_dir = f"{WORKSPACE_DIR}/{NCOMPUTE_DIR_NAME}"

        RESET_CMD = (
            f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"
        )
        result = droplet_with_gpu.run_container_command(RESET_CMD)
        if result["exit_code"] != 0:
            raise RuntimeError(
                f"Failed to reset remote directory: {result['exit_code']}"
            )

        for file in files_to_sync:
            self._sync_file(droplet_with_gpu, Path(file), remote_profile_dir)
            if file.endswith(".py"):
                self._ensure_pytorch(droplet_with_gpu)

        CD_CMD = f"cd {remote_profile_dir}"
        PROFILE_CMD = f"ncu {ncompute_flags} -o nvidia_ncompute_profile {command_to_profile}"
        FULL_CMD = f"{CD_CMD} && {PROFILE_CMD}"
        console.print(
            f"[cyan]Running NVIDIA ncu with command: {FULL_CMD}[/cyan]"
        )
        ncu_result = droplet_with_gpu.run_container_command(
            FULL_CMD, timeout=600
        )
        if ncu_result["exit_code"] != 0:
            console.print(
                f"[red]ncu command failed with exit code {ncu_result['exit_code']}[/red]"
            )
            console.print(
                f"[red]stdout: {ncu_result.get('stdout', 'No stdout')}[/red]"
            )
            console.print(
                f"[red]stderr: {ncu_result.get('stderr', 'No stderr')}[/red]"
            )
            raise RuntimeError(
                f"ncu profiling failed with exit code {ncu_result['exit_code']}: {ncu_result.get('stderr', 'No error details')}"
            )

        nvidia_files = self._download_results(
            droplet_with_gpu, remote_profile_dir, output_dir
        )
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

    def _sync_file(
        self, droplet_info: Droplet, source_file: Path, remote_dir: str
    ):
        """Sync a file or directory to the droplet with proper permissions."""
        success = droplet_info.sync_file(str(source_file), f"{remote_dir}/")
        if not success:
            raise RuntimeError(
                f"Failed to sync {source_file} to {remote_dir}. Ensure the file exists and is accessible."
            )

        # If we synced a directory, make all files in it executable
        if source_file.is_dir():
            console.print(
                f"[cyan]📁 {source_file.name} is a directory - making all files executable[/cyan]"
            )
            chmod_dir_cmd = f"find {remote_dir}/{source_file.name} -type f -exec chmod +x {{}} \\;"
            result = droplet_info.run_container_command(chmod_dir_cmd)

            if result["exit_code"] != 0:
                console.print(
                    f"[red]Failed to make directory files executable: {result.get('stderr', '')}[/red]. got stdout: {result.get('stdout', '')}"
                )
            else:
                console.print(
                    f"[green]✓ Made all files in {source_file.name} executable[/green]"
                )
        else:
            # Single file - we'll handle executable permissions later based on what command is actually run
            chmod_cmd = f"chmod +x {remote_dir}/{source_file.name}"
            result = droplet_info.run_container_command(chmod_cmd)
            if result["exit_code"] != 0:
                console.print(
                    f"[red]Failed to make {source_file.name} executable: {result.get('stderr', '')}[/red]"
                )
            else:
                console.print(
                    f"[green]✓ Made {source_file.name} executable[/green]"
                )

            console.print(f"[cyan]✓ {source_file.name} synced[/cyan]")

        console.print(
            f"[green]✓ File synced to {remote_dir} on remote server[/green]"
        )
        return remote_dir

    def _make_command_executable(
        self, droplet_info: Droplet, command: str, remote_dir: str
    ):
        """Make the specific command being run executable."""
        # Extract the actual executable from the command (remove ./ prefix, arguments, etc.)
        command_parts = command.strip().split()
        if not command_parts:
            return

        executable_path = command_parts[0]

        # Remove ./ prefix if present
        if executable_path.startswith("./"):
            executable_path = executable_path[2:]

        # Full path to the executable on remote system
        full_executable_path = f"{remote_dir}/{executable_path}"

        console.print(
            f"[cyan]🔧 Making command executable: {executable_path}[/cyan]"
        )

        # Make the specific executable file executable
        chmod_cmd = f"chmod +x {full_executable_path}"
        result = droplet_info.run_container_command(chmod_cmd)

        if result["exit_code"] != 0:
            console.print(
                f"[red]Failed to make {executable_path} executable: {result.get('stderr', '')}[/red]"
            )
            # Try more permissive permissions
            chmod_cmd_alt = f"chmod 755 {full_executable_path}"
            result_alt = droplet_info.run_container_command(chmod_cmd_alt)
            if result_alt["exit_code"] == 0:
                console.print(
                    f"[green]✓ Made {executable_path} executable with 755 permissions[/green]"
                )
            else:
                console.print(
                    f"[yellow]Warning: Could not make {executable_path} executable[/yellow]"
                )
        else:
            console.print(f"[green]✓ Made {executable_path} executable[/green]")

            # Verify it's actually executable
            verify_cmd = (
                f"test -x {full_executable_path} && echo 'Executable confirmed'"
            )
            verify_result = droplet_info.run_container_command(verify_cmd)
            if verify_result["exit_code"] == 0:
                console.print(
                    f"[green]✓ Verified {executable_path} is executable[/green]"
                )

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
                                "time_ms": event["args"]["DurationNs"]
                                / 1_000_000,
                            }
                        )

                # Sort by time
                kernels.sort(key=lambda x: x["time_ms"], reverse=True)
                summary["top_kernels"] = kernels[:10]

            except Exception as e:
                console.print(
                    f"[yellow]Could not parse JSON results: {e}[/yellow]"
                )

        return summary

    def _ensure_nvidia_profilers(self, droplet_info: Droplet):
        """Ensure both nsight-compute and nsight-systems are installed on the droplet."""
        try:
            # First ensure the host system is properly set up
            if not self.ensure_host_system_ready(droplet_info):
                console.print(
                    "[yellow]Host system setup had issues, but continuing...[/yellow]"
                )

            # Check if NVIDIA profilers are available on host system
            check_cmd = (
                "which ncu && ncu --version && which nsys && nsys --version"
            )
            result = droplet_info.run_container_command(check_cmd)

            if result["exit_code"] == 0:
                console.print(
                    "[green]✓ NVIDIA profilers (ncu + nsys) already available[/green]"
                )
                return

            console.print(
                "[yellow]Profilers not found, installing NVIDIA profilers...[/yellow]"
            )

            # Install NVIDIA profilers on host system
            console.print(
                "[yellow]Installing NVIDIA profilers (nsight-compute + nsight-systems)...[/yellow]"
            )

            # First, ensure we have basic tools and try to install CUDA if missing
            basic_setup_cmd = """
            apt-get update -y &&
            apt-get install -y wget gnupg software-properties-common build-essential
            """

            setup_result = droplet_info.run_command(
                basic_setup_cmd, timeout=180
            )
            if setup_result["exit_code"] != 0:
                console.print(
                    "[yellow]Warning: Failed to install basic dependencies[/yellow]"
                )

            # Try installing CUDA toolkit first if not present
            cuda_check = droplet_info.run_command(
                "which nvcc || ls /usr/local/cuda*/bin/nvcc"
            )
            if cuda_check["exit_code"] != 0:
                console.print(
                    "[yellow]CUDA not detected, attempting to install CUDA toolkit...[/yellow]"
                )
                cuda_install_cmd = """
                timeout 600 bash -c '
                wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb &&
                dpkg -i cuda-keyring_1.1-1_all.deb &&
                apt-get update &&
                apt-get install -y cuda-toolkit-12-6 &&
                echo "✓ CUDA toolkit installed"
                '
                """
                cuda_result = droplet_info.run_command(
                    cuda_install_cmd, timeout=700
                )
                if cuda_result["exit_code"] == 0:
                    console.print("[green]✓ CUDA toolkit installed[/green]")
                else:
                    console.print(
                        "[yellow]CUDA installation failed, but continuing with profiler installation...[/yellow]"
                    )

            # Try installing from snap (most reliable for profilers)
            snap_install_cmd = """
            timeout 600 bash -c '
            apt-get install -y snapd &&
            snap install nsight-compute nsight-systems &&
            ln -sf /snap/nsight-compute/current/bin/ncu /usr/local/bin/ncu &&
            ln -sf /snap/nsight-systems/current/bin/nsys /usr/local/bin/nsys &&
            echo "✓ Installed via snap"
            '
            """

            console.print(
                "[cyan]Trying snap installation for profilers...[/cyan]"
            )
            snap_result = droplet_info.run_command(
                snap_install_cmd, timeout=700
            )

            if snap_result["exit_code"] == 0:
                console.print(
                    "[green]✓ NVIDIA profilers installed via snap[/green]"
                )
            else:
                console.print(
                    "[yellow]Snap installation failed, trying alternative methods...[/yellow]"
                )

                # Try installing from NVIDIA's CUDA repositories
                cuda_repo_install_cmd = """
                timeout 600 bash -c '
                apt-get update -y && 
                apt-get install -y wget gnupg && 
                wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | apt-key add - && 
                echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list && 
                apt-get update -y && 
                apt-get install -y nsight-compute nsight-systems-cli && 
                echo "✓ Installed via NVIDIA CUDA repository"
                '
                """

                console.print(
                    "[cyan]Trying NVIDIA CUDA repository installation...[/cyan]"
                )
                cuda_result = droplet_info.run_command(
                    cuda_repo_install_cmd, timeout=700
                )

                if cuda_result["exit_code"] != 0:
                    console.print(
                        "[yellow]CUDA repository installation failed, trying direct download...[/yellow]"
                    )

                    # Last resort: Download and install manually (simplified approach)
                    direct_install_cmd = """
                    timeout 600 bash -c '
                    cd /tmp && 
                    # Try to get profilers from Ubuntu packages as fallback
                    apt-get install -y nvidia-profiler nvidia-cuda-toolkit-doc || echo "Standard packages not available" &&
                    # Create simple wrapper scripts if binaries not available
                    if ! which ncu >/dev/null 2>&1; then
                        echo "#!/bin/bash" > /usr/local/bin/ncu &&
                        echo "echo \\\"ncu not available - please install nsight-compute manually\\\"" >> /usr/local/bin/ncu &&
                        chmod +x /usr/local/bin/ncu
                    fi &&
                    if ! which nsys >/dev/null 2>&1; then
                        echo "#!/bin/bash" > /usr/local/bin/nsys &&
                        echo "echo \\\"nsys not available - please install nsight-systems manually\\\"" >> /usr/local/bin/nsys &&
                        chmod +x /usr/local/bin/nsys
                    fi &&
                    echo "✓ Fallback installation completed"
                    '
                    """

                    console.print(
                        "[cyan]Trying fallback installation...[/cyan]"
                    )
                    direct_result = droplet_info.run_command(
                        direct_install_cmd, timeout=700
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
                        uname -a &&
                        echo "=== Virtual environment ===" &&
                        ls -la /opt/venv/bin/activate 2>/dev/null || echo "Virtual environment not found"
                        """

                        debug_result = droplet_info.run_command(debug_cmd)
                        console.print("[cyan]Debug information:[/cyan]")
                        console.print(
                            debug_result.get("stdout", "No debug output")
                        )
                        console.print("[red]Debug stderr:[/red]")
                        console.print(
                            debug_result.get("stderr", "No debug stderr")
                        )

                        console.print(
                            "[yellow]Warning: Failed to install NVIDIA profilers using all available methods.[/yellow]"
                        )
                        console.print(
                            "[yellow]The droplet may not have the required NVIDIA drivers or repositories configured.[/yellow]"
                        )
                        console.print(
                            "[yellow]Profiling may still work if tools are available through other means.[/yellow]"
                        )
                        # Don't raise an error, just continue and let the verification handle it

            # Verify both installations work
            verify_ncu_result = droplet_info.run_container_command(
                "which ncu && ncu --version"
            )
            verify_nsys_result = droplet_info.run_container_command(
                "which nsys && nsys --version"
            )

            ncu_available = verify_ncu_result["exit_code"] == 0
            nsys_available = verify_nsys_result["exit_code"] == 0

            if ncu_available and nsys_available:
                console.print(
                    "[green]✓ NVIDIA profilers installed and verified successfully (ncu + nsys)[/green]"
                )
            elif ncu_available:
                console.print("[green]✓ nsight-compute (ncu) available[/green]")
                console.print(
                    "[yellow]⚠ nsight-systems (nsys) not available[/yellow]"
                )
            elif nsys_available:
                console.print(
                    "[green]✓ nsight-systems (nsys) available[/green]"
                )
                console.print(
                    "[yellow]⚠ nsight-compute (ncu) not available[/yellow]"
                )
            else:
                console.print(
                    "[yellow]⚠ Neither NVIDIA profiler is available - profiling may fail[/yellow]"
                )
                console.print(
                    "[yellow]You may need to manually install nsight-compute and nsight-systems[/yellow]"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to setup NVIDIA profilers: {e}")

    def _ensure_pytorch(self, droplet_info: Droplet):
        """Check that PyTorch with CUDA support is available on the host system."""

        try:
            # Check if PyTorch is available with virtual environment activated
            check_cmd = "python -c \"import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')\""
            result = droplet_info.run_container_command(check_cmd)

            if result["exit_code"] == 0:
                console.print(
                    "[green]✓ PyTorch with CUDA available on host system[/green]"
                )
                console.print(
                    f"[cyan]PyTorch info: {result['stdout'].strip()}[/cyan]"
                )
                return

            console.print(
                "[yellow]PyTorch not detected - checking if virtual environment is properly set up[/yellow]"
            )

            # TODO: fix this logic; we should either chose to patch for the users or throw immediately.
            if not self.ensure_host_system_ready(droplet_info):
                console.print(
                    "[yellow]Host system not ready, but continuing...[/yellow]"
                )

            install_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            result = droplet_info.run_container_command(
                install_cmd, timeout=300
            )

            if result["exit_code"] == 0:
                console.print("[green]✓ PyTorch installed successfully[/green]")
                # Verify installation
                verify_result = droplet_info.run_container_command(check_cmd)
                if verify_result["exit_code"] == 0:
                    console.print(
                        f"[cyan]PyTorch info: {verify_result['stdout'].strip()}[/cyan]"
                    )
                return
            else:
                console.print(
                    "[yellow]Warning: Failed to install PyTorch with CUDA[/yellow]"
                )
                # Try CPU version as fallback
                fallback_cmd = "pip install torch torchvision torchaudio"
                fallback_result = droplet_info.run_container_command(
                    fallback_cmd, timeout=300
                )
                if fallback_result["exit_code"] == 0:
                    console.print(
                        "[yellow]✓ PyTorch (CPU version) installed as fallback[/yellow]"
                    )
                else:
                    console.print(
                        "[yellow]Profiling may still work with existing packages[/yellow]"
                    )

        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not verify PyTorch: {e}[/yellow]"
            )
            console.print(
                "[yellow]Continuing anyway - setup may still be in progress[/yellow]"
            )

    def _ensure_rocprofv3(self, droplet_info: Droplet):
        """Ensure rocprofv3 and dependencies are installed on the AMD droplet."""
        try:
            # First ensure the host system is properly set up
            if not self.ensure_host_system_ready(droplet_info):
                console.print(
                    "[yellow]Host system setup had issues, but continuing...[/yellow]"
                )

            # Check if rocprofv3 is already available
            check_cmd = "which rocprofv3 && echo 'rocprofv3 available'"
            result = droplet_info.run_container_command(check_cmd)

            if result["exit_code"] == 0:
                console.print("[green]✓ rocprofv3 already available[/green]")
                return

            console.print(
                "[yellow]rocprofv3 not found, checking system setup...[/yellow]"
            )

            # First validate that ROCm is properly installed
            self._validate_rocm_installation(droplet_info)

            console.print(
                "[yellow]Installing rocprofv3 and dependencies...[/yellow]"
            )

            # Install build dependencies and build tools with verbose output
            setup_cmd = """
            timeout 1800 bash -c '
            apt-get update -y && 
            apt-get install -y git cmake build-essential python3 python3-pip wget && \\
            echo "✓ Build dependencies installed"
            '
            """

            console.print("[cyan]Installing build dependencies...[/cyan]")
            setup_result = droplet_info.run_container_command(
                setup_cmd, timeout=1900
            )
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
            aql_result = droplet_info.run_container_command(
                build_aqlprofile_cmd, timeout=1200
            )
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
            decoder_result = droplet_info.run_container_command(
                download_decoder_cmd, timeout=300
            )
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

            console.print(
                "[green]✓ rocprofv3 and dependencies installed successfully[/green]"
            )
            console.print("[cyan]rocprofv3 help preview:[/cyan]")
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
                rocm_check_cmd = (
                    "ls -la /opt/rocm/ && echo 'ROCm directory exists'"
                )
                rocm_result = droplet_info.run_container_command(rocm_check_cmd)

                if rocm_result["exit_code"] != 0:
                    raise RuntimeError(
                        "ROCm installation not found at /opt/rocm/"
                    )

                # Try alternative rocminfo locations
                alt_rocminfo_cmd = "which rocminfo || find /opt -name rocminfo 2>/dev/null || echo 'rocminfo not found'"
                alt_result = droplet_info.run_container_command(
                    alt_rocminfo_cmd
                )

                if "rocminfo not found" in alt_result.get("stdout", ""):
                    console.print(
                        "[yellow]Warning: rocminfo not available, but ROCm directory exists[/yellow]"
                    )
                    console.print(
                        "[yellow]This may still work for profiling[/yellow]"
                    )
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
                    console.print(
                        "[green]✓ GPU device(s) detected by ROCm[/green]"
                    )
                else:
                    console.print(
                        "[yellow]Warning: No GPU devices detected in rocminfo output[/yellow]"
                    )

        except Exception as e:
            console.print(
                f"[yellow]Warning: ROCm validation failed: {e}[/yellow]"
            )
            console.print(
                "[yellow]Continuing anyway - this may still work in the host environment[/yellow]"
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
                console.print("[cyan]Debug output:[/cyan]")
                for line in result.get("stdout", "").split("\n"):
                    if line.strip():
                        console.print(f"  {line.strip()}")
            else:
                console.print(
                    "[yellow]Could not gather ROCm debug information[/yellow]"
                )

        except Exception as e:
            console.print(
                f"[yellow]Debug information gathering failed: {e}[/yellow]"
            )

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
            result = subprocess.run(
                scp_cmd, capture_output=True, text=True, timeout=300
            )
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
                                local_output_dir
                                / f"{name_parts[0]}_{counter}.{name_parts[1]}"
                            )
                        else:
                            target_path = (
                                local_output_dir / f"{clean_name}_{counter}"
                            )
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
                                    local_output_dir
                                    / f"{name_parts[0]}_{counter}.{name_parts[1]}"
                                )
                            else:
                                target_path = (
                                    local_output_dir / f"{clean_name}_{counter}"
                                )
                            counter += 1

                        if target_path != file_path:
                            file_path.rename(target_path)
                        console.print(
                            f"[green]✓ Downloaded: {original_name} -> {target_path.name}[/green]"
                        )
                        downloaded_files.append(target_path.name)
                    else:
                        console.print(
                            f"[green]✓ Downloaded: {original_name}[/green]"
                        )
                        downloaded_files.append(original_name)

            # Remove any empty subdirectories
            for item in local_output_dir.iterdir():
                if item.is_dir():
                    try:
                        item.rmdir()  # Only removes if empty
                        console.print(
                            f"[green]✓ Removed empty directory: {item.name}[/green]"
                        )
                    except OSError:
                        # Directory not empty, leave it
                        pass

            if not downloaded_files:
                console.print(
                    "[yellow]Warning: No files were downloaded[/yellow]"
                )
                return []

            console.print(
                f"[green]✓ Profiling results downloaded ({len(downloaded_files)} files)[/green]"
            )
            return downloaded_files

        except subprocess.TimeoutExpired:
            console.print("[yellow]Warning: Download timed out[/yellow]")
            return []
        except Exception as e:
            console.print(
                f"[yellow]Warning: Unexpected error during download: {e}[/yellow]"
            )
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
            if (
                stats_file.suffix == ".json"
                or stats_file.name == "results.json"
            ):
                with open(stats_file, "r") as f:
                    data = json.load(f)

                kernels = []
                for event in data.get("traceEvents", []):
                    if (
                        event.get("ph") == "X"
                        and "pid" in event
                        and event.get("pid") in [6, 7]
                        and "DurationNs" in event.get("args", {})
                    ):
                        kernel_name = event.get("name", "")
                        duration_ns = int(event["args"]["DurationNs"])

                        kernels.append(
                            {
                                "name": kernel_name,
                                "total_time": duration_ns
                                / 1_000_000,  # Convert to ms
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
                                "total_time": duration_ns
                                / 1_000_000,  # Convert to ms
                                "duration_ns": duration_ns,
                            }
                        )

                # Sort by total time
                hip_calls.sort(key=lambda x: x["total_time"], reverse=True)

                if hip_calls:
                    console.print(
                        "\n[cyan]Top HIP API Calls by Total Time:[/cyan]"
                    )
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
            console.print(
                f"[yellow]Could not parse profile summary: {e}[/yellow]"
            )

    def _ensure_pytorch_rocm(self, droplet_info: Droplet):
        """Check that PyTorch with ROCm support is available on the host system."""

        try:
            # Check if PyTorch is available with virtual environment activated
            check_cmd = "python -c \"import torch; print(f'PyTorch {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')\""
            result = droplet_info.run_container_command(check_cmd)

            if result["exit_code"] == 0:
                console.print(
                    "[green]✓ PyTorch with ROCm available on host system[/green]"
                )
                console.print(
                    f"[cyan]PyTorch info: {result['stdout'].strip()}[/cyan]"
                )
                return

            console.print(
                "[yellow]PyTorch not detected - checking if virtual environment is properly set up[/yellow]"
            )

            # Try installing PyTorch with ROCm support if it's missing
            # First ensure virtual environment is ready
            if not self.ensure_host_system_ready(droplet_info):
                console.print(
                    "[yellow]Host system not ready, but continuing...[/yellow]"
                )

            install_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1"
            result = droplet_info.run_container_command(
                install_cmd, timeout=300
            )

            if result["exit_code"] == 0:
                console.print(
                    "[green]✓ PyTorch with ROCm installed successfully[/green]"
                )
                # Verify installation
                verify_result = droplet_info.run_container_command(check_cmd)
                if verify_result["exit_code"] == 0:
                    console.print(
                        f"[cyan]PyTorch info: {verify_result['stdout'].strip()}[/cyan]"
                    )
                return
            else:
                console.print(
                    "[yellow]Warning: Failed to install PyTorch with ROCm[/yellow]"
                )
                # Try CPU version as fallback
                fallback_cmd = "pip install torch torchvision torchaudio"
                fallback_result = droplet_info.run_container_command(
                    fallback_cmd, timeout=300
                )
                if fallback_result["exit_code"] == 0:
                    console.print(
                        "[yellow]✓ PyTorch (CPU version) installed as fallback[/yellow]"
                    )
                else:
                    console.print(
                        "[yellow]Profiling may still work with existing packages[/yellow]"
                    )

        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not verify PyTorch: {e}[/yellow]"
            )
            console.print(
                "[yellow]Continuing anyway - setup may still be in progress[/yellow]"
            )

    def debug_droplet_system_status(self, droplet_info: Droplet):
        """Debug the system status on the droplet."""
        try:
            console.print("[cyan]🔍 Debugging droplet system status...[/cyan]")

            # Check virtual environment
            venv_status_cmd = "ls -la /opt/venv && source /opt/venv/bin/activate && python --version"
            result = droplet_info.run_command(venv_status_cmd)
            console.print("[cyan]Virtual environment status:[/cyan]")
            console.print(result.get("stdout", "No output"))

            # Check workspace directory
            workspace_cmd = (
                "ls -la /workspace || echo 'Workspace directory not found'"
            )
            result = droplet_info.run_command(workspace_cmd)
            console.print("[cyan]Workspace directory:[/cyan]")
            console.print(result.get("stdout", "No workspace info"))

            # Check cloud-init logs
            cloud_init_cmd = (
                "cloud-init status && tail -50 /var/log/cloud-init-output.log"
            )
            result = droplet_info.run_command(cloud_init_cmd)
            console.print("[cyan]Cloud-init status and logs:[/cyan]")
            console.print(result.get("stdout", "No cloud-init logs"))

            # Check GPU devices
            gpu_cmd = "ls -la /dev/kfd /dev/dri/ 2>/dev/null || echo 'No GPU devices found'"
            result = droplet_info.run_command(gpu_cmd)
            console.print("[cyan]GPU devices:[/cyan]")
            console.print(result.get("stdout", "No GPU info"))

            # Check ROCm installation
            rocm_cmd = "ls -la /opt/rocm/ 2>/dev/null || echo 'ROCm not found'"
            result = droplet_info.run_command(rocm_cmd)
            console.print("[cyan]ROCm installation:[/cyan]")
            console.print(result.get("stdout", "No ROCm info"))

            # Check CUDA installation
            cuda_cmd = (
                "ls -la /usr/local/cuda/ 2>/dev/null || echo 'CUDA not found'"
            )
            result = droplet_info.run_command(cuda_cmd)
            console.print("[cyan]CUDA installation:[/cyan]")
            console.print(result.get("stdout", "No CUDA info"))

        except Exception as e:
            console.print(f"[red]Debug failed: {e}[/red]")

    def wait_for_cloud_init(self, droplet_info: Droplet, timeout: int = 600):
        import time

        # First check if cloud-init has already completed
        if self._is_cloud_init_already_complete(droplet_info):
            console.print(
                "[green]✓ Cloud-init already completed - system is ready![/green]"
            )
            return True

        # Show progress with estimated timing
        estimated_time = 120  # 2 minutes estimate
        console.print(
            "[cyan]🚀 Setting up GPU droplet (estimated: ~2 minutes)[/cyan]"
        )
        console.print(
            "[dim cyan]📦 Downloading system dependencies and GPU drivers...[/dim cyan]"
        )

        start_time = time.time()
        last_log_lines_shown = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            setup_task = progress.add_task(
                "Initializing system...", total=estimated_time
            )

            while time.time() - start_time < timeout:
                elapsed = time.time() - start_time

                # Update progress based on elapsed time and estimated completion
                progress_percentage = min(
                    (elapsed / estimated_time) * 100, 95
                )  # Cap at 95% until actually done
                progress.update(
                    setup_task, completed=min(elapsed, estimated_time * 0.95)
                )

                # Update task description based on progress
                if elapsed < 30:
                    description = "Starting cloud-init and package manager..."
                elif elapsed < 60:
                    description = "Installing CUDA/ROCm drivers and toolkits..."
                elif elapsed < 90:
                    description = "Setting up Python environment and PyTorch..."
                elif elapsed < 120:
                    description = "Finalizing configuration and environment..."
                else:
                    description = "Completing final setup steps..."

                progress.update(setup_task, description=description)

                try:
                    result = droplet_info.run_command(
                        "cloud-init status --long"
                    )

                    if result.get("exit_code") != 0:
                        # Try fallback methods
                        simple_result = droplet_info.run_command(
                            "cloud-init status"
                        )
                        if simple_result.get("exit_code") == 0:
                            status = simple_result.get("stdout", "").strip()

                            if "status: done" in status:
                                progress.update(
                                    setup_task,
                                    completed=estimated_time,
                                    description="✓ Setup completed!",
                                )
                                console.print(
                                    "[green]✓ Cloud-init setup completed successfully![/green]"
                                )
                                self._show_cloud_init_summary(droplet_info)
                                return True
                            elif "status: running" in status:
                                # Show some activity but continue with progress bar
                                last_log_lines_shown = (
                                    self._show_recent_activity(
                                        droplet_info, last_log_lines_shown
                                    )
                                )
                        else:
                            # Check if cloud-init has actually completed using alternative methods
                            if self._is_cloud_init_actually_complete(
                                droplet_info
                            ):
                                progress.update(
                                    setup_task,
                                    completed=estimated_time,
                                    description="✓ Setup completed!",
                                )
                                console.print(
                                    "[green]✓ Cloud-init setup completed (detected via fallback)![/green]"
                                )
                                return True

                            # Show warning and continue with limited detection
                            if (
                                int(elapsed) % 30 == 0
                            ):  # Show warning every 30 seconds
                                console.print(
                                    f"[yellow]⚠️ cloud-init status not available yet ({int(elapsed)}s) - trying fallback methods...[/yellow]"
                                )
                                self._check_system_activity_with_completion(
                                    droplet_info
                                )

                        time.sleep(5)
                        continue

                    status_output = result.get("stdout", "").strip()
                    lines = status_output.split("\n")
                    overall_status = lines[0] if lines else "unknown"

                    if "status: done" in overall_status:
                        progress.update(
                            setup_task,
                            completed=estimated_time,
                            description="✓ Setup completed!",
                        )
                        console.print(
                            "[green]✓ Cloud-init setup completed successfully![/green]"
                        )
                        self._show_cloud_init_summary(droplet_info)
                        return True
                    elif "status: error" in overall_status:
                        progress.update(
                            setup_task,
                            completed=estimated_time,
                            description="⚠️ Completed with warnings",
                        )
                        console.print(
                            "[yellow]⚠️ Cloud-init completed with errors[/yellow]"
                        )
                        self._show_cloud_init_logs(droplet_info, lines=10)
                        return True
                    elif "status: running" in overall_status:
                        # Show recent activity but keep progress bar going
                        last_log_lines_shown = self._show_recent_activity(
                            droplet_info, last_log_lines_shown
                        )
                        time.sleep(5)
                    else:
                        time.sleep(5)

                except Exception:
                    # Continue with progress bar even if status check fails
                    time.sleep(5)

        # If we exit the progress loop, show timeout message
        console.print("[yellow]⚠️ Setup is taking longer than expected[/yellow]")
        console.print(
            "[cyan]💡 The system may still be working - this is normal for GPU droplets[/cyan]"
        )
        self._show_helpful_timeout_guidance(droplet_info)
        return False

    def _show_recent_activity(
        self, droplet_info: Droplet, last_lines_shown: int
    ) -> int:
        """Show recent activity without overwhelming the progress display."""
        try:
            # Only show activity every few checks to avoid spam
            if (
                random.randint(1, 3) == 1
            ):  # Show activity roughly 1/3 of the time
                result = droplet_info.run_command(
                    "tail -n 3 /var/log/cloud-init-output.log 2>/dev/null"
                )
                if result.get("exit_code") == 0:
                    output = result.get("stdout", "").strip()
                    if output:
                        lines = output.split("\n")
                        # Show just one meaningful line
                        for line in reversed(lines):
                            if (
                                line.strip()
                                and not line.startswith(" ")
                                and len(line.strip()) > 10
                            ):
                                console.print(
                                    f"[dim blue]  📋 {line.strip()}[/dim blue]"
                                )
                                break
            return last_lines_shown + 1
        except Exception:
            return last_lines_shown

    def _is_cloud_init_already_complete(self, droplet_info: Droplet) -> bool:
        """Check if cloud-init has already completed on this system."""
        try:
            result = droplet_info.run_command(
                "test -f /var/lib/cloud/instance/boot-finished && echo 'completed'"
            )
            if result.get("exit_code") == 0 and "completed" in result.get(
                "stdout", ""
            ):
                console.print(
                    "[dim green]✓ Found cloud-init completion marker[/dim green]"
                )
                return True

            result = droplet_info.run_command("cloud-init status")
            if result.get("exit_code") == 0:
                status = result.get("stdout", "").strip()
                if "status: done" in status:
                    console.print(
                        "[dim green]✓ Cloud-init status shows completed[/dim green]"
                    )
                    return True
                elif "status: disabled" in status:
                    console.print(
                        "[dim green]✓ Cloud-init is disabled - system ready[/dim green]"
                    )
                    return True

            result = droplet_info.run_command(
                "grep -q 'Cloud-init.*finished' /var/log/cloud-init.log 2>/dev/null && echo 'finished'"
            )
            if result.get("exit_code") == 0 and "finished" in result.get(
                "stdout", ""
            ):
                console.print(
                    "[dim green]✓ Cloud-init logs show completion[/dim green]"
                )
                return True

            result = droplet_info.run_command(
                "test -d /opt/venv && test -f /root/.bashrc && echo 'setup_complete'"
            )
            if result.get("exit_code") == 0 and "setup_complete" in result.get(
                "stdout", ""
            ):
                console.print(
                    "[dim green]✓ System appears fully configured[/dim green]"
                )
                return True

        except Exception as e:
            console.print(
                f"[dim yellow]Could not check cloud-init completion status: {e}[/dim yellow]"
            )

        return False

    def _check_system_activity(self, droplet_info: Droplet):
        """Check for system activity when cloud-init status isn't available."""
        try:
            # Check if processes are running
            result = droplet_info.run_command(
                "ps aux | grep -E '(apt|dpkg|python|pip)' | grep -v grep | wc -l"
            )
            if result.get("exit_code") == 0:
                process_count = int(result.get("stdout", "0").strip())
                if process_count > 0:
                    console.print(
                        f"[dim cyan]System activity detected: {process_count} setup processes running[/dim cyan]"
                    )

            # Check if we can see any startup activity in logs
            result = droplet_info.run_command(
                "tail -n 3 /var/log/syslog 2>/dev/null || echo 'No syslog available'"
            )
            if result.get("exit_code") == 0:
                output = result.get("stdout", "").strip()
                if output and "No syslog available" not in output:
                    console.print(
                        "[dim cyan]Recent system activity:[/dim cyan]"
                    )
                    for line in output.split("\n")[-2:]:
                        if line.strip():
                            console.print(
                                f"[dim cyan]  {line.strip()}[/dim cyan]"
                            )
        except Exception:
            pass

    def _check_system_activity_with_completion(self, droplet_info: Droplet):
        """Check for system activity and also look for completion signs."""
        try:
            # Check if processes are running
            result = droplet_info.run_command(
                "ps aux | grep -E '(cloud-init|apt|dpkg|python|pip)' | grep -v grep | wc -l"
            )
            if result.get("exit_code") == 0:
                process_count = int(result.get("stdout", "0").strip())
                if process_count > 0:
                    console.print(
                        "Cloud-init not ready yet, checking system activity..."
                    )
                    console.print(
                        f"System activity detected: {process_count} setup processes running"
                    )
                else:
                    console.print(
                        "No setup processes detected - system may be ready"
                    )

            # Check recent system logs for completion or error patterns
            result = droplet_info.run_command(
                "tail -n 3 /var/log/syslog 2>/dev/null | grep -E '(session|Started|Deactivated)' || echo 'No recent activity'"
            )
            if result.get("exit_code") == 0:
                output = result.get("stdout", "").strip()
                if output and "No recent activity" not in output:
                    console.print("Recent system activity:")
                    for line in output.split("\n")[-2:]:
                        if line.strip():
                            console.print(f"  {line.strip()}")
        except Exception:
            pass

    def _is_cloud_init_actually_complete(self, droplet_info: Droplet) -> bool:
        """Robust check for cloud-init completion when status commands fail."""
        try:
            # Check 1: Look for completion marker file
            result = droplet_info.run_command(
                "test -f /var/lib/cloud/instance/boot-finished && echo 'boot_finished'"
            )
            if result.get("exit_code") == 0 and "boot_finished" in result.get(
                "stdout", ""
            ):
                console.print(
                    "[dim green]✓ Found cloud-init boot completion marker[/dim green]"
                )
                return True

            # Check 2: Look for completion in logs
            result = droplet_info.run_command(
                "grep -q 'Cloud-init.*finished' /var/log/cloud-init.log 2>/dev/null && echo 'log_finished'"
            )
            if result.get("exit_code") == 0 and "log_finished" in result.get(
                "stdout", ""
            ):
                console.print(
                    "[dim green]✓ Found completion message in cloud-init logs[/dim green]"
                )
                return True

            # Check 3: Look for expected environment setup
            result = droplet_info.run_command(
                "test -d /opt/venv && test -f /root/.bashrc && ls /opt/venv/bin/activate >/dev/null 2>&1 && echo 'env_ready'"
            )
            if result.get("exit_code") == 0 and "env_ready" in result.get(
                "stdout", ""
            ):
                console.print(
                    "[dim green]✓ Python virtual environment appears configured[/dim green]"
                )
                return True

            # Check 4: Look for evidence that user-data script completed
            result = droplet_info.run_command(
                "ls /var/lib/cloud/instances/*/sem/config_scripts_user >/dev/null 2>&1 && echo 'scripts_done'"
            )
            if result.get("exit_code") == 0 and "scripts_done" in result.get(
                "stdout", ""
            ):
                console.print(
                    "[dim green]✓ User-data scripts appear to have completed[/dim green]"
                )
                return True

            # Check 5: No active cloud-init processes and system looks configured
            result = droplet_info.run_command(
                "ps aux | grep 'cloud-init' | grep -v grep | wc -l"
            )
            if result.get("exit_code") == 0:
                cloud_init_processes = int(result.get("stdout", "0").strip())
                if cloud_init_processes == 0:
                    # No cloud-init processes running, check if basic setup exists
                    result = droplet_info.run_command(
                        "which python3 >/dev/null && which pip >/dev/null && echo 'basic_tools_ready'"
                    )
                    if result.get(
                        "exit_code"
                    ) == 0 and "basic_tools_ready" in result.get("stdout", ""):
                        console.print(
                            "[dim green]✓ No cloud-init processes running and basic tools available[/dim green]"
                        )
                        return True

        except Exception as e:
            console.print(
                f"[dim yellow]Error checking completion status: {e}[/dim yellow]"
            )

        return False

    def _show_helpful_timeout_guidance(self, droplet_info: Droplet):
        """Show helpful guidance when cloud-init monitoring times out."""
        console.print("\n[cyan]💡 Helpful next steps:[/cyan]")
        console.print(
            "[dim cyan]  • The droplet might still be initializing in the background[/dim cyan]"
        )
        console.print(
            "[dim cyan]  • You can check manually with: ssh root@<droplet-ip> 'cloud-init status'[/dim cyan]"
        )
        console.print(
            "[dim cyan]  • GPU setup can take 5-10 minutes on first boot[/dim cyan]"
        )
        console.print(
            "[dim cyan]  • Try connecting via SSH to see current progress[/dim cyan]"
        )

        # Try to give current droplet IP if available
        if hasattr(droplet_info, "ip") and droplet_info.ip:
            console.print(
                f"[dim cyan]  • Your droplet IP: {droplet_info.ip}[/dim cyan]"
            )

        console.print(
            "[green]🚀 Chisel will continue - the system should be ready soon![/green]"
        )

    def _show_all_cloud_init_activity(
        self, droplet_info: Droplet, last_lines_shown: int
    ) -> int:
        try:
            # Get total line count first
            count_result = droplet_info.run_command(
                "wc -l /var/log/cloud-init-output.log 2>/dev/null || echo '0'"
            )
            if count_result.get("exit_code") != 0:
                return last_lines_shown

            total_lines = int(
                count_result.get("stdout", "0").strip().split()[0]
            )

            if total_lines <= last_lines_shown:
                return last_lines_shown

            # Show new lines since last check
            new_lines = total_lines - last_lines_shown
            if new_lines > 0:
                # Show the new lines (up to 50 at a time to avoid spam)
                lines_to_show = min(new_lines, 50)
                result = droplet_info.run_command(
                    f"tail -n {lines_to_show} /var/log/cloud-init-output.log"
                )

                if result.get("exit_code") == 0:
                    output = result.get("stdout", "").strip()
                    if output:
                        console.print(
                            f"[dim yellow]Activity (showing last {lines_to_show} lines):[/dim yellow]"
                        )
                        for line in output.split("\n"):
                            if line.strip():
                                console.print(
                                    f"[dim yellow]  {line.strip()}[/dim yellow]"
                                )
                        console.print()  # Empty line for readability

            return total_lines

        except Exception as e:
            console.print(f"[dim red]Error reading activity log: {e}[/dim red]")
            return last_lines_shown

    def _show_final_cloud_init_activity(self, droplet_info: Droplet):
        try:
            # Show the full cloud-init output log on completion/timeout
            result = droplet_info.run_command(
                "wc -l /var/log/cloud-init-output.log 2>/dev/null || echo '0'"
            )
            if result.get("exit_code") == 0:
                total_lines = int(result.get("stdout", "0").strip().split()[0])
                console.print(
                    f"[cyan]📋 Complete cloud-init activity log ({total_lines} lines total):[/cyan]"
                )

                # Show last 30 lines of the complete log
                result = droplet_info.run_command(
                    "tail -n 30 /var/log/cloud-init-output.log"
                )
                if result.get("exit_code") == 0:
                    output = result.get("stdout", "").strip()
                    if output:
                        for line in output.split("\n"):
                            if line.strip():
                                console.print(
                                    f"[dim green]  {line.strip()}[/dim green]"
                                )
        except Exception:
            pass

    def _show_current_cloud_init_activity(
        self, droplet_info: Droplet, last_position: int
    ):
        # This method is now replaced by _show_all_cloud_init_activity but keeping for compatibility
        pass

    def _show_cloud_init_summary(self, droplet_info: Droplet):
        try:
            result = droplet_info.run_command("cloud-init analyze show")
            if result.get("exit_code") == 0:
                console.print("[green]Cloud-init modules completed:[/green]")
                output = result.get("stdout", "")
                for line in output.split("\n")[:10]:
                    if line.strip() and "Analyzing" not in line:
                        console.print(
                            f"[dim green]  {line.strip()}[/dim green]"
                        )
        except Exception:
            pass

    def _show_cloud_init_logs(self, droplet_info: Droplet, lines: int = 20):
        try:
            result = droplet_info.run_command(
                f"tail -n {lines} /var/log/cloud-init.log"
            )
            if result.get("exit_code") == 0:
                console.print(
                    f"[yellow]Last {lines} lines of cloud-init.log:[/yellow]"
                )
                output = result.get("stdout", "")
                for line in output.split("\n")[-10:]:
                    if line.strip():
                        console.print(
                            f"[dim yellow]  {line.strip()}[/dim yellow]"
                        )
        except Exception:
            pass

    def get_cloud_init_detailed_status(
        self, droplet_info: Droplet
    ) -> Dict[str, Any]:
        status_info = {
            "overall_status": "unknown",
            "stages": {},
            "modules": [],
            "errors": [],
            "timing": {},
        }

        try:
            result = droplet_info.run_command("cloud-init status --long")
            if result.get("exit_code") == 0:
                status_info["overall_status"] = result.get("stdout", "").strip()

            result = droplet_info.run_command("cloud-init analyze show")
            if result.get("exit_code") == 0:
                timing_output = result.get("stdout", "")
                status_info["timing"]["raw"] = timing_output

                for line in timing_output.split("\n"):
                    if "took" in line and "seconds" in line:
                        status_info["modules"].append(line.strip())

            result = droplet_info.run_command(
                "grep -i error /var/log/cloud-init.log | tail -5"
            )
            if result.get("exit_code") == 0:
                errors = result.get("stdout", "").strip()
                if errors:
                    status_info["errors"] = errors.split("\n")

            result = droplet_info.run_command(
                "ls -la /var/lib/cloud/instances/*/sem/ 2>/dev/null | grep -E 'config|final' || echo 'No stage info'"
            )
            if result.get("exit_code") == 0:
                stage_output = result.get("stdout", "")
                if "No stage info" not in stage_output:
                    status_info["stages"]["raw"] = stage_output

        except Exception as e:
            status_info["error"] = str(e)

        return status_info

    def get_full_cloud_init_log(self, droplet_info: Droplet) -> str:
        """Get the complete cloud-init output log for debugging."""
        try:
            result = droplet_info.run_command(
                "cat /var/log/cloud-init-output.log"
            )
            if result.get("exit_code") == 0:
                return result.get("stdout", "")
        except Exception:
            pass
        return "Could not retrieve cloud-init log"

    def monitor_cloud_init_realtime(
        self, droplet_info: Droplet, duration: int = 300
    ):
        """Monitor cloud-init in real-time for a specified duration."""
        import time

        console.print(
            f"[cyan]🔍 Monitoring cloud-init activity for {duration} seconds...[/cyan]"
        )
        start_time = time.time()
        last_lines_shown = 0

        while time.time() - start_time < duration:
            try:
                # Check if cloud-init is still running
                status_result = droplet_info.run_command("cloud-init status")
                if status_result.get("exit_code") == 0:
                    status = status_result.get("stdout", "").strip()
                    elapsed = int(time.time() - start_time)
                    console.print(f"[cyan]⏱️  {elapsed}s - {status}[/cyan]")

                    if "status: done" in status:
                        console.print("[green]✅ Cloud-init completed![/green]")
                        break

                # Show new activity
                last_lines_shown = self._show_all_cloud_init_activity(
                    droplet_info, last_lines_shown
                )
                time.sleep(5)  # More frequent updates for real-time monitoring

            except Exception as e:
                console.print(f"[red]Error during monitoring: {e}[/red]")
                break

        console.print("[cyan]📊 Final summary:[/cyan]")
        self._show_cloud_init_summary(droplet_info)

    def ensure_host_system_ready(self, droplet_info: Droplet):
        """Ensure the host system is properly set up for profiling."""
        try:
            console.print(
                "[yellow]🔧 Ensuring host system is ready for profiling...[/yellow]"
            )

            # Wait for cloud-init to finish if it's still running
            self.wait_for_cloud_init(droplet_info)

            # Check if virtual environment exists and is working
            venv_check_cmd = "test -f /opt/venv/bin/activate && source /opt/venv/bin/activate && python --version"
            result = droplet_info.run_command(venv_check_cmd)

            if result.get("exit_code") == 0:
                console.print("[green]✓ Virtual environment is ready[/green]")
                return True

            console.print(
                "[yellow]Virtual environment missing or broken, creating new one...[/yellow]"
            )

            # First check if we have python3
            python_check = droplet_info.run_command("which python3")
            if python_check.get("exit_code") != 0:
                console.print(
                    "[yellow]Installing Python3 and dependencies...[/yellow]"
                )
                install_python_cmd = "apt-get update && apt-get install -y python3 python3-venv python3-pip build-essential"
                result = droplet_info.run_command(
                    install_python_cmd, timeout=180
                )
                if result.get("exit_code") != 0:
                    console.print(
                        f"[red]Failed to install Python3: {result.get('stderr', '')}[/red]"
                    )
                    return False

            # Create workspace directory
            workspace_cmd = "mkdir -p /workspace && chmod 755 /workspace"
            droplet_info.run_command(workspace_cmd)

            # Remove any broken virtual environment
            cleanup_cmd = "rm -rf /opt/venv"
            droplet_info.run_command(cleanup_cmd)

            # Create virtual environment if it doesn't exist
            create_venv_cmd = """
            python3 -m venv /opt/venv &&
            source /opt/venv/bin/activate &&
            pip install --upgrade pip setuptools wheel &&
            pip install rich pydo paramiko
            """

            result = droplet_info.run_command(create_venv_cmd, timeout=300)

            if result.get("exit_code") == 0:
                console.print(
                    "[green]✓ Virtual environment created successfully[/green]"
                )

                # Set up environment for future logins
                bashrc_setup = """
                echo 'source /opt/venv/bin/activate' >> /root/.bashrc
                echo 'cd /workspace' >> /root/.bashrc
                """
                droplet_info.run_command(bashrc_setup)

                return True
            else:
                console.print(
                    f"[red]Virtual environment creation failed: {result.get('stderr', '')}[/red]"
                )
                console.print("[yellow]Will try to continue anyway...[/yellow]")
                return False

        except Exception as e:
            console.print(f"[red]Failed to ensure host system ready: {e}[/red]")
            return False
