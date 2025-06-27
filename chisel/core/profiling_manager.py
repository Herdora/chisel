"""Profile manager for orchestrating GPU profiling workflows."""

# TODO: Have the name of profile output be <target>-<vendor>-<gpu>-<time>-<date>

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console

from chisel.core.droplet_service import DropletService, Droplet
from .types.gpu_profiles import GPUType

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
        target: str,
        gpu_type: GPUType,
        output_dir: Optional[str] = None,
        rocprofv3_flag: Optional[str] = None,
        rocprof_compute_flag: Optional[str] = None,
        nsys_flag: Optional[str] = None,
        ncompute_flag: Optional[str] = None,
    ) -> ProfilingResults:
        """
        Execute a complete profiling workflow.

        Args:
            target: File path or command to profile
            gpu_type: GPU type override - "nvidia-h100" or "nvidia-l40s" for NVIDIA (optional)
            output_dir: Custom output directory for results (optional)
            rocprofv3_flag: Full command to run with rocprofv3 (AMD)
            rocprof_compute_flag: Full command to run with rocprof-compute (AMD)
            nsys_flag: Full command to run with nsys (NVIDIA)
            ncompute_flag: Full command to run with ncu (NVIDIA)

        Returns:
            ProfilingResults with profiling data and summary
        """

        try:
            console.print(f"[cyan]Ensuring {gpu_type.value} droplet is ready...[/cyan]")
            droplet_info = self.droplet_service.get_or_create_droplet_by_type(gpu_type)
            console.print(f"[green]Droplet {droplet_info.name} is ready[/green]")

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
                    droplet_info,
                    target_info,
                    output_path,
                    rocprof_compute_flag,
                )
                all_results.append(result)
            if nsys_flag:
                result = self.run_nsys(droplet_info, target_info, output_path, nsys_flag)
                all_results.append(result)
            if ncompute_flag:
                result = self.run_ncompute(droplet_info, target_info, output_path, ncompute_flag)
                all_results.append(result)

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
        droplet_info: Droplet,
        target: TargetInfo,
        local_output_dir: Path,
        extra_flags: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run rocprofv3 on the droplet."""

        # Validate ROCm and ensure rocprofv3 is available
        self._ensure_rocprofv3(droplet_info)

        # Check if this is a Python file and ensure ROCm Python libraries are available
        is_python = target.file_extension == ".py"
        if is_python:
            self._ensure_pytorch_rocm(droplet_info)

        # Use mounted volume path that container can access
        remote_profile_dir = "/mnt/share/chisel-rocprofv3"

        # Get the source file name and binary name
        source_file = target.file_path or Path(target.raw_target)
        source_name = source_file.name
        binary_name = source_file.stem
        remote_source = f"{remote_profile_dir}/{source_name}"
        remote_binary = f"{remote_profile_dir}/{binary_name}"

        EXPORT_LIB_CMD = "export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/"
        RESET_DIR_CMD = f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"
        CD_CMD = f"cd {remote_profile_dir}"

        if is_python:
            # For Python files, we don't need to build - just run with python
            # Note: Virtual environment is automatically activated in the container
            PROFILE_CMD = f"rocprofv3 -S --summary-output-file amd_profile_summary.txt {extra_flags or '--sys-trace'} -- python {remote_source}"

            # First reset directory, then sync file, then profile directly
            reset_cmd = f"{EXPORT_LIB_CMD} && {RESET_DIR_CMD}"
            result = droplet_info.run_container_command(reset_cmd)
            if result["exit_code"] != 0:
                raise RuntimeError(f"Failed to reset remote directory: {result['exit_code']}")

            # Now sync the file to the clean directory
            _ = self._sync_file(
                droplet_info,
                target.file_path or Path(target.raw_target),
                remote_profile_dir,
            )

            # Test if the Python script runs standalone first
            console.print("[cyan]Testing if Python script runs standalone...[/cyan]")
            test_cmd = f"{CD_CMD} && timeout 30 python {remote_source}"

            test_result = droplet_info.run_container_command(test_cmd, timeout=60)

            if test_result["exit_code"] != 0:
                console.print(
                    f"[red]Python script test failed with exit code {test_result['exit_code']}[/red]"
                )
                if test_result.get("stdout"):
                    console.print(f"[red]STDOUT:[/red] {test_result['stdout']}")
                if test_result.get("stderr"):
                    console.print(f"[red]STDERR:[/red] {test_result['stderr']}")
                console.print("[yellow]Script has issues - profiling may fail[/yellow]")
            else:
                console.print("[green]✓ Python script runs successfully[/green]")

            # Profile directly with python
            profile_cmd = f"{CD_CMD} && {PROFILE_CMD}"
            full_cmd = profile_cmd
            console.print(
                f"[cyan]Running AMD rocprofv3 on Python script with flags '{full_cmd}'[/cyan]"
            )
        else:
            # For compiled languages, build then profile
            BUILD_CMD = f"hipcc {remote_source} -o {remote_binary}"
            PROFILE_CMD = f"rocprofv3 -S --summary-output-file amd_profile_summary.txt {extra_flags or '--sys-trace'} -- {remote_binary}"

            # First reset directory, then sync file, then build and profile
            reset_cmd = f"{EXPORT_LIB_CMD} && {RESET_DIR_CMD}"
            result = droplet_info.run_container_command(reset_cmd)
            if result["exit_code"] != 0:
                raise RuntimeError(f"Failed to reset remote directory: {result['exit_code']}")

            # Now sync the file to the clean directory
            _ = self._sync_file(
                droplet_info,
                target.file_path or Path(target.raw_target),
                remote_profile_dir,
            )

            # Build and profile
            build_profile_cmd = f"{CD_CMD} && {BUILD_CMD} && {PROFILE_CMD}"
            full_cmd = build_profile_cmd
            console.print(f"[cyan]Running AMD rocprofv3 with flags '{full_cmd}'[/cyan]")

        rocprof_result = droplet_info.run_container_command(full_cmd, timeout=600)
        if rocprof_result["exit_code"] != 0:
            # Show detailed error information to help with debugging
            console.print(
                f"[red]rocprofv3 command failed with exit code {rocprof_result['exit_code']}[/red]"
            )
            if rocprof_result.get("stdout"):
                console.print(f"[red]STDOUT:[/red] {rocprof_result['stdout']}")
            if rocprof_result.get("stderr"):
                console.print(f"[red]STDERR:[/red] {rocprof_result['stderr']}")
            console.print(f"[red]Command that failed:[/red] {full_cmd}")
            raise RuntimeError(
                f"rocprofv3 profiling failed with exit code {rocprof_result['exit_code']}. "
                f"Check the output above for details."
            )

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
        droplet_info: Droplet,
        target: TargetInfo,
        local_output_dir: Path,
        extra_flags: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run rocprof-compute on the droplet."""
        # TODO: Implement rocprof-compute when ready

        console.print("[yellow]rocprof-compute support not yet implemented[/yellow]")
        raise RuntimeError("rocprof-compute is not yet supported")

    def run_nsys(
        self,
        droplet_info: Droplet,
        target: TargetInfo,
        local_output_dir: Path,
        extra_flags: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run nsys on the droplet."""
        self._ensure_nvidia_profilers(droplet_info)

        # If this is a Python file, ensure PyTorch is available for GPU libraries
        is_python_file = target.is_source_file and target.file_extension == ".py"
        if is_python_file:
            self._ensure_pytorch(droplet_info)

        # Use mounted volume path that container can access
        remote_profile_dir = "/mnt/share/chisel-nsys"

        # Check if this is a Python file that needs to be synced
        if is_python_file:
            # For Python files, sync the file and construct the command
            source_file = target.file_path or Path(target.raw_target)
            remote_script = f"{remote_profile_dir}/{source_file.name}"
            command_to_run = f"python {remote_script}"

            # Sync the Python file to remote
            reset_cmd = f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"
            result = droplet_info.run_container_command(reset_cmd)
            if result["exit_code"] != 0:
                raise RuntimeError(f"Failed to reset remote directory: {result['exit_code']}")

            _ = self._sync_file(droplet_info, source_file, remote_profile_dir)
        else:
            # For other commands (non-Python files or direct commands), use as-is
            command_to_run = target.raw_target

        # Combine setup and profiling into a single atomic command to ensure cd works properly
        def make_full_cmd(remote_profile_dir: str, extra_flags: str):
            if is_python_file:
                # For Python files, we already set up the directory above
                return f"cd {remote_profile_dir} && nsys profile {extra_flags or '--stats=true --force-overwrite=true'} -o nvidia_profile -- {command_to_run}"
            else:
                # For other commands, set up directory as part of the command
                return f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir} && cd {remote_profile_dir} && nsys profile {extra_flags or '--stats=true --force-overwrite=true'} -o nvidia_profile -- {command_to_run}"

        full_cmd = make_full_cmd(
            remote_profile_dir, extra_flags or "--stats=true --force-overwrite=true"
        )
        console.print(f"[cyan]Running NVIDIA nsys with flags '{full_cmd}'[/cyan]")
        nsys_result = droplet_info.run_container_command(full_cmd, timeout=600)
        if nsys_result["exit_code"] != 0:
            raise RuntimeError(f"nsys profiling failed with exit code {nsys_result['exit_code']}")

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
        droplet_info: Droplet,
        target: TargetInfo,
        local_output_dir: Path,
        extra_flags: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run ncu (nsight-compute) on the droplet."""
        self._ensure_nvidia_profilers(droplet_info)

        # If this is a Python file, ensure PyTorch is available for GPU libraries
        is_python_file = target.is_source_file and target.file_extension == ".py"
        if is_python_file:
            self._ensure_pytorch(droplet_info)

        # Use mounted volume path that container can access
        remote_profile_dir = "/mnt/share/chisel-ncompute"

        # Check if this is a Python file that needs to be synced
        if is_python_file:
            # For Python files, sync the file and construct the command
            source_file = target.file_path or Path(target.raw_target)
            remote_script = f"{remote_profile_dir}/{source_file.name}"
            command_to_run = f"python {remote_script}"

            # Sync the Python file to remote
            reset_cmd = f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"
            result = droplet_info.run_container_command(reset_cmd)
            if result["exit_code"] != 0:
                raise RuntimeError(f"Failed to reset remote directory: {result['exit_code']}")

            _ = self._sync_file(droplet_info, source_file, remote_profile_dir)
        else:
            # For other commands (non-Python files or direct commands), use as-is
            command_to_run = target.raw_target

        # Combine setup and profiling into a single atomic command to ensure cd works properly
        def make_full_cmd(remote_profile_dir: str, extra_flags: str):
            if is_python_file:
                # For Python files, we already set up the directory above
                return f"cd {remote_profile_dir} && ncu {extra_flags or '--set full --force-overwrite'} -o nvidia_ncompute_profile -- {command_to_run}"
            else:
                # For other commands, set up directory as part of the command
                return f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir} && cd {remote_profile_dir} && ncu {extra_flags or '--set full --force-overwrite'} -o nvidia_ncompute_profile -- {command_to_run}"

        full_cmd = make_full_cmd(remote_profile_dir, extra_flags or "--set full --force-overwrite")
        console.print(f"[cyan]Running NVIDIA ncu with flags '{full_cmd}'[/cyan]")
        ncu_result = droplet_info.run_container_command(full_cmd, timeout=600)
        if ncu_result["exit_code"] != 0:
            raise RuntimeError(f"ncu profiling failed with exit code {ncu_result['exit_code']}")

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
            # Check if both profilers are already available
            check_cmd = "which ncu && ncu --version && which nsys && nsys --version"
            result = droplet_info.run_container_command(check_cmd)

            if result["exit_code"] == 0:
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

            install_result = droplet_info.run_container_command(install_cmd, timeout=700)

            if install_result["exit_code"] != 0:
                raise RuntimeError(
                    "Failed to install NVIDIA profilers. This may be due to package repository issues or network connectivity."
                )

            # Verify both installations
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

            console.print("[green]✓ NVIDIA profilers installed successfully (ncu + nsys)[/green]")

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
            # First validate that ROCm is properly installed
            self._validate_rocm_installation(droplet_info)

            # Check if rocprofv3 is already available
            check_cmd = "which rocprofv3 && echo 'rocprofv3 available'"
            result = droplet_info.run_container_command(check_cmd)

            if result["exit_code"] == 0:
                console.print("[green]✓ rocprofv3 already available[/green]")
                return

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
