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

        # We'll use a separate state file for the new profiling system
        self.state = ProfilingState()

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

            target_info = self._analyze_target(target)

            if target_info.is_source_file and target_info.file_path:
                console.print(
                    f"[cyan]Syncing {target_info.file_path.name} to remote server...[/cyan]"
                )
                synced_file_path = self._sync_file(droplet_info, target_info.file_path)
                base_command = self._build_command(vendor, target_info, synced_file_path)
                command = base_command
            else:
                command = target

            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                timestamp = datetime.now().strftime("%H%M%S-%Y%m%d")
                output_path = Path(f"./{CHISEL_PROFILING_DIR_NAME}-{timestamp}")
                output_path.mkdir(parents=True, exist_ok=True)

            all_results = []
            if rocprofv3_flag:
                result = self._run_rocprofv3(droplet_info, command, output_path, rocprofv3_flag)
                all_results.append(result)
            if rocprof_compute_flag:
                result = self._run_rocprof_compute(
                    droplet_info, command, output_path, rocprof_compute_flag
                )
                all_results.append(result)
            if nsys_flag:
                result = self._run_nsys(droplet_info, command, output_path, nsys_flag)
                all_results.append(result)
            if ncompute_flag:
                result = self._run_ncompute(droplet_info, command, output_path, ncompute_flag)
                all_results.append(result)

            elapsed_hours = (time.time() - start_time) / 3600
            hourly_rate = 4.89 if vendor == "nvidia" else 1.99
            cost_estimate = elapsed_hours * hourly_rate

            self.state.update_activity(resolved_gpu_type)

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
        command: str,
        local_output_dir: Path,
        extra_flags: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run rocprofv3 on the droplet."""
        ssh_manager = SSHManager()
        self._ensure_rocprofv3(droplet_info)

        remote_profile_dir = "/tmp/chisel-rocprofv3"

        def make_profile_setup_cmd(remote_profile_dir: str):
            return f"export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ && rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir} && cd {remote_profile_dir}"

        profile_setup_results = ssh_manager.run(
            make_profile_setup_cmd(remote_profile_dir),
            droplet_info["gpu_type"],
        )
        if profile_setup_results != 0:
            raise RuntimeError(f"Failed to setup remote profile directory: {profile_setup_results}")

        def make_rocprof_cmd(remote_profile_dir: str, extra_flags: str):
            return f"rocprofv3 -S --summary-output-file amd_profile_summary.txt {extra_flags or '--sys-trace'} -- {command}"

        rocprof_cmd = make_rocprof_cmd(remote_profile_dir, extra_flags or "--sys-trace")
        console.print(f"[cyan]Running AMD rocprofv3 with flags '{rocprof_cmd}'[/cyan]")
        rocprof_exit_code = ssh_manager.run(rocprof_cmd, droplet_info["gpu_type"])
        if rocprof_exit_code != 0:
            raise RuntimeError(f"rocprofv3 profiling failed with exit code {rocprof_exit_code}")

        rocprof_files = self._download_amd_att_results(
            droplet_info, remote_profile_dir, local_output_dir
        )
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

        def make_profile_setup_cmd(remote_profile_dir: str):
            return f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir} && cd {remote_profile_dir}"

        profile_setup_results = ssh_manager.run(
            make_profile_setup_cmd(remote_profile_dir),
            droplet_info["gpu_type"],
        )
        if profile_setup_results != 0:
            raise RuntimeError(f"Failed to setup remote profile directory: {profile_setup_results}")

        def make_nsys_cmd(remote_profile_dir: str, extra_flags: str):
            return f"nsys profile {extra_flags or '--stats=true --force-overwrite=true'} -o nvidia_profile -- {command}"

        nsys_cmd = make_nsys_cmd(
            remote_profile_dir, extra_flags or "--stats=true --force-overwrite=true"
        )
        console.print(f"[cyan]Running NVIDIA nsys with flags '{nsys_cmd}'[/cyan]")
        nsys_exit_code = ssh_manager.run(nsys_cmd, droplet_info["gpu_type"])
        if nsys_exit_code != 0:
            raise RuntimeError(f"nsys profiling failed with exit code {nsys_exit_code}")

        nvidia_files = self._download_nvidia_results(
            droplet_info, remote_profile_dir, local_output_dir
        )

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

        def make_profile_setup_cmd(remote_profile_dir: str):
            return f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir} && cd {remote_profile_dir}"

        profile_setup_results = ssh_manager.run(
            make_profile_setup_cmd(remote_profile_dir),
            droplet_info["gpu_type"],
        )
        if profile_setup_results != 0:
            raise RuntimeError(f"Failed to setup remote profile directory: {profile_setup_results}")

        def make_ncu_cmd(remote_profile_dir: str, extra_flags: str):
            return f"ncu {extra_flags or '--set full --force-overwrite'} -o nvidia_ncompute_profile -- {command}"

        ncu_cmd = make_ncu_cmd(remote_profile_dir, extra_flags or "--set full --force-overwrite")
        console.print(f"[cyan]Running NVIDIA ncu with flags '{ncu_cmd}'[/cyan]")
        ncu_exit_code = ssh_manager.run(ncu_cmd, droplet_info["gpu_type"])
        if ncu_exit_code != 0:
            raise RuntimeError(f"ncu profiling failed with exit code {ncu_exit_code}")

        nvidia_files = self._download_nvidia_results(
            droplet_info, remote_profile_dir, local_output_dir
        )

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
        """Sync a file to the droplet with proper temp directory setup."""
        ssh_manager = SSHManager()

        # Create a unique temp directory for this profiling session
        import time

        session_id = int(time.time())
        temp_dir = f"/tmp/chisel_session_{session_id}"

        # First, ensure the temp directory exists on the remote server
        setup_cmd = f"mkdir -p {temp_dir}"
        exit_code = ssh_manager.run(setup_cmd, droplet_info["gpu_type"])
        if exit_code != 0:
            raise RuntimeError(f"Failed to create temp directory {temp_dir} on remote server")

        # Sync the file to the temp directory
        success = ssh_manager.sync(str(file_path), f"{temp_dir}/", droplet_info["gpu_type"])

        if not success:
            raise RuntimeError(
                f"Failed to sync {file_path} to {temp_dir}. Ensure the file exists and is accessible."
            )

        # Make the file executable
        chmod_cmd = f"chmod +x {temp_dir}/{file_path.name}"
        exit_code = ssh_manager.run(chmod_cmd, droplet_info["gpu_type"])
        if exit_code != 0:
            console.print("[yellow]Warning: Failed to make file executable[/yellow]")

        # Update the file path to point to the synced location
        file_path = Path(f"{temp_dir}/{file_path.name}")

        console.print(f"[green]✓ File synced to {file_path} on remote server[/green]")

        return str(file_path)

    def _build_command(
        self, vendor: str, target_info: TargetInfo, synced_file_path: Optional[str] = None
    ) -> str:
        """Build the compilation and execution command."""
        if not target_info.file_path:
            return target_info.raw_target

        # Use synced file path if provided, otherwise use default /tmp location
        if synced_file_path:
            remote_source = synced_file_path
            # Extract directory and binary name from synced path
            synced_path = Path(synced_file_path)
            binary_name = synced_path.stem
            remote_binary = f"{synced_path.parent}/{binary_name}"
        else:
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
        target_info: Optional[TargetInfo] = None,
        rocprofv3_cmd: Optional[str] = None,
        rocprof_compute_cmd: Optional[str] = None,
        nsys_cmd: Optional[str] = None,
        ncompute_cmd: Optional[str] = None,
        custom_output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run the profiler on the droplet."""
        ssh_manager = SSHManager()

        # Create output directory with simple timestamp naming
        now = datetime.now()
        timestamp = now.strftime("%H%M%S-%Y%m%d")
        output_dir = custom_output_dir or Path(f"./{CHISEL_PROFILING_DIR_NAME}/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        combined_summary = {"profile_files": [], "profilers_run": []}

        if vendor == "amd":
            # Run enabled AMD profilers
            if rocprofv3_cmd:
                try:
                    result = self._run_amd_profiler(
                        droplet_info,
                        command,
                        output_dir,
                        rocprofv3_cmd,
                        rocprof_compute_cmd,
                    )
                    all_results.append(result)
                    combined_summary["profilers_run"].append("rocprofv3")
                    combined_summary["profile_files"].extend(
                        result["summary"].get("profile_files", [])
                    )
                except Exception as e:
                    console.print(f"[yellow]AMD rocprofv3 profiling failed: {e}[/yellow]")
            elif rocprof_compute_cmd:
                # TODO: Implement rocprof-compute support
                console.print("[yellow]rocprof-compute support not yet implemented[/yellow]")
                raise RuntimeError("rocprof-compute is not yet supported")
            else:
                # TODO: phase out this code
                # Original behavior for backward compatibility
                # Setup remote profiling environment
                remote_profile_dir = "/tmp/chisel_amd_profile"
                profile_setup = f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"

                # For AMD, we need to separate compilation from profiling
                if " && " in command:
                    # Split compilation and execution
                    compile_part, execute_part = command.split(" && ", 1)
                    # Execute compilation first, then profile with rocprofv3 with summary output
                    rocprof_cmd = f"{compile_part} && export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ && rocprofv3 -S --summary-output-file amd_profile_summary.txt --sys-trace -- {execute_part}"
                else:
                    # Just profile the single command with summary output
                    rocprof_cmd = f"export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ && rocprofv3 -S --summary-output-file amd_profile_summary.txt --sys-trace -- {command}"

                # Full profiling command
                full_cmd = f"{profile_setup} && {rocprof_cmd}"

                console.print(f"[cyan]Running AMD rocprofv3 with ATT traces: {command}[/cyan]")

                # Execute profiling command
                exit_code = ssh_manager.run(full_cmd, droplet_info["gpu_type"])

                if exit_code != 0:
                    raise RuntimeError(f"rocprofv3 profiling failed with exit code {exit_code}")

                # Download results
                rocprof_files = self._download_amd_att_results(
                    droplet_info, remote_profile_dir, output_dir
                )

                # Create summary
                summary = {
                    "profile_files": rocprof_files,
                    "summary_file": rocprof_files[0] if rocprof_files else None,
                    "profile_type": "rocprofv3",
                    "message": "AMD rocprofv3 profiling completed. Generated profile summary.",
                }

                # Cleanup remote files
                self._cleanup_amd_remote(droplet_info, remote_profile_dir)

                return {
                    "output_dir": output_dir,
                    "stdout": "AMD rocprofv3 profiling completed successfully",
                    "stderr": "",
                    "summary": summary,
                }

            # Download results
            if rocprof_compute_cmd:
                try:
                    # TODO: Implement rocprof-compute when ready
                    console.print("[yellow]rocprof-compute support not yet implemented[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]AMD rocprof-compute profiling failed: {e}[/yellow]")

            if not all_results:
                # Fallback to legacy behavior
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
            # Run enabled NVIDIA profilers
            if nsys_cmd or ncompute_cmd:
                try:
                    result = self._run_nvidia_profiler(
                        droplet_info, command, output_dir, nsys_cmd, ncompute_cmd
                    )
                    all_results.append(result)
                    if nsys_cmd:
                        combined_summary["profilers_run"].append("nsys")
                    if ncompute_cmd:
                        combined_summary["profilers_run"].append("ncompute")
                    combined_summary["profile_files"].extend(
                        result["summary"].get("profile_files", [])
                    )
                except Exception as e:
                    console.print(f"[yellow]NVIDIA profiling failed: {e}[/yellow]")

            if not all_results:
                # Fallback to basic execution
                console.print("[yellow]Falling back to basic execution...[/yellow]")
                exit_code = ssh_manager.run(command, droplet_info["gpu_type"])
                return {
                    "output_dir": output_dir,
                    "stdout": f"Command executed with exit code: {exit_code}",
                    "stderr": "",
                    "summary": {},
                }

        # Return combined results
        if all_results:
            # Use the first successful result as base, but update summary
            base_result = all_results[0]
            base_result["summary"].update(
                {
                    "profilers_run": combined_summary["profilers_run"],
                    "profile_files": combined_summary["profile_files"],
                    "summary_file": combined_summary["profile_files"][0]
                    if combined_summary["profile_files"]
                    else None,
                    "message": f"Profiling completed with {', '.join(combined_summary['profilers_run'])}",
                }
            )
            return base_result
        else:
            return {
                "output_dir": output_dir,
                "stdout": "No profilers succeeded",
                "stderr": "All profilers failed",
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
        self,
        droplet_info: Dict[str, Any],
        command: str,
        output_dir: Path,
        nsys_cmd: Optional[str] = None,
        ncompute_cmd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run NVIDIA profilers (nsight-compute + nsight-systems) on the droplet."""
        ssh_manager = SSHManager()

        # TODO: phase out this code

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

        # Handle direct profiler commands
        if nsys_cmd or ncompute_cmd:
            # If direct commands are provided, use them
            cmds = []
            if ncompute_cmd:
                cmds.append(
                    f"ncu --set full --target-processes all --export {profile_filename}_%p.ncu-rep {ncompute_cmd}"
                )
            if nsys_cmd:
                cmds.append(f"nsys profile --output={profile_filename}.nsys-rep {nsys_cmd}")

            if len(cmds) == 2:
                # Run both profilers
                full_cmd = f"{profile_setup} && {cmds[0]} && {cmds[1]}"
            else:
                # Run single profiler
                full_cmd = f"{profile_setup} && {cmds[0]}"
        else:
            # Original behavior for backward compatibility
            # For NVIDIA, we need to separate compilation from profiling
            # The command might be a compile+run, so we need to handle this properly
            if " && " in command:
                # Split compilation and execution
                compile_part, execute_part = command.split(" && ", 1)
                # Execute compilation first, then profile with both tools
                ncu_cmd = f"{compile_part} && ncu --set full --target-processes all --export {profile_filename}_%p.ncu-rep {execute_part}"
                nsys_cmd_built = f"nsys profile --output={profile_filename}.nsys-rep {execute_part}"
            else:
                # Just profile the single command with both tools
                ncu_cmd = f"ncu --set full --target-processes all --export {profile_filename}_%p.ncu-rep {command}"
                nsys_cmd_built = f"nsys profile --output={profile_filename}.nsys-rep {command}"

            # Run both profilers - ncu first (more likely to fail), then nsys
            full_cmd = f"{profile_setup} && {ncu_cmd} && {nsys_cmd_built}"

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

        # Create basic summary with text summary
        summary = {
            "profile_files": profile_files,
            "summary_file": profile_files[0] if profile_files else None,
            "message": "NVIDIA profiling completed. Generated profile summary.",
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
        rocprofv3_cmd: Optional[str] = None,
        rocprof_compute_cmd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run AMD rocprofv3 profiler with ATT traces on the droplet."""
        ssh_manager = SSHManager()

        # TODO: phase out this code

        # Ensure rocprofv3 is available
        self._ensure_rocprofv3(droplet_info)

        # Setup remote profiling environment
        remote_profile_dir = "/tmp/chisel_amd_profile"

        # Build rocprofv3 profiling command
        profile_setup = f"rm -rf {remote_profile_dir} && mkdir -p {remote_profile_dir}"

        # If direct command is provided, use it
        if rocprofv3_cmd:
            # Extract extra flags from rocprofv3_cmd
            # rocprofv3_cmd format: "target extra_flags" or just "target"
            parts = rocprofv3_cmd.split(" ", 1)
            target_cmd = parts[0]
            extra_flags = parts[1] if len(parts) > 1 else ""

            # Check if target_cmd is a source file that needs compilation
            if (
                target_cmd.endswith(".hip")
                or target_cmd.endswith(".cu")
                or target_cmd.endswith(".cpp")
                or target_cmd.endswith(".c")
            ):
                # Compile the source file first
                if target_cmd.endswith(".hip"):
                    compile_cmd = f"hipcc {target_cmd} -o {target_cmd[:-4]}"
                elif target_cmd.endswith(".cu"):
                    compile_cmd = f"nvcc -O3 -lineinfo {target_cmd} -o {target_cmd[:-3]}"
                else:
                    compile_cmd = f"gcc {target_cmd} -o {target_cmd[:-4] if target_cmd.endswith('.cpp') else target_cmd[:-2]}"

                # Use compiled binary for profiling
                binary_cmd = (
                    target_cmd[:-4]
                    if target_cmd.endswith(".hip")
                    else (
                        target_cmd[:-3]
                        if target_cmd.endswith(".cu")
                        else (target_cmd[:-4] if target_cmd.endswith(".cpp") else target_cmd[:-2])
                    )
                )
                rocprof_cmd = f"{compile_cmd} && export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ && rocprofv3 -S --summary-output-file amd_profile_summary.txt {extra_flags} -- {binary_cmd}"
            else:
                # Use the command as-is (already compiled or script)
                rocprof_cmd = f"export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ && rocprofv3 -S --summary-output-file amd_profile_summary.txt {extra_flags} -- {target_cmd}"

        # Full profiling command
        full_cmd = f"{profile_setup} && {rocprof_cmd}"

        console.print(f"[cyan]Running AMD rocprofv3 with ATT traces: {command}[/cyan]")

        # Execute profiling command
        exit_code = ssh_manager.run(full_cmd, droplet_info["gpu_type"])

        if exit_code != 0:
            raise RuntimeError(f"rocprofv3 profiling failed with exit code {exit_code}")

        # Download results
        rocprof_files = self._download_amd_att_results(droplet_info, remote_profile_dir, output_dir)

        # Create summary
        summary = {
            "profile_files": rocprof_files,
            "summary_file": rocprof_files[0] if rocprof_files else None,
            "profile_type": "rocprofv3",
            "message": "AMD rocprofv3 profiling completed. Generated profile summary.",
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
        local_output_dir: Path,
    ) -> list:
        import subprocess

        ssh_manager = SSHManager()
        ip = droplet_info["ip"]
        console.print("[cyan]Downloading AMD profiling results...[/cyan]")
        # rocprofv3 creates files with session prefix inside a subdirectory, we need to find the actual file
        find_cmd = f"find {remote_dir} -name '*amd_profile_summary.txt*' -type f | head -1"
        exit_code = ssh_manager.run(find_cmd, droplet_info["gpu_type"])

        if exit_code != 0:
            console.print("[yellow]Warning: Could not find AMD summary file[/yellow]")
            return []

        # Get the actual file path by running the find command and capturing output
        import subprocess as sp

        find_result = sp.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"root@{ip}", find_cmd],
            capture_output=True,
            text=True,
        )

        if find_result.returncode != 0 or not find_result.stdout.strip():
            console.print("[yellow]Warning: AMD summary file not found[/yellow]")
            return []

        remote_summary_path = find_result.stdout.strip()
        local_summary_path = local_output_dir / "amd_profile_summary.txt"
        scp_cmd = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            f"root@{ip}:{remote_summary_path}",
            str(local_summary_path),
        ]
        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                console.print(
                    f"[yellow]Warning: Failed to download AMD ATT results: {result.stderr}[/yellow]"
                )
                return []
            if not local_summary_path.exists() or local_summary_path.stat().st_size == 0:
                console.print("[yellow]Warning: Downloaded summary is empty or missing[/yellow]")
                return []
            console.print(f"[green]✓ AMD profile summary saved to {local_summary_path}[/green]")
            return ["amd_profile_summary.txt"]
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

    def _download_nvidia_results(
        self, droplet_info: Dict[str, Any], remote_dir: str, local_output_dir: Path
    ) -> list:
        import subprocess

        ssh_manager = SSHManager()
        ip = droplet_info["ip"]
        console.print("[cyan]Generating comprehensive profile summary...[/cyan]")
        convert_cmd = f"""cd {remote_dir} && \
for nsys_file in *.nsys-rep; do
    [ -f \"$nsys_file\" ] || continue
    {{
        echo "=== CUDA API Summary ==="
        nsys stats --report cuda_api_sum \"$nsys_file\" 2>/dev/null || true
        echo -e \"\\n=== GPU Kernel Summary ===\"
        nsys stats --report cuda_gpu_kern_sum \"$nsys_file\" 2>/dev/null || true
        echo -e \"\\n=== Memory Operations Summary ===\"
        nsys stats --report cuda_gpu_mem_time_sum \"$nsys_file\" 2>/dev/null || true
        echo -e \"\\n=== CUDA GPU Summary (Combined) ===\"
        nsys stats --report cuda_api_gpu_sum \"$nsys_file\" 2>/dev/null || true
    }} > nvidia_profile_summary.txt
    break
done"""
        ssh_manager.run(convert_cmd, droplet_info["gpu_type"])
        console.print("[cyan]Downloading NVIDIA profiling results...[/cyan]")
        local_summary_path = local_output_dir / "nvidia_profile_summary.txt"
        scp_cmd = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            f"root@{ip}:{remote_dir}/nvidia_profile_summary.txt",
            str(local_summary_path),
        ]
        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                console.print(
                    f"[yellow]Warning: Failed to download NVIDIA profile results: {result.stderr}[/yellow]"
                )
                return []
            if not local_summary_path.exists() or local_summary_path.stat().st_size == 0:
                console.print("[yellow]Warning: Downloaded summary is empty or missing[/yellow]")
                return []
            console.print(f"[green]✓ NVIDIA profile summary saved to {local_summary_path}[/green]")
            return ["nvidia_profile_summary.txt"]
        except subprocess.TimeoutExpired:
            console.print("[yellow]Warning: Download timed out[/yellow]")
            return []
        except Exception as e:
            console.print(f"[yellow]Warning: Unexpected error during download: {e}[/yellow]")
            return []

    def _cleanup_nvidia_remote(self, droplet_info: Dict[str, Any], remote_dir: str):
        """Clean up remote NVIDIA profiling files."""
        ssh_manager = SSHManager()

        cleanup_cmd = f"rm -rf {remote_dir}"
        ssh_manager.run(cleanup_cmd, droplet_info["gpu_type"])

        console.print("[green]✓ Remote cleanup completed[/green]")
