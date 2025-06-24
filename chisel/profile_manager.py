"""Profile manager for orchestrating GPU profiling workflows."""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

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
            console.print(f"\n[green]✓ Profiling completed successfully[/green]")
            console.print(f"[cyan]Results saved to:[/cyan] {self.output_dir}")
            
            # Show cost estimate
            console.print(f"[yellow]Estimated cost:[/yellow] ${self.cost_estimate:.2f}")
            
            # Show top kernels if available
            if "top_kernels" in self.summary:
                console.print("\n[cyan]Top GPU Kernels:[/cyan]")
                for i, kernel in enumerate(self.summary["top_kernels"][:5], 1):
                    console.print(f"  {i}. {kernel['name'][:50]:<50} {kernel['time_ms']:8.3f} ms")
        else:
            console.print(f"\n[red]✗ Profiling failed[/red]")
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
            console.print(f"[cyan]Running profiler...[/cyan]")
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
                cost_estimate=cost_estimate
            )
            
        except Exception as e:
            console.print(f"[red]Error during profiling: {e}[/red]")
            return ProfileResult(
                success=False,
                output_dir=Path("./chisel-results/failed"),
                stdout="",
                stderr=str(e),
                summary={},
                cost_estimate=0.0
            )
    
    def _ensure_droplet(self, gpu_type: str) -> Dict[str, any]:
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
            "created_at": droplet["created_at"]
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
                ".c": "gcc"
            }
            
            return TargetInfo(
                raw_target=target,
                is_source_file=True,
                file_path=target_path.resolve(),
                file_extension=extension,
                compiler=compiler_map.get(extension, "gcc")
            )
        
        # It's a command
        return TargetInfo(
            raw_target=target,
            is_source_file=False
        )
    
    def _sync_file(self, droplet_info: Dict[str, any], file_path: Path):
        """Sync a file to the droplet."""
        ssh_manager = SSHManager()
        
        # For the new system, we'll sync to /tmp for simplicity
        success = ssh_manager.sync(
            str(file_path), 
            "/tmp/",
            droplet_info["gpu_type"]
        )
        
        if not success:
            raise RuntimeError(f"Failed to sync {file_path}")
    
    def _build_command(self, vendor: str, target_info: TargetInfo) -> str:
        """Build the compilation and execution command."""
        remote_source = f"/tmp/{target_info.file_path.name}"
        binary_name = target_info.file_path.stem
        remote_binary = f"/tmp/{binary_name}"
        
        if vendor == "nvidia":
            if target_info.compiler == "nvcc":
                compile_cmd = f"nvcc {remote_source} -o {remote_binary}"
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
    
    def _run_profiler(self, droplet_info: Dict[str, any], vendor: str, command: str) -> Dict[str, any]:
        """Run the profiler on the droplet."""
        ssh_manager = SSHManager()
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(f"./chisel-results/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if vendor == "amd":
            # Use existing profile method for AMD
            result_path = ssh_manager.profile(
                command,
                droplet_info["gpu_type"],
                trace="hip,hsa",
                output_dir=str(output_dir),
                open_result=False
            )
            
            # Parse results
            summary = self._parse_amd_results(output_dir)
            
            return {
                "output_dir": output_dir,
                "stdout": "",
                "stderr": "",
                "summary": summary
            }
        else:
            # For NVIDIA, just compile and run for now
            console.print("[yellow]Note: NVIDIA profiling not yet implemented. Running command only.[/yellow]")
            
            exit_code = ssh_manager.run(command, droplet_info["gpu_type"])
            
            return {
                "output_dir": output_dir,
                "stdout": f"Command executed with exit code: {exit_code}",
                "stderr": "",
                "summary": {}
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
        import csv
        
        # Try JSON first
        json_file = profile_dir / "results.json"
        if json_file.exists():
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    
                kernels = []
                for event in data.get('traceEvents', []):
                    if (event.get('ph') == 'X' and 
                        'pid' in event and 
                        event.get('pid') in [6, 7] and
                        'DurationNs' in event.get('args', {})):
                        
                        kernels.append({
                            'name': event.get('name', ''),
                            'time_ms': event['args']['DurationNs'] / 1_000_000
                        })
                
                # Sort by time
                kernels.sort(key=lambda x: x['time_ms'], reverse=True)
                summary['top_kernels'] = kernels[:10]
                
            except Exception as e:
                console.print(f"[yellow]Could not parse JSON results: {e}[/yellow]")
        
        return summary