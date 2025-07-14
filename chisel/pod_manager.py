"""
PodManager - Core functionality for managing RunPod instances.

Provides high-level operations for creating, managing, and tearing down
RunPod instances for profiling workloads.
"""

import subprocess
import atexit
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime

import runpod
from dotenv import load_dotenv

from .pod import Pod, get_pods

load_dotenv()


class PodManagerError(Exception):
    """Base exception for PodManager operations."""
    pass


class PodManager:
    """
    Manages RunPod instances for profiling workloads.
    
    Provides methods to:
    - Find or create pods with specific configurations
    - Sync code to remote instances
    - Execute commands via SSH
    - Clean up resources
    """
    
    def __init__(self, auto_cleanup: bool = True, ssh_key_path: Optional[str] = None):
        """Initialize PodManager."""
        self.pod: Optional[Pod] = None
        
        # Detect SSH key configuration
        self.ssh_key_path, self.ssh_public_key = self._detect_ssh_keys(ssh_key_path)
        
        # Ensure RunPod API key is available
        if not runpod.api_key:
            raise PodManagerError(
                "RUNPOD_API_KEY not found. Set it as environment variable or in .env file"
            )
        
        # Register cleanup handler
        if auto_cleanup:
            atexit.register(self.teardown)
    
    def ensure_pod(self) -> Pod:
        """
        Find an existing running pod.
        
        Returns:
            Pod instance that's ready for use
            
        Raises:
            PodManagerError: If no suitable pod found
        """
        existing_pod = self._find_existing_pod()
        if existing_pod:
            print(f"✓ Found existing pod: {existing_pod.name} ({existing_pod.id})")
            self.pod = existing_pod
            return existing_pod
        
        raise PodManagerError("No suitable running pod found. Please start a pod manually.")
    
    def sync_code(self, local_path: str, remote_path: str = "/workspace") -> None:
        """
        Sync local code to the remote pod.
        
        Args:
            local_path: Local file or directory path
            remote_path: Remote destination path
            
        Raises:
            PodManagerError: If sync fails
        """
        if not self.pod or not self.pod.can_ssh():
            raise PodManagerError("No pod available or SSH not accessible")
        
        local_path_obj = Path(local_path)
        if not local_path_obj.exists():
            raise PodManagerError(f"Local path does not exist: {local_path}")
        
        # Use rsync for efficient file transfer
        ssh_details = self.pod.get_ssh_details()
        if not ssh_details:
            raise PodManagerError("Could not get SSH details from pod")
        
        ssh_details = self.pod.get_ssh_details()
        if not ssh_details:
            raise PodManagerError("Could not get SSH details from pod")
        
        scp_cmd = [
            "scp",
            "-i", self.ssh_key_path,
            "-P", str(ssh_details['port']),  # Note: scp uses -P (capital)
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            str(local_path_obj),
            f"{ssh_details['user']}@{ssh_details['host']}:{remote_path}/"
        ]
        
        print(f"📤 Syncing {local_path} to {remote_path}")
        try:
            subprocess.run(scp_cmd, capture_output=True, text=True, check=True)
            print("✓ Sync completed successfully")
        except subprocess.CalledProcessError as e:
            raise PodManagerError(f"Sync failed: {e.stderr}")
    
    def exec(self, command: str, capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Execute a command on the remote pod via SSH.
        
        Args:
            command: Command to execute
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            CompletedProcess with result
            
        Raises:
            PodManagerError: If execution fails
        """
        if not self.pod or not self.pod.can_ssh():
            raise PodManagerError("No pod available or SSH not accessible")
        
        ssh_details = self.pod.get_ssh_details()
        if not ssh_details:
            raise PodManagerError("Could not get SSH details from pod")
        
        # Use direct TCP connection format: ssh -p PORT root@IP
        ssh_details = self.pod.get_ssh_details()
        if not ssh_details:
            raise PodManagerError("Could not get SSH details from pod")
        
        ssh_cmd = [
            "ssh", 
            "-i", self.ssh_key_path,
            "-p", str(ssh_details['port']),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"{ssh_details['user']}@{ssh_details['host']}",
            command
        ]
        
        try:
            result = subprocess.run(ssh_cmd, capture_output=capture_output, text=True)
            return result
        except Exception as e:
            raise PodManagerError(f"Command execution failed: {e}")
    
    def exec_with_profiling(self, script_path: str, target: str = "amd", capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Execute a Python script with profiling enabled.
        
        Wraps the script execution with:
        - torch.profiler for PyTorch operations
        - rocprof for AMD GPU kernels (when target=amd)
        
        Stores all traces in /workspace/chisel_out
        
        Args:
            script_path: Path to Python script on remote pod
            target: Target platform ('amd' or 'cuda')
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            CompletedProcess with result
            
        Raises:
            PodManagerError: If execution fails
        """
        if not self.pod or not self.pod.can_ssh():
            raise PodManagerError("No pod available or SSH not accessible")
        
        # Create the profiling wrapper script
        wrapper_script = self._create_profiling_wrapper(script_path, target)
        
        # Upload the wrapper to the pod
        wrapper_path = "/tmp/chisel_profiling_wrapper.py"
        wrapper_upload_cmd = f"cat > {wrapper_path} << 'EOF'\n{wrapper_script}\nEOF"
        
        # Create wrapper script on pod
        result = self.exec(wrapper_upload_cmd)
        if result.returncode != 0:
            raise PodManagerError("Failed to create profiling wrapper script")
        
        # Setup output directory
        setup_cmd = "mkdir -p /workspace/chisel_out"
        self.exec(setup_cmd)
        
        # Execute with profiling
        if target.lower() == "amd":
            # Use rocprof for AMD GPU profiling
            cmd = f"cd /tmp && rocprof --hsa-trace --hip-trace -o /workspace/chisel_out/rocprof_trace.csv python {wrapper_path}"
        else:
            # For CUDA, just run the wrapper (torch.profiler will handle GPU profiling)
            cmd = f"cd /tmp && python {wrapper_path}"
        
        return self.exec(cmd, capture_output=capture_output)
    
    def _create_profiling_wrapper(self, script_path: str, target: str) -> str:
        """Create a profiling wrapper script that instruments the user's script."""
        script_name = Path(script_path).name
        
        wrapper = f'''#!/usr/bin/env python3
"""
Chisel Profiling Wrapper
Instruments user script with torch.profiler for comprehensive profiling.
"""

import sys
import os
from pathlib import Path

# Ensure we can import torch
try:
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
except ImportError as e:
    print(f"Error importing torch: {{e}}")
    print("Running script without torch profiling...")
    # Fallback: just run the script normally
    exec(open("{script_name}").read())
    sys.exit(0)

def main():
    """Run user script with torch.profiler instrumentation."""
    
    # Configure profiler activities based on target
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and "{target.lower()}" == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    print(f"🔥 Starting profiling session...")
    print(f"📊 Activities: {{', '.join(str(a) for a in activities)}}")
    print(f"📄 Script: {script_name}")
    print(f"💾 Output: /workspace/chisel_out/")
    
    # Setup profiler with comprehensive settings
    profiler_kwargs = {{
        'activities': activities,
        'record_shapes': True,
        'profile_memory': True,
        'with_stack': True,
        'with_flops': True,
        'with_modules': True,
        'on_trace_ready': torch.profiler.tensorboard_trace_handler('/workspace/chisel_out'),
        'schedule': torch.profiler.schedule(
            wait=1,    # Skip first step
            warmup=1,  # Warmup for 1 step  
            active=3,  # Record for 3 steps
            repeat=1   # Repeat the cycle 1 time
        )
    }}
    
    try:
        with profile(**profiler_kwargs) as prof:
            with record_function("user_script"):
                # Execute user script in profiling context
                print("\\n" + "="*50)
                print("🚀 EXECUTING USER SCRIPT")
                print("="*50 + "\\n")
                
                # Load and execute the user's script
                globals_dict = {{
                    '__name__': '__main__',
                    '__file__': '{script_name}',
                    'prof': prof  # Make profiler available to user script
                }}
                
                with open("{script_name}", 'r') as f:
                    script_content = f.read()
                
                exec(script_content, globals_dict)
                
                # Step the profiler if user didn't manually step
                prof.step()
        
        print("\\n" + "="*50)
        print("✅ PROFILING COMPLETED")
        print("="*50)
        
        # Export additional trace formats
        trace_path = "/workspace/chisel_out/torch_trace.json"
        prof.export_chrome_trace(trace_path)
        print(f"📄 Chrome trace saved: {{trace_path}}")
        
        # Print profiler summary to console
        print("\\n📊 Profiler Summary:")
        print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=20))
        
    except Exception as e:
        print(f"❌ Error during profiling: {{e}}")
        print("Running script without profiling...")
        # Fallback execution
        exec(open("{script_name}").read())
        raise

if __name__ == "__main__":
    main()
'''
        return wrapper
    
    def setup_profiling_env(self, target: str = "amd") -> None:
        """Setup profiling environment on first run."""
        if not self.pod:
            raise PodManagerError("No pod available")
        
        sentinel_file = f"~/.chisel_{target}_ready"
        
        # Check if already setup
        check_cmd = f"test -f {sentinel_file}"
        result = self.exec(check_cmd)
        if result.returncode == 0:
            print("✓ Profiling environment already configured")
            return
        
        print(f"🔧 Setting up {target.upper()} profiling environment...")
        
        if target.lower() == "amd":
            setup_script = f"""
            set -e
            echo "Installing ROCm profiling tools..."
            apt-get update -qq
            apt-get install -y -qq rocprofiler-dev roctracer-dev
            pip install --quiet torch torchvision
            touch {sentinel_file}
            echo "AMD profiling environment ready!"
            """
        else:  # CUDA
            setup_script = f"""
            set -e
            echo "Installing CUDA profiling tools..."
            pip install --quiet torch torchvision
            touch {sentinel_file}
            echo "CUDA profiling environment ready!"
            """
        
        try:
            result = self.exec(f"bash -c '{setup_script}'")
            if result.returncode == 0:
                print(f"✓ {target.upper()} profiling environment configured")
            else:
                print(f"⚠️ Setup completed with warnings")
        except Exception as e:
            print(f"⚠️ Setup failed: {e}")
            print("Profiling may still work with existing tools")

    def teardown(self) -> None:
        """Clean up the pod reference."""
        if self.pod:
            print(f"🔌 Disconnecting from pod {self.pod.id}")
            self.pod = None
    
    def _find_existing_pod(self) -> Optional[Pod]:
        """Find an existing running pod."""
        try:
            pods = get_pods()
            for pod in pods:
                if pod.is_running() and pod.can_ssh():
                    return pod
        except Exception as e:
            print(f"⚠️ Warning: Could not check existing pods: {e}")
        return None
    
    def fetch_artifacts(self, local_dir: str = "chisel_out") -> Path:
        """
        Fetch profiling artifacts from remote pod to local directory.
        
        Creates a unique timestamped subdirectory for each run to prevent overwriting.
        
        Args:
            local_dir: Local directory to store artifacts (default: "chisel_out")
            
        Returns:
            Path to local artifacts directory for this run
            
        Raises:
            PodManagerError: If fetch fails or no artifacts found
        """
        if not self.pod or not self.pod.can_ssh():
            raise PodManagerError("No pod available or SSH not accessible")
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = Path(local_dir)
        run_path = base_path / timestamp
        
        base_path.mkdir(exist_ok=True)
        run_path.mkdir(exist_ok=True)
        
        # Check if remote artifacts exist
        check_cmd = "test -d /workspace/chisel_out && ls -la /workspace/chisel_out"
        result = self.exec(check_cmd)
        if result.returncode != 0:
            raise PodManagerError("No profiling artifacts found on remote pod")
        
        # Use SCP to download entire chisel_out directory
        ssh_details = self.pod.get_ssh_details()
        if not ssh_details:
            raise PodManagerError("No SSH details available for pod")
            
        scp_cmd = [
            "scp", "-r", "-P", str(ssh_details["port"]), "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            f"root@{ssh_details['host']}:/workspace/chisel_out/*",
            str(run_path)
        ]
        
        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PodManagerError(f"Failed to fetch artifacts: {result.stderr}")
        
        return run_path
    
    def _detect_ssh_keys(self, ssh_key_path: Optional[str] = None) -> Tuple[str, str]:
        """Detect SSH key pair."""
        if ssh_key_path:
            private_key = Path(ssh_key_path)
            public_key = Path(f"{ssh_key_path}.pub")
        else:
            # Try common key types
            ssh_dir = Path.home() / ".ssh"
            for key_type in ["id_ed25519", "id_rsa"]:
                private_key = ssh_dir / key_type
                public_key = ssh_dir / f"{key_type}.pub"
                if private_key.exists() and public_key.exists():
                    break
            else:
                raise PodManagerError("No SSH key pair found in ~/.ssh/")
        
        if not private_key.exists() or not public_key.exists():
            raise PodManagerError(f"SSH key pair not found: {private_key}")
        
        try:
            public_key_content = public_key.read_text().strip()
            print(f"✓ Using SSH key: {private_key}")
            return str(private_key), public_key_content
        except Exception as e:
            raise PodManagerError(f"Could not read SSH key: {e}")
    
 