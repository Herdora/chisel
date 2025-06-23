#!/usr/bin/env python3
"""MCP Server for Chisel - DigitalOcean GPU droplet management."""

import io
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Import chisel modules
from chisel.config import Config
from chisel.do_client import DOClient
from chisel.droplet import DropletManager
from chisel.ssh_manager import SSHManager

# Initialize FastMCP server
mcp = FastMCP("chisel")


class SuppressOutput:
    """Context manager to suppress stdout and stderr."""

    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr


@mcp.tool()
async def configure(token: Optional[str] = None) -> str:
    """Configure Chisel with your DigitalOcean API token.

    Args:
        token: Optional DigitalOcean API token. If not provided, will use existing token if available.
    """
    try:
        config = Config()

        # Check if token already exists
        existing_token = config.token

        if token:
            # Token provided via parameter
            api_token = token
        elif existing_token:
            # Token exists in config/env
            return "✓ DigitalOcean API token is already configured. To update, provide a new token parameter."
        else:
            # No token found
            return """❌ No DigitalOcean API token found.

To configure Chisel:
1. Go to: https://amd.digitalocean.com/account/api/tokens
2. Generate a new token with read and write access
3. Call this tool again with the token parameter

Example: configure with token="your_token_here" """

        # Validate token
        try:
            with SuppressOutput():
                do_client = DOClient(api_token)
                valid, account_info = do_client.validate_token()

            if valid and account_info:
                # Save token to config
                config.token = api_token

                # Get account info
                account_data = account_info.get("account", {})
                email = account_data.get("email", "N/A")
                status = account_data.get("status", "N/A")
                droplet_limit = account_data.get("droplet_limit", "N/A")

                return f"""✅ Token validated and saved successfully!

Account Information:
- Email: {email}
- Status: {status}
- Droplet Limit: {droplet_limit}

Configuration saved to: {config.config_file}

✅ Chisel is now configured and ready to use!"""

            else:
                return "❌ Invalid API token. Please check your token and try again."

        except Exception as e:
            return f"❌ Error validating token: {e}\n\nPlease ensure you have a valid DigitalOcean API token with read and write permissions."

    except Exception as e:
        return f"❌ Error during configuration: {e}"


@mcp.tool()
async def up() -> str:
    """Create or reuse a GPU droplet for development."""
    try:
        config = Config()

        # Check if configured
        if not config.token:
            return """❌ No API token configured.

Run the 'configure' tool first to set up your DigitalOcean API token."""

        # Initialize clients and suppress output
        with SuppressOutput():
            do_client = DOClient(config.token)
            droplet_manager = DropletManager(do_client)

            # Create or find droplet
            droplet = droplet_manager.up()

        # Format response
        name = droplet["name"]
        ip = droplet.get("ip", "N/A")
        region = droplet["region"]["slug"]
        size = droplet["size"]["slug"]

        return f"""✅ Droplet is ready!

Details:
- Name: {name}
- IP: {ip}
- Region: {region}
- Size: {size}

SSH: ssh root@{ip}"""

    except Exception as e:
        return f"❌ Error creating droplet: {e}"


@mcp.tool()
async def down() -> str:
    """Destroy the current droplet to stop billing."""
    try:
        config = Config()

        # Check if configured
        if not config.token:
            return """❌ No API token configured.

Run the 'configure' tool first to set up your DigitalOcean API token."""

        # Initialize clients and suppress output
        with SuppressOutput():
            do_client = DOClient(config.token)
            droplet_manager = DropletManager(do_client)

            # Note: In the original CLI, this prompts for confirmation
            # For MCP, we'll proceed directly since Claude will likely ask for confirmation
            success = droplet_manager.down()

        if success:
            return "✅ Droplet destroyed successfully. Billing has stopped."
        else:
            return "❌ Failed to destroy droplet or no droplet found to destroy."

    except Exception as e:
        return f"❌ Error destroying droplet: {e}"


@mcp.tool()
async def status() -> str:
    """Get status of current chisel droplets."""
    try:
        config = Config()

        # Check if configured
        if not config.token:
            return """❌ No API token configured.

Run the 'configure' tool first to set up your DigitalOcean API token."""

        # Initialize clients and suppress output
        with SuppressOutput():
            do_client = DOClient(config.token)
            droplet_manager = DropletManager(do_client)

            # Get droplets
            droplets = droplet_manager.list_droplets()

        if not droplets:
            return "ℹ️ No chisel droplets found"

        # Format droplet info
        result = f"📋 Found {len(droplets)} chisel droplet(s):\n\n"

        for droplet in droplets:
            name = droplet["name"]
            ip = droplet.get("ip", "N/A")
            status_val = droplet["status"]
            region = droplet["region"]["slug"]
            size = droplet["size"]["slug"]
            created = droplet["created_at"][:19].replace("T", " ")

            result += f"• {name}\n"
            result += f"  IP: {ip}\n"
            result += f"  Status: {status_val}\n"
            result += f"  Region: {region}\n"
            result += f"  Size: {size}\n"
            result += f"  Created: {created}\n\n"

        # Show state info
        with SuppressOutput():
            state_info = droplet_manager.state.get_droplet_info()
        if state_info:
            result += f"🎯 Active droplet: {state_info['name']} ({state_info['ip']})"

        return result.strip()

    except Exception as e:
        return f"❌ Error getting status: {e}"


@mcp.tool()
async def profile(
    file_or_command: str,
    trace: Optional[str] = "hip,hsa",
    analyze: bool = True
) -> str:
    """Profile a HIP file or command on the GPU droplet.
    
    Args:
        file_or_command: Either a path to a HIP file (e.g., 'kernel.hip') or a command to profile
        trace: Trace options (default: 'hip,hsa'). Can include 'hip', 'hsa', 'roctx'
        analyze: Whether to analyze and summarize the profiling results (default: True)
    
    Examples:
        - profile("matrix_multiply.hip")
        - profile("/root/chisel/my_kernel", trace="hip,hsa,roctx")
        - profile("ls -la", trace="hsa")
    """
    import os
    from pathlib import Path
    
    try:
        config = Config()
        
        # Check if configured
        if not config.token:
            return """❌ No API token configured.

Run the 'configure' tool first to set up your DigitalOcean API token."""
        
        # Initialize SSH manager
        with SuppressOutput():
            ssh_manager = SSHManager()
            
            # Get droplet info to ensure we have an active droplet
            droplet_info = ssh_manager.get_droplet_info()
            if not droplet_info:
                return """❌ No active droplet found.

Run the 'up' tool first to create a GPU droplet."""
        
        # Check if it's a local file that needs to be synced
        is_source_file = file_or_command.endswith(('.cpp', '.c', '.hip', '.cu'))
        is_local_file = not file_or_command.startswith('/') and is_source_file
        
        command_to_profile = file_or_command
        
        if is_local_file:
            # It's a local source file - need to sync and compile
            source_path = Path(file_or_command)
            
            if not source_path.exists():
                return f"❌ Source file '{file_or_command}' not found in local directory"
            
            # Sync the file
            with SuppressOutput():
                success = ssh_manager.sync(str(source_path))
            
            if not success:
                return f"❌ Failed to sync source file '{file_or_command}' to droplet"
            
            # Prepare compilation command
            remote_source = f"/root/chisel/{source_path.name}"
            remote_binary = f"/tmp/{source_path.stem}"
            
            if source_path.suffix in ['.cpp', '.hip']:
                compiler = "hipcc"
            elif source_path.suffix == '.cu':
                compiler = "nvcc"
            else:
                compiler = "gcc"
            
            # Build compile and run command
            compile_cmd = f"{compiler} {remote_source} -o {remote_binary}"
            command_to_profile = f"{compile_cmd} && {remote_binary}"
            
            result = f"📂 Synced and will compile: {file_or_command}\n"
            result += f"🔨 Compile command: {compile_cmd}\n\n"
        else:
            result = f"🎯 Profiling command: {command_to_profile}\n\n"
        
        # Run profiling
        with SuppressOutput():
            output_dir = "/tmp/chisel_mcp_profile"
            local_archive = ssh_manager.profile(
                command_to_profile, 
                trace=trace,
                output_dir=output_dir,
                open_result=False
            )
        
        if not local_archive:
            return "❌ Profiling failed. Check the command and ensure the droplet is accessible."
        
        result += f"✅ Profiling completed successfully!\n"
        result += f"📊 Results saved to: {local_archive}\n\n"
        
        # Read and analyze results if requested
        if analyze:
            try:
                # Look for result files
                import json
                import csv
                from pathlib import Path
                
                profile_dir = Path(local_archive)
                json_file = profile_dir / "results.json"
                csv_file = profile_dir / "results.csv"
                stats_csv_file = profile_dir / "results.stats.csv"
                
                # Try to read and summarize results
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    result += "📈 Profile Analysis (JSON format):\n"
                    result += f"   - Total events: {len(data) if isinstance(data, list) else 'N/A'}\n"
                    
                elif stats_csv_file.exists() or csv_file.exists():
                    # Use stats file if available, otherwise main CSV
                    target_file = stats_csv_file if stats_csv_file.exists() else csv_file
                    
                    result += "📈 Profile Analysis (CSV format):\n\n"
                    
                    # Read CSV and provide summary
                    with open(target_file, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    
                    if rows:
                        # Group by kernel name and summarize
                        kernel_stats = {}
                        for row in rows:
                            # Different formats based on rocprof version
                            kernel_name = row.get('Name', row.get('KernelName', row.get('kernelName', 'Unknown')))
                            
                            if kernel_name not in kernel_stats:
                                kernel_stats[kernel_name] = {
                                    'calls': 0,
                                    'total_duration': 0,
                                    'avg_duration': 0
                                }
                            
                            kernel_stats[kernel_name]['calls'] += 1
                            
                            # Try different duration field names
                            duration = None
                            for dur_field in ['DurationNs', 'Duration(ns)', 'duration', 'Duration']:
                                if dur_field in row:
                                    try:
                                        duration = float(row[dur_field])
                                        break
                                    except:
                                        continue
                            
                            if duration:
                                kernel_stats[kernel_name]['total_duration'] += duration
                        
                        # Calculate averages and format output
                        result += "Kernel Performance Summary:\n"
                        result += "-" * 60 + "\n"
                        
                        for kernel, stats in sorted(kernel_stats.items(), 
                                                   key=lambda x: x[1]['total_duration'], 
                                                   reverse=True)[:10]:  # Top 10 kernels
                            avg_duration = stats['total_duration'] / stats['calls'] if stats['calls'] > 0 else 0
                            total_ms = stats['total_duration'] / 1_000_000  # Convert ns to ms
                            avg_ms = avg_duration / 1_000_000
                            
                            result += f"\n🔸 {kernel}\n"
                            result += f"   Calls: {stats['calls']}\n"
                            result += f"   Total time: {total_ms:.3f} ms\n"
                            result += f"   Avg time: {avg_ms:.3f} ms\n"
                        
                        if len(kernel_stats) > 10:
                            result += f"\n... and {len(kernel_stats) - 10} more kernels\n"
                else:
                    result += "⚠️ No standard result files found for analysis\n"
                
                # List all files in the profile directory
                result += f"\n📁 Profile output files:\n"
                for file in sorted(profile_dir.glob("*")):
                    if file.is_file():
                        size = file.stat().st_size
                        size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
                        result += f"   - {file.name} ({size_str})\n"
                
            except Exception as e:
                result += f"\n⚠️ Could not analyze results: {e}\n"
                result += "The raw results are still available in the output directory.\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error during profiling: {e}"


@mcp.tool()
async def sync(
    source: str,
    destination: Optional[str] = None
) -> str:
    """Sync files or directories to the GPU droplet.
    
    Args:
        source: Local file or directory path to sync
        destination: Remote destination path (default: /root/chisel/)
    
    Examples:
        - sync("my_kernel.hip")
        - sync("./src", destination="/root/project/src")
    """
    try:
        config = Config()
        
        # Check if configured
        if not config.token:
            return """❌ No API token configured.

Run the 'configure' tool first to set up your DigitalOcean API token."""
        
        # Initialize SSH manager
        with SuppressOutput():
            ssh_manager = SSHManager()
            
            # Check for active droplet
            droplet_info = ssh_manager.get_droplet_info()
            if not droplet_info:
                return """❌ No active droplet found.

Run the 'up' tool first to create a GPU droplet."""
            
            # Perform sync
            success = ssh_manager.sync(source, destination)
        
        if success:
            dest = destination or "/root/chisel/"
            return f"""✅ Successfully synced '{source}' to droplet

📁 Source: {source}
📍 Destination: {dest}
🖥️  Droplet: {droplet_info['name']} ({droplet_info['ip']})"""
        else:
            return f"❌ Failed to sync '{source}' to droplet"
            
    except Exception as e:
        return f"❌ Error during sync: {e}"


@mcp.tool()
async def run(command: str) -> str:
    """Execute a command on the GPU droplet.
    
    Args:
        command: Command to execute on the remote droplet
    
    Examples:
        - run("ls -la")
        - run("hipcc my_kernel.hip -o my_kernel && ./my_kernel")
        - run("rocm-smi")
    """
    try:
        config = Config()
        
        # Check if configured
        if not config.token:
            return """❌ No API token configured.

Run the 'configure' tool first to set up your DigitalOcean API token."""
        
        # Initialize SSH manager
        with SuppressOutput():
            ssh_manager = SSHManager()
            
            # Check for active droplet
            droplet_info = ssh_manager.get_droplet_info()
            if not droplet_info:
                return """❌ No active droplet found.

Run the 'up' tool first to create a GPU droplet."""
        
        # Capture output
        import subprocess
        from io import StringIO
        import contextlib
        
        output_buffer = StringIO()
        
        # Run command and capture output
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            exit_code = ssh_manager.run(command)
        
        output = output_buffer.getvalue()
        
        result = f"🖥️  Droplet: {droplet_info['name']} ({droplet_info['ip']})\n"
        result += f"📟 Command: {command}\n"
        result += f"{"─" * 60}\n"
        
        if output:
            result += output
            if not output.endswith('\n'):
                result += '\n'
        
        result += f"{"─" * 60}\n"
        
        if exit_code == 0:
            result += "✅ Command completed successfully"
        else:
            result += f"❌ Command failed with exit code: {exit_code}"
        
        return result
        
    except Exception as e:
        return f"❌ Error executing command: {e}"


@mcp.tool()
async def pull(
    remote_path: str,
    local_path: Optional[str] = None
) -> str:
    """Pull files or directories from the GPU droplet to local machine.
    
    Args:
        remote_path: Remote file or directory path to pull
        local_path: Local destination path (default: current directory)
    
    Examples:
        - pull("/root/chisel/results.csv")
        - pull("/root/chisel/output/", local_path="./results/")
    """
    try:
        config = Config()
        
        # Check if configured
        if not config.token:
            return """❌ No API token configured.

Run the 'configure' tool first to set up your DigitalOcean API token."""
        
        # Initialize SSH manager
        with SuppressOutput():
            ssh_manager = SSHManager()
            
            # Check for active droplet
            droplet_info = ssh_manager.get_droplet_info()
            if not droplet_info:
                return """❌ No active droplet found.

Run the 'up' tool first to create a GPU droplet."""
            
            # Perform pull
            success = ssh_manager.pull(remote_path, local_path)
        
        if success:
            import os
            from pathlib import Path
            
            # Determine actual local path
            if local_path is None:
                actual_local = Path.cwd() / os.path.basename(remote_path.rstrip('/'))
            else:
                actual_local = Path(local_path)
            
            return f"""✅ Successfully pulled files from droplet

📍 Remote: {remote_path}
📁 Local: {actual_local}
🖥️  Droplet: {droplet_info['name']} ({droplet_info['ip']})"""
        else:
            return f"❌ Failed to pull '{remote_path}' from droplet"
            
    except Exception as e:
        return f"❌ Error during pull: {e}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
