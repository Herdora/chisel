"""MCP Server for Chisel GPU profiling tool."""

import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("chisel", dependencies=["chisel-cli"])


@mcp.tool()
def configure(token: str) -> str:
    """
    Configure Chisel with your DigitalOcean API token.
    
    Args:
        token: Your DigitalOcean API token
        
    Returns:
        Success or error message
    """
    try:
        # Run chisel configure command
        result = subprocess.run(
            ["chisel", "configure", "--token", token],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check if successful
        if result.returncode == 0:
            # Parse output for account info
            output = result.stdout
            if "Configuration saved" in output and "ready to use" in output:
                # Extract email if present
                email_match = re.search(r"Email\s+│\s+(\S+)", output)
                email = email_match.group(1) if email_match else "configured"
                
                return f"✓ Chisel configured successfully for {email}. Ready to profile on AMD MI300X and NVIDIA H100/L40S GPUs."
            else:
                return "✓ Chisel configuration updated successfully."
        else:
            # Extract error message
            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
            if "Invalid API token" in error_msg:
                return "✗ Invalid API token. Please check your token and try again."
            else:
                return f"✗ Configuration failed: {error_msg}"
                
    except subprocess.TimeoutExpired:
        return "✗ Configuration timed out. Please try again."
    except Exception as e:
        return f"✗ Configuration error: {str(e)}"


@mcp.tool()
def profile_nvidia(
    target: str,
    gpu_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Profile a GPU kernel or command on NVIDIA infrastructure.
    
    Args:
        target: File to compile and profile (e.g., kernel.cu) or command to run
        gpu_type: Optional GPU type - "h100" (default) or "l40s"
        
    Returns:
        Dictionary containing profiling results, output paths, and cost estimate
    """
    try:
        # Build command
        cmd = ["chisel", "profile", "nvidia", target]
        if gpu_type:
            if gpu_type not in ["h100", "l40s"]:
                return {
                    "success": False,
                    "error": f"Invalid GPU type '{gpu_type}'. Must be 'h100' or 'l40s'."
                }
            cmd.extend(["--gpu-type", gpu_type])
        
        # Check if target is a file
        target_path = Path(target)
        if target_path.exists() and not target_path.is_file():
            return {
                "success": False,
                "error": f"Target '{target}' is not a file."
            }
        
        # Run profiling
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        output = result.stdout
        
        # Parse results
        if result.returncode == 0 and "Profiling completed successfully" in output:
            # Extract key information
            results_match = re.search(r"Results saved to:\s+(.+)", output)
            cost_match = re.search(r"Estimated cost:\s+\$([0-9.]+)", output)
            
            # Parse profile files
            ncu_files = re.findall(r"• (.+\.ncu-rep)", output)
            nsys_files = re.findall(r"• (.+\.nsys-rep)", output)
            
            return {
                "success": True,
                "output_dir": results_match.group(1) if results_match else "chisel-results",
                "cost_estimate": float(cost_match.group(1)) if cost_match else 0.0,
                "profile_files": {
                    "ncu": ncu_files,
                    "nsys": nsys_files
                },
                "gpu_type": gpu_type or "h100",
                "message": f"NVIDIA profiling completed. Generated {len(ncu_files)} ncu and {len(nsys_files)} nsys files."
            }
        else:
            # Extract error
            error_msg = result.stderr.strip() if result.stderr else output.strip()
            if "No API token configured" in error_msg:
                return {
                    "success": False,
                    "error": "No API token configured. Run 'configure' first."
                }
            else:
                return {
                    "success": False,
                    "error": error_msg or "Profiling failed"
                }
                
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Profiling timed out after 30 minutes."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Profiling error: {str(e)}"
        }


@mcp.tool()
def profile_amd(
    target: str,
    pmc: Optional[str] = None
) -> Dict[str, Any]:
    """
    Profile a GPU kernel or command on AMD MI300X infrastructure.
    
    Args:
        target: File to compile and profile (e.g., kernel.cpp) or command to run
        pmc: Optional performance counters (comma-separated, e.g., "GRBM_GUI_ACTIVE,SQ_WAVES")
        
    Returns:
        Dictionary containing profiling results, output paths, and cost estimate
    """
    try:
        # Build command
        cmd = ["chisel", "profile", "amd", target]
        if pmc:
            cmd.extend(["--pmc", pmc])
        
        # Check if target is a file
        target_path = Path(target)
        if target_path.exists() and not target_path.is_file():
            return {
                "success": False,
                "error": f"Target '{target}' is not a file."
            }
        
        # Run profiling
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        output = result.stdout
        
        # Parse results
        if result.returncode == 0 and "Profiling completed successfully" in output:
            # Extract key information
            results_match = re.search(r"Results saved to:\s+(.+)", output)
            cost_match = re.search(r"Estimated cost:\s+\$([0-9.]+)", output)
            
            # Parse rocprofv3 files
            att_files = re.findall(r"• (.+)", output)
            
            # Check if performance counters were collected
            pmc_collected = "Performance counters collected:" in output
            
            response = {
                "success": True,
                "output_dir": results_match.group(1) if results_match else "chisel-results",
                "cost_estimate": float(cost_match.group(1)) if cost_match else 0.0,
                "profile_files": att_files,
                "message": f"AMD rocprofv3 profiling completed. Generated {len(att_files)} output files."
            }
            
            if pmc_collected and pmc:
                response["performance_counters"] = pmc
                
            return response
        else:
            # Extract error
            error_msg = result.stderr.strip() if result.stderr else output.strip()
            if "No API token configured" in error_msg:
                return {
                    "success": False,
                    "error": "No API token configured. Run 'configure' first."
                }
            else:
                return {
                    "success": False,
                    "error": error_msg or "Profiling failed"
                }
                
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Profiling timed out after 30 minutes."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Profiling error: {str(e)}"
        }


def main():
    """Main entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()