"""
Chisel CLI - Profile AMD HIP kernels and PyTorch code on RunPod instances.
"""

import sys
from pathlib import Path

import click
from . import __version__
from .pod_manager import PodManager, PodManagerError


@click.group()
@click.version_option(version=__version__)
def main():
    """Chisel - Profile AMD HIP kernels and PyTorch code on RunPod instances."""
    pass


@main.command()
@click.argument('target', type=click.Choice(['amd', 'cuda']))
@click.option('-f', '--file', 'script_file', required=True, 
              help='Python script to profile')
@click.option('--no-cleanup', is_flag=True, 
              help='Keep connection to pod after execution')
def profile(target: str, script_file: str, no_cleanup: bool):
    """Profile a script on remote GPU instances."""
    
    # Validate input file
    script_path = Path(script_file)
    if not script_path.exists():
        click.echo(f"❌ Error: Script file '{script_file}' not found.", err=True)
        sys.exit(1)
    
    if not script_path.suffix == '.py':
        click.echo(f"❌ Error: '{script_file}' must be a Python file (.py)", err=True)
        sys.exit(1)
    
    click.echo(f"🔥 Starting {target.upper()} profiling session...")
    click.echo(f"📄 Script: {script_file}")
    
    try:
        # Initialize PodManager
        manager = PodManager(auto_cleanup=not no_cleanup)
        
        # Connect to pod
        click.echo("🔍 Finding available pod...")
        manager.ensure_pod()
        
        # Setup profiling environment on first run
        manager.setup_profiling_env(target)
        
        # Upload script
        click.echo("📤 Uploading script to pod...")
        remote_script = f"/tmp/{script_path.name}"
        manager.sync_code(str(script_path), "/tmp/")
        
        # Execute script with profiling
        click.echo(f"🚀 Executing {target.upper()} profiling...")
        result = manager.exec_with_profiling(remote_script, target=target, capture_output=False)
        
        if result.returncode == 0:
            click.echo("✅ Profiling completed successfully!")
            
            # Fetch artifacts from remote pod
            click.echo("📥 Fetching profiling artifacts...")
            try:
                artifacts_path = manager.fetch_artifacts()
                click.echo(f"📂 Artifacts saved to: {artifacts_path.absolute()}")
            except PodManagerError as e:
                click.echo(f"⚠️  Warning: Could not fetch artifacts: {e}")
        else:
            click.echo(f"❌ Script execution failed (exit code: {result.returncode})")
            if result.stderr:
                click.echo(f"Error output: {result.stderr}")
            sys.exit(result.returncode)
            
    except PodManagerError as e:
        click.echo(f"❌ Pod error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n⏹ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 