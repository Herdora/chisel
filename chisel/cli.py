"""
Chisel CLI - Profile AMD HIP kernels and PyTorch code on RunPod instances.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from . import __version__
from .pod_manager import PodManager, PodManagerError
from .perfetto_viewer import open_trace_in_perfetto


@click.group()
@click.version_option(version=__version__)
def main():
    """Chisel - Profile AMD HIP kernels and PyTorch code on RunPod instances."""
    pass


@main.command()
@click.argument("target", type=click.Choice(["amd", "cuda"]))
@click.option(
    "-f", "--file", "script_file", required=True, help="Python script to profile"
)
@click.option(
    "--no-cleanup", is_flag=True, help="Keep connection to pod after execution"
)
def profile(target: str, script_file: str, no_cleanup: bool):
    """Profile a script on remote GPU instances."""

    # Validate input file
    script_path = Path(script_file)
    if not script_path.exists():
        click.echo(f"❌ Error: Script file '{script_file}' not found.", err=True)
        sys.exit(1)

    if not script_path.suffix == ".py":
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
        result = manager.exec_with_profiling(
            remote_script, target=target, capture_output=False
        )

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


def _run_hip_profiling(
    file_path: str,
    hipcc_flags: str,
    no_cleanup: bool,
    session_name: str,
    profiling_args: dict,
    counters: Optional[str] = None,
):
    """Generic HIP profiling workflow."""
    # Validate input file
    hip_path = Path(file_path)
    if not hip_path.suffix == ".hip":
        click.echo(f"❌ Error: '{file_path}' must be a HIP file (.hip)", err=True)
        sys.exit(1)

    click.echo(f"🔥 Starting HIP {session_name} session...")
    click.echo(f"📄 HIP file: {file_path}")
    click.echo(f"⚙️  Compile flags: {hipcc_flags}")
    if counters:
        click.echo(f"📊 Counters: {counters}")

    try:
        # Initialize PodManager
        manager = PodManager(auto_cleanup=not no_cleanup)

        # Connect to pod
        click.echo("🔍 Finding available pod...")
        manager.ensure_pod()

        # Setup profiling environment
        click.echo("🔧 Setting up profiling environment...")
        manager.setup_profiling_env("amd")

        # Upload and compile HIP file
        click.echo("📤 Uploading HIP file to pod...")
        manager.sync_code(str(hip_path), "/tmp/")

        click.echo("🔨 Compiling HIP file...")
        executable_path = manager.compile_hip_file(f"/tmp/{hip_path.name}", hipcc_flags)

        # Execute with specified profiling
        click.echo(f"🚀 Executing HIP {session_name}...")
        if counters is not None:
            result = manager.exec_hip_with_counter_profiling(executable_path, counters)
        else:
            result = manager.exec_hip_with_profiling(executable_path, **profiling_args)

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
            click.echo(f"❌ Execution failed (exit code: {result.returncode})")
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


@main.command()
@click.argument("trace_file", type=click.Path(exists=True))
def visualize(trace_file: str):
    """Open trace files in Perfetto UI for interactive analysis."""

    trace_path = Path(trace_file)

    # Check if it's a JSON file
    if not trace_path.suffix == ".json":
        click.echo(f"❌ Error: '{trace_file}' must be a JSON trace file", err=True)
        sys.exit(1)

    click.echo("🚀 Opening trace file in Perfetto UI...")
    click.echo(f"📄 Trace: {trace_file}")

    try:
        success = open_trace_in_perfetto(trace_file)
        if not success:
            click.echo("❌ Failed to open trace in Perfetto", err=True)
            sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n⏹ Visualization stopped by user")
        sys.exit(0)
    except Exception as e:
        click.echo(f"❌ Error opening trace: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--hipcc-flags", default="-O2", help="Additional hipcc compilation flags")
@click.option(
    "--no-cleanup", is_flag=True, help="Keep connection to pod after execution"
)
def hip_trace(file_path: str, hipcc_flags: str, no_cleanup: bool):
    """Profile HIP kernels with basic runtime tracing."""
    _run_hip_profiling(
        file_path,
        hipcc_flags,
        no_cleanup,
        "runtime tracing",
        {"include_hip_trace": True, "output_prefix": "rocprof_trace"},
    )


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--hipcc-flags", default="-O2", help="Additional hipcc compilation flags")
@click.option(
    "--no-cleanup", is_flag=True, help="Keep connection to pod after execution"
)
def hip_kernel(file_path: str, hipcc_flags: str, no_cleanup: bool):
    """Profile HIP kernels with detailed kernel analysis."""
    _run_hip_profiling(
        file_path,
        hipcc_flags,
        no_cleanup,
        "kernel analysis",
        {"include_hip_trace": False, "output_prefix": "rocprof_kernel_trace"},
    )


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--hipcc-flags", default="-O2", help="Additional hipcc compilation flags")
@click.option(
    "--no-cleanup", is_flag=True, help="Keep connection to pod after execution"
)
def hip_system(file_path: str, hipcc_flags: str, no_cleanup: bool):
    """Complete system trace: HIP + HSA + comprehensive profiling."""
    _run_hip_profiling(
        file_path,
        hipcc_flags,
        no_cleanup,
        "system tracing",
        {"include_hip_trace": True, "output_prefix": "rocprof_system_trace"},
    )


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--hipcc-flags", default="-O2", help="Additional hipcc compilation flags")
@click.option(
    "--no-cleanup", is_flag=True, help="Keep connection to pod after execution"
)
def hip_memory(file_path: str, hipcc_flags: str, no_cleanup: bool):
    """Memory-focused profiling: allocation + copy traces."""
    _run_hip_profiling(
        file_path,
        hipcc_flags,
        no_cleanup,
        "memory profiling",
        {"include_hip_trace": False, "output_prefix": "rocprof_memory_trace"},
    )


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--counters",
    help="Comma-separated list of performance counters (e.g., SQ_WAVES,GRBM_COUNT)",
)
@click.option("--hipcc-flags", default="-O2", help="Additional hipcc compilation flags")
@click.option(
    "--no-cleanup", is_flag=True, help="Keep connection to pod after execution"
)
def hip_counters(file_path: str, counters: str, hipcc_flags: str, no_cleanup: bool):
    """Collect performance counters (e.g., SQ_WAVES,GRBM_COUNT)."""
    _run_hip_profiling(
        file_path, hipcc_flags, no_cleanup, "counter collection", {}, counters
    )


if __name__ == "__main__":
    main()
