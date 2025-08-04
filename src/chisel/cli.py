import os
import sys
import subprocess
import tempfile
import tarfile
import requests
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from .auth import _auth_service
from .constants import (
    CHISEL_BACKEND_URL,
    CHISEL_BACKEND_URL_ENV_KEY,
    MINIMUM_PACKAGES,
    GPUType,
)
from .spinner import SimpleSpinner
from .cached_files import process_directory_for_cached_files, scan_directory_for_large_files

# Rich imports for beautiful CLI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.layout import Layout
    from rich.columns import Columns
    from rich import box
    from rich.live import Live
    from rich.align import Align

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

EXCLUDE_PATTERNS = {
    ".venv",
    "venv",
    ".env",
    "__pycache__",
}


def should_exclude(path):
    path_parts = Path(path).parts
    for part in path_parts:
        if part in EXCLUDE_PATTERNS:
            return True
    return False


def tar_filter(tarinfo):
    if should_exclude(tarinfo.name):
        return None
    return tarinfo


class ChiselCLI:
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.gpu_options = [
            ("1", "A100-80GB:1", "Single GPU - Development, inference"),
            ("2", "A100-80GB:2", "2x GPUs - Medium training"),
            ("4", "A100-80GB:4", "4x GPUs - Large models"),
            ("8", "A100-80GB:8", "8x GPUs - Massive models"),
        ]
        self.gpu_map = {option: gpu_type for option, gpu_type, _ in self.gpu_options}

    def print_header(self):
        """Print the CLI header with styling."""
        if RICH_AVAILABLE:
            title = Text("🚀 Chisel CLI", style="bold blue")
            subtitle = Text("GPU Job Submission Tool", style="dim")

            header = Panel(
                title, subtitle=subtitle, border_style="blue", box=box.ROUNDED, padding=(1, 2)
            )
            self.console.print(header)
        else:
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║                    🚀 Chisel CLI                            ║")
            print("║                GPU Job Submission Tool                      ║")
            print("╚══════════════════════════════════════════════════════════════╝")
            print()

    def print_section_header(self, title: str):
        """Print a section header."""
        if RICH_AVAILABLE:
            self.console.print(f"\n[bold cyan]📋 {title}[/bold cyan]")
            self.console.print("─" * (len(title) + 4), style="dim")
        else:
            print(f"📋 {title}")
            print("─" * (len(title) + 4))

    def get_input_with_default(self, prompt: str, default: str = "", required: bool = True) -> str:
        """Get user input with a default value."""
        if RICH_AVAILABLE:
            if default:
                return Prompt.ask(f"{prompt}", default=default, console=self.console)
            else:
                while True:
                    user_input = Prompt.ask(f"{prompt}", console=self.console)
                    if user_input or not required:
                        return user_input
                    self.console.print("❌ This field is required!", style="red")
        else:
            if default:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                return user_input if user_input else default
            else:
                while True:
                    user_input = input(f"{prompt}: ").strip()
                    if user_input or not required:
                        return user_input
                    print("❌ This field is required!")

    def select_gpu(self) -> str:
        """Interactive GPU selection with navigation."""
        if RICH_AVAILABLE:
            self.console.print("\n[bold green]🎮 GPU Configuration:[/bold green]")

            # Create a table for better presentation
            table = Table(
                title="Available GPU Configurations",
                box=box.ROUNDED,
                border_style="green",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Option", style="cyan", no_wrap=True, width=8)
            table.add_column("GPU Type", style="yellow", no_wrap=True, width=15)
            table.add_column("Description", style="white")

            for option, gpu_type, description in self.gpu_options:
                table.add_row(option, gpu_type, description)

            self.console.print(table)
            self.console.print()

            # Use a simple but effective selection method
            while True:
                choice = Prompt.ask(
                    "Select GPU configuration",
                    choices=["1", "2", "4", "8"],
                    default="1",
                    console=self.console,
                )

                if choice in self.gpu_map:
                    selected_gpu = self.gpu_map[choice]
                    self.console.print(f"✅ Selected: [bold green]{selected_gpu}[/bold green]")
                    return selected_gpu
                else:
                    self.console.print(
                        "❌ Invalid choice. Please select 1, 2, 4, or 8.", style="red"
                    )
        else:
            print("\n🎮 GPU Configuration:")
            print("─" * 20)

            for option, gpu_type, description in self.gpu_options:
                print(f"  {option}. {gpu_type}")
                print(f"     {description}")
                print()

            while True:
                choice = input("Select GPU configuration (1-8, default: 1): ").strip()
                if not choice:
                    choice = "1"

                if choice in self.gpu_map:
                    selected_gpu = self.gpu_map[choice]
                    print(f"✅ Selected: {selected_gpu}")
                    return selected_gpu
                else:
                    print("❌ Invalid choice. Please select 1, 2, 4, or 8.")

    def get_user_inputs_interactive(self, script_path: str = "<script.py>") -> Dict[str, Any]:
        """Interactive questionnaire to get job submission parameters."""
        self.print_header()

        # App name
        self.print_section_header("Job Configuration")
        app_name = self.get_input_with_default("📝 App name (for job tracking)")

        # Upload directory
        upload_dir = self.get_input_with_default("📁 Upload directory", default=".", required=False)

        # Requirements file
        requirements_file = self.get_input_with_default(
            "📋 Requirements file", default="requirements.txt", required=False
        )

        # GPU selection
        gpu = self.select_gpu()

        # Show equivalent command for copy/paste
        self.show_equivalent_command(app_name, upload_dir, requirements_file, gpu, script_path)

        return {
            "app_name": app_name,
            "upload_dir": upload_dir,
            "requirements_file": requirements_file,
            "gpu": gpu,
        }

    def show_equivalent_command(
        self, app_name: str, upload_dir: str, requirements_file: str, gpu: str, script_path: str
    ):
        """Show the equivalent command-line command for copy/paste."""
        if RICH_AVAILABLE:
            # Build the command
            cmd_parts = ["chisel", "python", script_path]

            # Add flags
            if app_name:
                cmd_parts.append(f'--app-name "{app_name}"')

            if upload_dir != ".":
                cmd_parts.append(f'--upload-dir "{upload_dir}"')

            if requirements_file != "requirements.txt":
                cmd_parts.append(f'--requirements "{requirements_file}"')

            if gpu != "A100-80GB:1":
                # Convert GPU type back to number
                gpu_number = {v: k for k, v in self.gpu_map.items()}[gpu]
                cmd_parts.append(f"--gpu {gpu_number}")

            command = " ".join(cmd_parts)

            # Create a beautiful panel for the command
            command_text = Text(command, style="bold green")
            panel = Panel(
                command_text,
                title="📋 Equivalent Command (copy/paste for future use)",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2),
            )
            self.console.print(panel)
        else:
            print("\n" + "═" * 60)
            print("📋 Equivalent Command (copy/paste for future use):")
            print("═" * 60)

            # Build the command
            cmd_parts = ["chisel", "python", script_path]

            # Add flags
            if app_name:
                cmd_parts.append(f'--app-name "{app_name}"')

            if upload_dir != ".":
                cmd_parts.append(f'--upload-dir "{upload_dir}"')

            if requirements_file != "requirements.txt":
                cmd_parts.append(f'--requirements "{requirements_file}"')

            if gpu != "A100-80GB:1":
                # Convert GPU type back to number
                gpu_number = {v: k for k, v in self.gpu_map.items()}[gpu]
                cmd_parts.append(f"--gpu {gpu_number}")

            command = " ".join(cmd_parts)
            print(f"$ {command}")
            print("═" * 60)
            print()

    def parse_command_line_args(self, args: List[str]) -> Optional[Dict[str, Any]]:
        """Parse command line arguments and return configuration."""
        parser = argparse.ArgumentParser(
            description="Chisel CLI - GPU Job Submission Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  chisel python script.py --app-name my-job --gpu 4
  chisel python train.py --upload-dir ./project --requirements dev.txt
  chisel python inference.py --app-name inference-job --gpu 1
            """,
        )

        # Script and arguments (positional)
        parser.add_argument(
            "command", nargs="+", help="Python command to run (e.g., python script.py arg1 arg2)"
        )

        # Job configuration
        parser.add_argument("--app-name", "-a", help="App name for job tracking")
        parser.add_argument(
            "--upload-dir",
            "-d",
            default=".",
            help="Directory to upload (default: current directory)",
        )
        parser.add_argument(
            "--requirements",
            "-r",
            default="requirements.txt",
            help="Requirements file (default: requirements.txt)",
        )
        parser.add_argument(
            "--gpu",
            "-g",
            choices=["1", "2", "4", "8"],
            default="1",
            help="GPU configuration: 1, 2, 4, or 8 GPUs (default: 1)",
        )
        parser.add_argument(
            "--interactive",
            "-i",
            action="store_true",
            help="Force interactive mode even when flags are provided",
        )

        try:
            parsed_args = parser.parse_args(args)

            # Extract python command parts
            if parsed_args.command[0] != "python":
                print("❌ Chisel currently only supports 'python' commands!")
                print("Usage: chisel python <script.py> [args...]")
                return None

            script_path = parsed_args.command[1]
            script_args = parsed_args.command[2:] if len(parsed_args.command) > 2 else []

            return {
                "script_path": script_path,
                "script_args": script_args,
                "app_name": parsed_args.app_name,
                "upload_dir": parsed_args.upload_dir,
                "requirements_file": parsed_args.requirements,
                "gpu": self.gpu_map[parsed_args.gpu],
                "interactive": parsed_args.interactive,
            }
        except SystemExit:
            return None

    def submit_job(
        self,
        app_name: str,
        upload_dir: str,
        script_path: str,
        gpu: str,
        requirements_file: str,
        script_args: List[str],
        api_key: str,
    ) -> Dict[str, Any]:
        """Submit job to backend."""
        backend_url = os.environ.get(CHISEL_BACKEND_URL_ENV_KEY) or CHISEL_BACKEND_URL

        upload_dir = Path(upload_dir)

        # Process directory for cached files
        processed_dir = upload_dir
        cached_files_info = []

        # Check for large files that could be cached
        large_files = scan_directory_for_large_files(upload_dir)
        if large_files:
            if RICH_AVAILABLE:
                self.console.print(
                    f"[yellow]Found {len(large_files)} large file(s) that could be cached[/yellow]"
                )
            else:
                print(f"Found {len(large_files)} large file(s) that could be cached")

            try:
                # Create a temporary directory for processing
                temp_processing_dir = Path(tempfile.mkdtemp())

                # Process the directory to handle cached files
                print(f"Processing directory: {upload_dir}")
                processed_dir, cached_files_info = process_directory_for_cached_files(
                    upload_dir, api_key, temp_processing_dir
                )

                if cached_files_info:
                    if RICH_AVAILABLE:
                        self.console.print(
                            f"[green]Successfully processed {len(cached_files_info)} cached file(s)[/green]"
                        )
                    else:
                        print(f"Successfully processed {len(cached_files_info)} cached file(s)")

            except Exception as e:
                if RICH_AVAILABLE:
                    self.console.print(
                        f"[yellow]Warning: Could not process cached files: {e}[/yellow]"
                    )
                    self.console.print("[yellow]Continuing with original directory...[/yellow]")
                else:
                    print(f"Warning: Could not process cached files: {e}")
                    print("Continuing with original directory...")

                # Fallback to original directory if processing fails
                processed_dir = upload_dir
                cached_files_info = []

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = tmp_file.name

        try:
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task(
                        f"Creating archive from {processed_dir.name}", total=None
                    )

                    try:
                        with tarfile.open(tar_path, "w:gz") as tar:
                            tar.add(processed_dir, arcname=".", filter=tar_filter)

                        tar_size = Path(tar_path).stat().st_size
                        size_mb = tar_size / (1024 * 1024)
                        size_kb = tar_size / 1024

                        # Show KB for small archives, MB for larger ones
                        if size_mb < 1.0:
                            progress.update(task, description=f"Archive created: {size_kb:.2f} KB")
                        else:
                            progress.update(task, description=f"Archive created: {size_mb:.3f} MB")
                    except Exception as e:
                        progress.update(task, description="Archive creation failed")
                        raise e
            else:
                spinner = SimpleSpinner(f"Creating archive from {processed_dir.name}")
                spinner.start()

                try:
                    with tarfile.open(tar_path, "w:gz") as tar:
                        tar.add(processed_dir, arcname=".", filter=tar_filter)

                    tar_size = Path(tar_path).stat().st_size
                    size_mb = tar_size / (1024 * 1024)
                    size_kb = tar_size / 1024

                    # Show KB for small archives, MB for larger ones
                    if size_mb < 1.0:
                        spinner.stop(f"Archive created: {size_kb:.2f} KB")
                    else:
                        spinner.stop(f"Archive created: {size_mb:.3f} MB")
                except Exception as e:
                    spinner.stop("Archive creation failed")
                    raise e

            headers = {"Authorization": f"Bearer {api_key}"}
            files = {"file": ("src.tar.gz", open(tar_path, "rb"), "application/gzip")}
            data = {
                "script_path": script_path,
                "app_name": app_name,
                "pip_packages": ",".join(MINIMUM_PACKAGES),
                "gpu": gpu,
                "script_args": " ".join(script_args) if script_args else "",
                "cached_files": json.dumps(cached_files_info) if cached_files_info else "",
                "requirements_file": requirements_file,
            }

            endpoint = f"{backend_url.rstrip('/')}/api/v1/submit-cachy-job-new"

            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Uploading work to backend and running", total=None)

                    try:
                        response = requests.post(
                            endpoint, data=data, files=files, headers=headers, timeout=12 * 60 * 60
                        )
                        response.raise_for_status()

                        result = response.json()
                        job_id = result.get("job_id")
                        message = result.get("message", "Job submitted")
                        visit_url = result.get("visit_url", f"/jobs/{job_id}")

                        progress.update(
                            task, description="Work uploaded successfully! Job submitted"
                        )

                        # Create a success panel
                        success_panel = Panel(
                            f"🔗 Job ID: {job_id}\n🌐 Visit: {visit_url}\n📊 Job is running in the background on cloud GPUs",
                            title="✅ Job Submitted Successfully",
                            border_style="green",
                            box=box.ROUNDED,
                        )
                        self.console.print(success_panel)

                    except Exception as e:
                        progress.update(task, description="Upload failed")
                        raise e
            else:
                upload_spinner = SimpleSpinner("Uploading work to backend and running.")
                upload_spinner.start()

                try:
                    response = requests.post(
                        endpoint, data=data, files=files, headers=headers, timeout=12 * 60 * 60
                    )
                    response.raise_for_status()

                    result = response.json()
                    job_id = result.get("job_id")
                    message = result.get("message", "Job submitted")
                    visit_url = result.get("visit_url", f"/jobs/{job_id}")

                    upload_spinner.stop("Work uploaded successfully! Job submitted")

                    print(f"🔗 Job ID: {job_id}")
                    print(f"🌐 Visit: {visit_url}")
                    print("📊 Job is running in the background on cloud GPUs")

                except Exception as e:
                    upload_spinner.stop("Upload failed")
                    raise e

            return {
                "job_id": job_id,
                "exit_code": 0,
                "logs": [f"{message} (Job ID: {job_id})"],
                "status": "submitted",
                "visit_url": visit_url,
            }
        except Exception as e:
            print(f"🔍 [submit_job] Error creating tar archive: {e}")
            raise
        finally:
            if os.path.exists(tar_path):
                os.unlink(tar_path)

    def run_chisel_command(self, command: List[str]) -> int:
        """Run the chisel command with improved interface."""
        if len(command) < 2:
            if RICH_AVAILABLE:
                self.console.print("❌ No command provided!", style="red")
                self.console.print("\n[bold]Usage:[/bold]")
                self.console.print("  chisel python <script.py> [args...]")
                self.console.print("  chisel python <script.py> --app-name my-job --gpu 4")
                self.console.print("  chisel python <script.py> --interactive")
            else:
                print("❌ No command provided!")
                print("Usage: chisel python <script.py> [args...]")
                print("       chisel python <script.py> --app-name my-job --gpu 4")
                print("       chisel python <script.py> --interactive")
            return 1

        # Try to parse command line arguments first
        parsed_config = self.parse_command_line_args(command)
        if parsed_config is None:
            return 1

        # If no app_name provided via flags, we need interactive mode
        if not parsed_config["app_name"] or parsed_config["interactive"]:
            # Get interactive inputs
            interactive_inputs = self.get_user_inputs_interactive(parsed_config["script_path"])

            # Merge with command line args (command line takes precedence)
            final_config = {**interactive_inputs}
            if parsed_config["app_name"]:
                final_config["app_name"] = parsed_config["app_name"]
            if parsed_config["gpu"]:
                final_config["gpu"] = parsed_config["gpu"]
            if parsed_config["upload_dir"]:
                final_config["upload_dir"] = parsed_config["upload_dir"]
            if parsed_config["requirements_file"]:
                final_config["requirements_file"] = parsed_config["requirements_file"]
        else:
            # Use command line configuration
            final_config = {
                "app_name": parsed_config["app_name"],
                "upload_dir": parsed_config["upload_dir"],
                "requirements_file": parsed_config["requirements_file"],
                "gpu": parsed_config["gpu"],
            }

        # Get script information
        script_path = parsed_config["script_path"]
        script_args = parsed_config["script_args"]

        # Get script absolute path
        script_abs_path = Path(script_path).resolve()

        # Authenticate first
        if RICH_AVAILABLE:
            self.console.print("🔑 Checking authentication...", style="yellow")
        else:
            print("🔑 Checking authentication...")

        backend_url = os.environ.get(CHISEL_BACKEND_URL_ENV_KEY) or CHISEL_BACKEND_URL
        api_key = _auth_service.authenticate(backend_url)

        if not api_key:
            if RICH_AVAILABLE:
                self.console.print("❌ Authentication failed. Please try again.", style="red")
            else:
                print("❌ Authentication failed. Please try again.")
            return 1

        if RICH_AVAILABLE:
            self.console.print("✅ Authentication successful!", style="green")
        else:
            print("✅ Authentication successful!")

        # Validate upload directory contains the script
        upload_dir = Path(final_config["upload_dir"]).resolve()
        try:
            script_relative = script_abs_path.relative_to(upload_dir)
        except ValueError:
            if RICH_AVAILABLE:
                self.console.print(
                    f"❌ Script {script_abs_path} is not inside upload_dir {upload_dir}",
                    style="red",
                )
            else:
                print(f"❌ Script {script_abs_path} is not inside upload_dir {upload_dir}")
            return 1

        script_name = str(script_relative)
        args_display = f" {' '.join(script_args)}" if script_args else ""

        if RICH_AVAILABLE:
            self.console.print(f"📦 Submitting job: [bold]{script_name}{args_display}[/bold]")
        else:
            print(f"📦 Submitting job: {script_name}{args_display}")

        try:
            result = self.submit_job(
                app_name=final_config["app_name"],
                upload_dir=final_config["upload_dir"],
                script_path=script_name,
                gpu=final_config["gpu"],
                script_args=script_args,
                requirements_file=final_config["requirements_file"],
                api_key=api_key,
            )

            return result["exit_code"]
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"❌ Job submission failed: {e}", style="red")
            else:
                print(f"❌ Job submission failed: {e}")
            return 1


def main():
    """Main CLI entry point."""
    cli = ChiselCLI()

    if len(sys.argv) < 2:
        if cli.console:
            cli.console.print("Chisel CLI is installed and working!", style="bold green")
            cli.console.print("\n[bold]Usage:[/bold]")
            cli.console.print("  chisel python <script.py> [args...]")
            cli.console.print("  chisel python <script.py> --app-name my-job --gpu 4")
            cli.console.print("  chisel python <script.py> --interactive")
            cli.console.print("  chisel --logout")
            cli.console.print("  chisel --version")
            cli.console.print("\n[bold]Examples:[/bold]")
            cli.console.print("  chisel python my_script.py")
            cli.console.print("  chisel python train.py --app-name training-job --gpu 4")
            cli.console.print(
                "  chisel python inference.py --upload-dir ./project --requirements dev.txt"
            )
            cli.console.print(
                "\n💡 Tip: Interactive mode shows the equivalent command-line for copy/paste!"
            )
        else:
            print("Chisel CLI is installed and working!")
            print()
            print("Usage:")
            print("  chisel python <script.py> [args...]")
            print("  chisel python <script.py> --app-name my-job --gpu 4")
            print("  chisel python <script.py> --interactive")
            print("  chisel --logout")
            print("  chisel --version")
            print()
            print("Examples:")
            print("  chisel python my_script.py")
            print("  chisel python train.py --app-name training-job --gpu 4")
            print("  chisel python inference.py --upload-dir ./project --requirements dev.txt")
            print()
            print("💡 Tip: Interactive mode shows the equivalent command-line for copy/paste!")
        return 0

    # Handle version flag
    if sys.argv[1] in ["--version", "-v", "version"]:
        from . import __version__

        print(f"Chisel CLI v{__version__}")
        return 0

    # Handle logout flag
    if sys.argv[1] == "--logout":
        if _auth_service.is_authenticated():
            _auth_service.clear()
            print("✅ Successfully logged out from Chisel CLI")
        else:
            print("ℹ️  No active authentication found")
        return 0

    # Run chisel command
    command = sys.argv[1:]
    return cli.run_chisel_command(command)
