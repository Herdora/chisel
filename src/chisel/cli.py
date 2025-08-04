import os
import sys
import subprocess
import tempfile
import tarfile
import requests
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


def get_user_inputs() -> Dict[str, Any]:
    """Interactive questionnaire to get job submission parameters."""
    print("🚀 Chisel CLI - GPU Job Submission")
    print("=" * 40)

    # App name
    app_name = input("📝 App name (for job tracking): ").strip()
    while not app_name:
        print("❌ App name is required!")
        app_name = input("📝 App name (for job tracking): ").strip()

    # Upload directory
    upload_dir = input("📁 Upload directory (default: current directory): ").strip()
    if not upload_dir:
        upload_dir = "."

    # Requirements file
    requirements_file = input("📋 Requirements file (default: requirements.txt): ").strip()
    if not requirements_file:
        requirements_file = "requirements.txt"

    # GPU selection
    print("\n🎮 GPU Options:")
    gpu_options = [
        ("1", "A100-80GB:1", "Single GPU - Development, inference"),
        ("2", "A100-80GB:2", "2x GPUs - Medium training"),
        ("4", "A100-80GB:4", "4x GPUs - Large models"),
        ("8", "A100-80GB:8", "8x GPUs - Massive models"),
    ]

    for option, gpu_type, description in gpu_options:
        print(f"  {option}. {gpu_type} - {description}")

    gpu_choice = input("\n🎮 Select GPU configuration (1-8, default: 1): ").strip()
    if not gpu_choice:
        gpu_choice = "1"

    gpu_map = {"1": "A100-80GB:1", "2": "A100-80GB:2", "4": "A100-80GB:4", "8": "A100-80GB:8"}
    gpu = gpu_map.get(gpu_choice, "A100-80GB:1")

    return {
        "app_name": app_name,
        "upload_dir": upload_dir,
        "requirements_file": requirements_file,
        "gpu": gpu,
    }


def submit_job(
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

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = tmp_file.name

    try:
        spinner = SimpleSpinner(f"Creating archive from {upload_dir.name}")
        spinner.start()

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(upload_dir, arcname=".", filter=tar_filter)

            tar_size = Path(tar_path).stat().st_size
            size_mb = tar_size / (1024 * 1024)
            spinner.stop(f"Archive created: {size_mb:.1f} MB")
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
            "requirements_file": requirements_file,
        }

        endpoint = f"{backend_url.rstrip('/')}/api/v1/submit-cachy-job-new"

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


def run_chisel_command(command: List[str]) -> int:
    """Run the chisel command with interactive job submission."""
    if len(command) < 2:
        print("❌ No command provided!")
        print("Usage: chisel python <script.py> [args...]")
        return 1

    # Check if it's a python command
    if command[0] != "python":
        print("❌ Chisel currently only supports 'python' commands!")
        print("Usage: chisel python <script.py> [args...]")
        return 1

    # Get the script path
    script_path = command[1]
    script_args = command[2:] if len(command) > 2 else []

    # Get script absolute path
    script_abs_path = Path(script_path).resolve()

    # Authenticate first
    print("🔑 Checking authentication...")
    backend_url = os.environ.get(CHISEL_BACKEND_URL_ENV_KEY) or CHISEL_BACKEND_URL
    api_key = _auth_service.authenticate(backend_url)

    if not api_key:
        print("❌ Authentication failed. Please try again.")
        return 1

    print("✅ Authentication successful!")

    # Get user inputs
    inputs = get_user_inputs()

    # Validate upload directory contains the script
    upload_dir = Path(inputs["upload_dir"]).resolve()
    try:
        script_relative = script_abs_path.relative_to(upload_dir)
    except ValueError:
        print(f"❌ Script {script_abs_path} is not inside upload_dir {upload_dir}")
        return 1

    script_name = str(script_relative)
    args_display = f" {' '.join(script_args)}" if script_args else ""
    print(f"📦 Submitting job: {script_name}{args_display}")

    try:
        result = submit_job(
            app_name=inputs["app_name"],
            upload_dir=inputs["upload_dir"],
            script_path=script_name,
            gpu=inputs["gpu"],
            script_args=script_args,
            requirements_file=inputs["requirements_file"],
            api_key=api_key,  # Pass the api_key here
        )

        return result["exit_code"]
    except Exception as e:
        print(f"❌ Job submission failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Chisel CLI is installed and working!")
        print("Usage: chisel <command>")
        print("       chisel --logout")
        print("       chisel --version")
        print("Example: chisel python my_script.py")
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
    return run_chisel_command(command)
