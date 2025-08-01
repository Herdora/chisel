from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import os
import requests
import sys
import json
import webbrowser
import time
import threading
from .constants import (
    CHISEL_BACKEND_URL,
    CHISEL_BACKEND_RUN_ENV_KEY,
    CHISEL_JOB_ID_ENV_KEY,
    CHISEL_BACKEND_URL_ENV_KEY,
    MINIMUM_PACKAGES,
    TRACE_DIR,
)


EXCLUDE_PATTERNS = {
    ".venv",
    "venv",
    ".env",
    "__pycache__",
}


class SimpleSpinner:
    """Simple loading spinner for terminal."""

    def __init__(self, message="Loading"):
        self.message = message
        self.spinning = False
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.thread = None

    def _spin(self):
        """Spin the loading indicator."""
        idx = 0
        while self.spinning:
            print(
                f"\r{self.spinner_chars[idx % len(self.spinner_chars)]} {self.message}...",
                end="",
                flush=True,
            )
            idx += 1
            time.sleep(0.1)

    def start(self):
        """Start the spinner."""
        self.spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def stop(self, final_message=None):
        """Stop the spinner."""
        self.spinning = False
        if self.thread:
            self.thread.join()
        if final_message:
            print(f"\r✅ {final_message}")
        else:
            print(f"\r✅ {self.message} complete!")
        print()  # Add newline


def should_exclude(path):
    """Check if a path should be excluded from the tar archive."""
    path_parts = Path(path).parts
    for part in path_parts:
        if part in EXCLUDE_PATTERNS:
            return True
    return False


def tar_filter(tarinfo):
    """Filter function for tar archive creation (silent during spinner)."""
    if should_exclude(tarinfo.name):
        return None
    return tarinfo


class AuthService:
    """Manages Chisel authentication and configuration."""

    def __init__(self):
        self.config_path = Path.home() / ".chisel"
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from ~/.chisel file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_config(self) -> None:
        """Save configuration to ~/.chisel file."""
        # Ensure parent directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with restricted permissions (owner read/write only)
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        # Set file permissions to 600 (owner read/write only)
        os.chmod(self.config_path, 0o600)

    def get_api_key(self) -> Optional[str]:
        """Get stored API key."""
        return self.config.get("api_key")

    def set_api_key(self, api_key: str, email: Optional[str] = None) -> None:
        """Store API key and optionally email."""
        self.config["api_key"] = api_key
        if email:
            self.config["email"] = email
        self._save_config()
        print(f"✅ API key saved to {self.config_path}")

    def get_email(self) -> Optional[str]:
        """Get stored email."""
        return self.config.get("email")

    def clear(self) -> None:
        """Clear stored credentials."""
        if self.config_path.exists():
            self.config_path.unlink()
            self.config = {}
            print(f"🗑️  Cleared credentials from {self.config_path}")

    def is_authenticated(self) -> bool:
        """Check if user has stored API key."""
        return bool(self.get_api_key())

    def validate_api_key(self, api_key: str, backend_url: str) -> tuple[bool, Optional[str]]:
        """Validate API key with the backend."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(
                f"{backend_url}/api/v1/auth/validate", headers=headers, timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                return True, data.get("email")
            else:
                return False, None
        except requests.exceptions.RequestException:
            return True, None

    def authenticate(self, backend_url: str) -> Optional[str]:
        """
        Perform full authentication flow.
        Returns API key if successful, None if failed.
        """
        # Check existing API key first
        api_key = self.get_api_key()
        if api_key:
            print("🔑 Validating stored API key...")
            is_valid, email = self.validate_api_key(api_key, backend_url)

            if is_valid:
                if email:
                    print(f"✅ Authenticated as: {email}")
                return api_key
            else:
                print("❌ Stored API key is invalid. Starting authentication...")
                api_key = None

        if not api_key:
            print("🔑 No valid API key found. Starting authentication...")
            api_key = self._perform_browser_auth(backend_url)

        return api_key

    def _perform_browser_auth(self, backend_url: str) -> Optional[str]:
        """Perform browser-based authentication flow."""
        try:
            response = requests.get(f"{backend_url}/api/v1/auth/init")
            if response.status_code == 200:
                auth_data = response.json()
                session_id = auth_data["session_id"]
                auth_url = auth_data["auth_url"]

                print("🌐 Opening browser for authentication...")
                print(f"📱 If browser doesn't open, visit: {auth_url}")
                webbrowser.open(auth_url)

                print("⏳ Waiting for authentication...")
                for i in range(60):  # Wait up to 60 seconds
                    time.sleep(1)
                    check_response = requests.get(f"{backend_url}/api/v1/auth/check/{session_id}")
                    if check_response.status_code == 200:
                        check_data = check_response.json()
                        if check_data.get("authenticated"):
                            api_key = check_data.get("api_key")
                            email = check_data.get("email", "unknown")
                            print(f"✅ Authentication successful! Welcome {email}")

                            self.set_api_key(api_key, email)
                            return api_key

                    if i % 5 == 0:
                        print(f"⏳ Still waiting... ({60 - i}s remaining)")

                print("❌ Authentication timed out. Please try again.")
                return None
            else:
                print(f"❌ Failed to initiate authentication: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to backend: {e}")
            return None


# Global auth service instance
_auth_service = AuthService()


# Convenience functions for auth service
def get_api_key() -> Optional[str]:
    """Get stored API key."""
    return _auth_service.get_api_key()


def set_api_key(api_key: str, email: Optional[str] = None) -> None:
    """Store API key and optionally email."""
    _auth_service.set_api_key(api_key, email)


def get_email() -> Optional[str]:
    """Get stored email."""
    return _auth_service.get_email()


def clear_credentials() -> None:
    """Clear stored credentials."""
    _auth_service.clear()


def is_authenticated() -> bool:
    """Check if user has stored API key."""
    return _auth_service.is_authenticated()


def authenticate(backend_url: str = None) -> Optional[str]:
    """Perform authentication flow."""
    backend_url = backend_url or CHISEL_BACKEND_URL
    return _auth_service.authenticate(backend_url)


class ChiselApp:
    def __init__(self, name: str, upload_dir: str = ".", **kwargs: Any) -> None:
        self.app_name = name

        if os.environ.get(CHISEL_BACKEND_RUN_ENV_KEY) == "1":
            assert os.environ.get(CHISEL_JOB_ID_ENV_KEY), f"{CHISEL_JOB_ID_ENV_KEY} is not set"
            self.job_id = os.environ.get(CHISEL_JOB_ID_ENV_KEY)
            self.on_backend = True
            return

        self.job_id = None
        self.on_backend = False

        # Handle authentication automatically
        backend_url = os.environ.get(CHISEL_BACKEND_URL_ENV_KEY) or CHISEL_BACKEND_URL
        print("🔑 Authenticating with Chisel backend...")
        self.api_key = _auth_service.authenticate(backend_url)
        if not self.api_key:
            raise RuntimeError("❌ Authentication failed. Unable to get valid API key.")

        print("✅ Authentication successful!")

        script_abs_path = Path(sys.argv[0]).resolve()
        upload_dir = Path(upload_dir).resolve()

        try:
            script_relative = script_abs_path.relative_to(upload_dir)
        except ValueError:
            raise AssertionError(f"Script {script_abs_path} is not inside upload_dir {upload_dir}")

        script_name = str(script_relative)

        print(f"📦 [ChiselApp] Auto-submitting job: {script_name}")
        print(f"    Script absolute path: {script_abs_path}")
        print(f"    Upload directory: {upload_dir}")
        print(f"    Script relative to upload dir: {script_name}")
        print("    Environment: LOCAL (will submit to backend)")

        def submit_job(
            name: str,
            upload_dir: str,
            script_path: str = "main.py",
            pip_packages: Optional[List[str]] = None,
            local_source: Optional[str] = None,
            backend_url: Optional[str] = None,
        ) -> Dict[str, Any]:
            import tarfile
            import tempfile

            backend_url = (
                backend_url or os.environ.get(CHISEL_BACKEND_URL_ENV_KEY) or CHISEL_BACKEND_URL
            )

            # Use the already authenticated API key from ChiselApp instance
            api_key = self.api_key
            print(f"🔍 [submit_job] Backend URL: {backend_url}")

            upload_dir = Path(upload_dir)

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
                tar_path = tmp_file.name

            try:
                # Create archive with loading spinner
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

                # Prepare request
                headers = {"Authorization": f"Bearer {api_key}"}
                files = {"file": ("src.tar.gz", open(tar_path, "rb"), "application/gzip")}
                data = {
                    "script_path": script_path,
                    "app_name": name,
                    "pip_packages": ",".join(pip_packages) if pip_packages else "",
                }

                endpoint = f"{backend_url.rstrip('/')}/api/v1/submit-cachy-job-new"
                print(f"[submit_job] Submitting to: {endpoint}")

                response = requests.post(
                    endpoint, data=data, files=files, headers=headers, stream=True, timeout=60
                )
                response.raise_for_status()

                # Process streaming response
                logs = []
                job_id = None
                exit_code = None

                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue

                    try:
                        payload = json.loads(line[5:].strip())
                        msg_type = payload.get("type")

                        if msg_type == "log":
                            msg = payload.get("msg", "")
                            logs.append(msg)
                            print(msg)
                        elif msg_type in ("success", "error"):
                            exit_code = payload.get("exit_code")
                            job_id = payload.get("job_id")
                            status = "✅ completed" if msg_type == "success" else "❌ failed"
                            print(f"\n{status} (exit_code={exit_code}, job_id={job_id})")

                    except json.JSONDecodeError:
                        continue

                return {
                    "job_id": job_id,
                    "exit_code": exit_code,
                    "logs": logs,
                    "status": "completed" if exit_code == 0 else "failed",
                }
            except Exception as e:
                print(f"🔍 [submit_job] Error creating tar archive: {e}")
                raise

            finally:
                if os.path.exists(tar_path):
                    os.unlink(tar_path)

        res = submit_job(
            name=self.app_name,
            script_path=script_name,
            upload_dir=upload_dir,
            pip_packages=MINIMUM_PACKAGES,
        )

        print(f"🔍 [ChiselApp] Job submitted: {res}")
        exit(res["exit_code"])

    def capture_trace(
        self,
        trace_name: Optional[str] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        **profiler_kwargs: Any,
    ) -> Callable:
        """
        Decorator for capturing PyTorch profiling traces.

        Usage:
            @app.capture_trace(trace_name="my_function", record_shapes=True)
            def my_function(x):
                return x * 2
        """

        def decorator(fn: Callable) -> Callable:
            if not self.on_backend:
                return fn

            def wrapped(*args: Any, **kwargs: Any) -> Any:
                return self._execute_with_trace(
                    fn, trace_name, record_shapes, profile_memory, *args, **kwargs
                )

            return wrapped

        return decorator

    def _execute_with_trace(
        self,
        fn: Callable,
        trace_name: Optional[str],
        record_shapes: bool,
        profile_memory: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with PyTorch profiling trace."""
        assert self.on_backend, "Chisel is not running on backend"

        import torch
        from torch.profiler import profile, ProfilerActivity

        trace_name = trace_name or fn.__name__

        # Setup trace directory - assuming volume is mounted at /volume
        volume_path = Path("/volume")
        job_trace_dir = volume_path / self.app_name / self.job_id / TRACE_DIR
        job_trace_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔍 [capture_trace] Tracing {fn.__name__} -> {job_trace_dir}/{trace_name}.json")

        # Setup profiling activities
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
            gpu_count = torch.cuda.device_count()
            print(f"🚀 [capture_trace] GPU(s) available: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"    GPU {i}: {gpu_name}")

        with profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=False,
        ) as prof:
            print(f"⚡ [capture_trace] Profiling {fn.__name__} (job_id: {self.job_id})")
            result = fn(*args, **kwargs)

        trace_file = job_trace_dir / f"{trace_name}.json"
        prof.export_chrome_trace(str(trace_file))

        print(f"💾 [capture_trace] Saved trace: {trace_file}")

        if torch.cuda.is_available():
            print("\n🏎️  GPU Profiling Summary")
            print("─" * 50)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
        else:
            print("\n💻 CPU Profiling Summary")
            print("─" * 50)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))

        return result
