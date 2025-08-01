from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import os
import requests
import sys
import tempfile
import tarfile
from .constants import (
    CHISEL_BACKEND_URL,
    CHISEL_BACKEND_RUN_ENV_KEY,
    CHISEL_JOB_ID_ENV_KEY,
    CHISEL_BACKEND_URL_ENV_KEY,
    MINIMUM_PACKAGES,
    TRACE_DIR,
    GPUType,
)
from .spinner import SimpleSpinner
from .auth import _auth_service


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


class ChiselApp:
    def __init__(self, name: str, upload_dir: str = ".", **kwargs: Any) -> None:
        self.app_name = name

        gpu_param = kwargs.get("gpu", None)
        self.gpu = self._normalize_gpu_param(gpu_param)

        if os.environ.get("CHISEL_ACTIVATED") != "1":
            self.job_id = None
            self.on_backend = False
            self.activated = False
            return

        if os.environ.get(CHISEL_BACKEND_RUN_ENV_KEY) == "1":
            assert os.environ.get(CHISEL_JOB_ID_ENV_KEY), f"{CHISEL_JOB_ID_ENV_KEY} is not set"
            self.job_id = os.environ.get(CHISEL_JOB_ID_ENV_KEY)
            self.on_backend = True
            self.activated = True
            return

        self.job_id = None
        self.on_backend = False
        self.activated = True

        backend_url = os.environ.get(CHISEL_BACKEND_URL_ENV_KEY) or CHISEL_BACKEND_URL
        self.api_key = _auth_service.authenticate(backend_url)
        if not self.api_key:
            raise RuntimeError("❌ Authentication failed. Unable to get valid API key.")

        script_abs_path = Path(sys.argv[0]).resolve()
        upload_dir = Path(upload_dir).resolve()

        try:
            script_relative = script_abs_path.relative_to(upload_dir)
        except ValueError:
            raise AssertionError(f"Script {script_abs_path} is not inside upload_dir {upload_dir}")

        script_name = str(script_relative)
        script_args = sys.argv[1:]

        args_display = f" {' '.join(script_args)}" if script_args else ""
        print(f"📦 Submitting job: {script_name}{args_display}")

        def submit_job(
            name: str,
            upload_dir: str,
            script_path: str = "main.py",
            pip_packages: Optional[List[str]] = None,
            local_source: Optional[str] = None,
            backend_url: Optional[str] = None,
            gpu: Optional[str] = None,
            script_args: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            backend_url = (
                backend_url or os.environ.get(CHISEL_BACKEND_URL_ENV_KEY) or CHISEL_BACKEND_URL
            )

            api_key = self.api_key

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
                    "app_name": name,
                    "pip_packages": ",".join(pip_packages) if pip_packages else "",
                    "gpu": gpu,
                    "script_args": " ".join(script_args) if script_args else "",
                }

                endpoint = f"{backend_url.rstrip('/')}/api/v1/submit-cachy-job-new"

                upload_spinner = SimpleSpinner("Uploading work to backend")
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

        res = submit_job(
            name=self.app_name,
            script_path=script_name,
            upload_dir=upload_dir,
            pip_packages=MINIMUM_PACKAGES,
            gpu=self.gpu,
            script_args=script_args,
        )

        exit(res["exit_code"])

    def _normalize_gpu_param(self, gpu_param):
        """Convert GPUType enum to string value, or pass through string/None."""
        if gpu_param is None:
            return None
        if isinstance(gpu_param, GPUType):
            return gpu_param.value
        return gpu_param

    def capture_trace(
        self,
        trace_name: Optional[str] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        **profiler_kwargs: Any,
    ) -> Callable:
        def decorator(fn: Callable) -> Callable:
            if not self.activated:
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
        assert self.on_backend, "Chisel is not running on backend"

        import torch
        from torch.profiler import profile, ProfilerActivity

        trace_name = trace_name or fn.__name__

        volume_path = Path("/volume")
        job_trace_dir = volume_path / self.app_name / self.job_id / TRACE_DIR
        job_trace_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔍 [capture_trace] Tracing {fn.__name__} -> {job_trace_dir}/{trace_name}.json")

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
            with_stack=True,
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
