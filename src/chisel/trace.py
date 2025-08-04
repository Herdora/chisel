import os
from typing import Optional, Callable, Any
from .constants import (
    CHISEL_BACKEND_RUN_ENV_KEY,
    CHISEL_JOB_ID_ENV_KEY,
    TRACE_DIR,
)


def capture_trace(
    trace_name: Optional[str] = None,
    record_shapes: bool = False,
    profile_memory: bool = False,
    **profiler_kwargs: Any,
) -> Callable:
    """
    Decorator for GPU execution and tracing.

    This function only activates when running on the Chisel backend.
    When running locally, it acts as a pass-through decorator.

    Args:
        trace_name: Operation identifier for the trace file
        record_shapes: Record tensor shapes for debugging
        profile_memory: Profile memory usage
        **profiler_kwargs: Additional profiler arguments

    Returns:
        Decorated function that traces execution on GPU backend
    """

    def decorator(fn: Callable) -> Callable:
        # Check if we're running on the Chisel backend
        if os.environ.get(CHISEL_BACKEND_RUN_ENV_KEY) != "1":
            # Running locally - return original function
            return fn

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return _execute_with_trace(
                fn, trace_name, record_shapes, profile_memory, *args, **kwargs
            )

        return wrapped

    return decorator


def _execute_with_trace(
    fn: Callable,
    trace_name: Optional[str],
    record_shapes: bool,
    profile_memory: bool,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute function with PyTorch profiling and save trace."""
    assert os.environ.get(CHISEL_BACKEND_RUN_ENV_KEY) == "1", "Chisel is not running on backend"

    import torch
    from torch.profiler import profile, ProfilerActivity
    from pathlib import Path

    trace_name = trace_name or fn.__name__
    job_id = os.environ.get(CHISEL_JOB_ID_ENV_KEY)

    if not job_id:
        print("⚠️  No job ID found, skipping trace")
        return fn(*args, **kwargs)

    volume_path = Path("/volume")
    job_trace_dir = volume_path / job_id / TRACE_DIR
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
        print(f"⚡ [capture_trace] Profiling {fn.__name__} (job_id: {job_id})")
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
