import os
import json
from typing import Optional, Callable, Any, Dict, List
from .constants import (
    KANDC_BACKEND_APP_NAME_ENV_KEY,
    KANDC_BACKEND_RUN_ENV_KEY,
    KANDC_JOB_ID_ENV_KEY,
    TRACE_DIR,
    KANDC_TRACE_BASE_DIR_ENV_KEY,
)


# Methods we attempt to profile in addition to forward(), when present
DEFAULT_FALLBACK_METHODS: tuple[str, ...] = (
    "generate",
    "sample",
    "predict",
    "encode",
    "decode",
    "__call__",
)


def capture_trace(
    trace_name: Optional[str] = None,
    record_shapes: bool = False,
    profile_memory: bool = False,
    **profiler_kwargs: Any,
) -> Callable:
    """
    Decorator for GPU execution and tracing.

    This function only activates when running on the Keys & Caches backend.
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
        # Check if we're running on the Keys & Caches backend

        if os.environ.get(KANDC_BACKEND_RUN_ENV_KEY) != "1":
            # Running locally - return original function
            return fn

        assert os.environ.get(KANDC_BACKEND_APP_NAME_ENV_KEY), "Keys & Caches app name is not set"

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return _execute_with_trace(
                fn, trace_name, record_shapes, profile_memory, *args, **kwargs
            )

        return wrapped

    return decorator


def capture_model_instance(
    model_instance,
    model_name: Optional[str] = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    **profiler_kwargs: Any,
):
    """
    Wrap a model instance to profile every forward pass.

    This function wraps an existing model instance (like HuggingFace models)
    to automatically profile each forward() call and save detailed traces.

    Args:
        model_instance: The model instance to wrap
        model_name: Name for the model traces (defaults to model class name)
        record_shapes: Record tensor shapes for each operation
        profile_memory: Profile memory usage
        **profiler_kwargs: Additional profiler arguments

    Returns:
        Wrapped model instance that profiles every forward pass

    Examples:
        # Wrap a HuggingFace model
        model = AutoModel.from_pretrained("bert-base-uncased")
        model = capture_model_instance(model, model_name="BERT")

        # Wrap any PyTorch model instance
        model = MyModel()
        model = capture_model_instance(model, model_name="MyModel")
    """
    # Check if we're running on the Keys & Caches backend
    if os.environ.get(KANDC_BACKEND_RUN_ENV_KEY) != "1":
        # Running locally - return original model instance
        return model_instance

    assert os.environ.get(KANDC_BACKEND_APP_NAME_ENV_KEY), "Keys & Caches app name is not set"

    # Store the original forward method
    original_forward = model_instance.forward
    trace_counter = 0

    # Initialize per-method counters and re-entrancy guard
    if not hasattr(model_instance, "_trace_counters"):
        model_instance._trace_counters = {}
    if not hasattr(model_instance, "_kandc_profiling_active"):
        model_instance._kandc_profiling_active = False

    def profiled_forward(*args, **kwargs):
        nonlocal trace_counter
        trace_counter += 1

        # Temporarily set trace counter on the model for compatibility
        model_instance._trace_counter = trace_counter

        return _execute_model_forward(
            model_instance,
            original_forward,
            model_name or model_instance.__class__.__name__,
            record_shapes,
            profile_memory,
            True,  # with_stack
            *args,
            **kwargs,
        )

    # Replace the forward method
    model_instance.forward = profiled_forward
    model_instance._trace_counter = trace_counter
    model_instance._model_name = model_name or model_instance.__class__.__name__

    # Attempt to wrap additional common methods if present on the instance
    try:
        _install_fallback_method_wrappers(
            model_instance,
            model_instance._model_name,
            record_shapes,
            profile_memory,
        )
    except Exception as _wrap_err:
        print(f"⚠️  Failed to install fallback method wrappers: {_wrap_err}")

    return model_instance


def capture_model_class(
    model_name: Optional[str] = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    **profiler_kwargs: Any,
):
    """
    Decorator for PyTorch model classes that profiles every forward pass.

    This creates a model wrapper that automatically profiles each forward() call
    and saves detailed traces with layer-level timing and shape information.

    Args:
        model_name: Name for the model traces (defaults to model class name)
        record_shapes: Record tensor shapes for each operation
        profile_memory: Profile memory usage
        **profiler_kwargs: Additional profiler arguments

    Returns:
        Model wrapper that profiles every forward pass

    Examples:
        @capture_model_class(model_name="MyModel")
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)
    """

    def decorator(model):
        # Check if we're running on the Keys & Caches backend
        if os.environ.get(KANDC_BACKEND_RUN_ENV_KEY) != "1":
            # Running locally - return original model
            return model

        assert os.environ.get(KANDC_BACKEND_APP_NAME_ENV_KEY), "Keys & Caches app name is not set"

        # Create a wrapper class that inherits from the original model
        class ProfiledModel(model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._trace_counter = 0
                self._model_name = model_name or model.__name__
                # Initialize per-method counters and re-entrancy guard
                self._trace_counters = {}
                self._kandc_profiling_active = False

            def forward(self, *args, **kwargs):
                return _execute_model_forward(
                    self,
                    super().forward,
                    self._model_name,
                    record_shapes,
                    profile_memory,
                    True,  # with_stack
                    *args,
                    **kwargs,
                )

        # Attach fallback wrappers after class definition by monkey-patching in __init__
        orig_init = ProfiledModel.__init__

        def __init_with_wrappers(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            try:
                _install_fallback_method_wrappers(
                    self,
                    self._model_name,
                    record_shapes,
                    profile_memory,
                )
            except Exception as _wrap_err:
                print(f"⚠️  Failed to install fallback method wrappers on class: {_wrap_err}")

        ProfiledModel.__init__ = __init_with_wrappers  # type: ignore[method-assign]

        return ProfiledModel

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
    assert os.environ.get(KANDC_BACKEND_RUN_ENV_KEY) == "1", (
        "Keys & Caches is not running on backend"
    )

    import torch
    from torch.profiler import profile, ProfilerActivity
    from pathlib import Path

    trace_name = trace_name or fn.__name__
    job_id = os.environ.get(KANDC_JOB_ID_ENV_KEY)

    if not job_id:
        print("⚠️  No job ID found, skipping trace")
        return fn(*args, **kwargs)

    base_path_env = os.environ.get(KANDC_TRACE_BASE_DIR_ENV_KEY)
    base_path = Path(base_path_env) if base_path_env else Path("/volume")
    job_trace_dir = base_path / os.environ.get(KANDC_BACKEND_APP_NAME_ENV_KEY) / job_id / TRACE_DIR
    job_trace_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=True,
    ) as prof:
        result = fn(*args, **kwargs)

    trace_file = job_trace_dir / f"{trace_name}.json"
    prof.export_chrome_trace(str(trace_file))

    print(f"💾 [capture_trace] {fn.__name__} → {trace_name}.json")

    return result


def _execute_model_forward(
    model,
    original_forward: Callable,
    model_name: str,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute model forward pass with PyTorch profiling and save trace."""
    assert os.environ.get(KANDC_BACKEND_RUN_ENV_KEY) == "1", (
        "Keys & Caches is not running on backend"
    )

    import torch
    from torch.profiler import profile, ProfilerActivity
    from pathlib import Path

    job_id = os.environ.get(KANDC_JOB_ID_ENV_KEY)
    if not job_id:
        print("⚠️  No job ID found, skipping model trace")
        return original_forward(*args, **kwargs)

    # Increment trace counter for this forward pass
    model._trace_counter += 1
    trace_name = f"{model_name}_forward_{model._trace_counter:03d}"

    base_path_env = os.environ.get(KANDC_TRACE_BASE_DIR_ENV_KEY)
    base_path = Path(base_path_env) if base_path_env else Path("/volume")
    job_trace_dir = base_path / os.environ.get(KANDC_BACKEND_APP_NAME_ENV_KEY) / job_id / TRACE_DIR
    job_trace_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Capture inputs summary BEFORE profiling to avoid polluting the trace
    _inputs_summary = _summarize_inputs_outputs(original_forward, args, kwargs)

    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        result = original_forward(*args, **kwargs)

    trace_file = job_trace_dir / f"{trace_name}.json"
    prof.export_chrome_trace(str(trace_file))

    # Capture outputs summary AFTER profiling to avoid polluting the trace
    _outputs_summary = _summarize_value(result)

    # Inject IO summary into the trace so the frontend can display without DB changes
    try:
        _inject_kandc_io_into_trace(
            str(trace_file),
            kandc_io={"inputs": _inputs_summary, "outputs": _outputs_summary},
        )
    except Exception as _inject_err:
        print(f"⚠️  Failed to inject kandc_io into trace: {_inject_err}")

    print(
        f"💾 [capture_model] {model_name} forward pass #{model._trace_counter} → {trace_name}.json"
    )

    return result


def _execute_model_method(
    model,
    original_method: Callable,
    model_name: str,
    method_name: str,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute any model method under PyTorch profiler and save trace.

    Also injects kandc_io with summarized inputs/outputs into the trace.
    """
    assert os.environ.get(KANDC_BACKEND_RUN_ENV_KEY) == "1", (
        "Keys & Caches is not running on backend"
    )

    import torch
    from torch.profiler import profile, ProfilerActivity
    from pathlib import Path

    job_id = os.environ.get(KANDC_JOB_ID_ENV_KEY)
    if not job_id:
        return original_method(*args, **kwargs)

    # Per-method counter on the instance
    counters: Dict[str, int] = getattr(model, "_trace_counters", {})
    current = counters.get(method_name, 0) + 1
    counters[method_name] = current
    model._trace_counters = counters

    trace_name = f"{model_name}_{method_name}_{current:03d}"

    base_path_env = os.environ.get(KANDC_TRACE_BASE_DIR_ENV_KEY)
    base_path = Path(base_path_env) if base_path_env else Path("/volume")
    job_trace_dir = base_path / os.environ.get(KANDC_BACKEND_APP_NAME_ENV_KEY) / job_id / TRACE_DIR
    job_trace_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Inputs summary before profiling to avoid polluting the trace
    _inputs_summary = _summarize_inputs_outputs(original_method, args, kwargs)

    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        result = original_method(*args, **kwargs)

    trace_file = job_trace_dir / f"{trace_name}.json"
    prof.export_chrome_trace(str(trace_file))

    # Outputs summary after profiling
    _outputs_summary = _summarize_value(result)

    try:
        _inject_kandc_io_into_trace(
            str(trace_file),
            kandc_io={"inputs": _inputs_summary, "outputs": _outputs_summary},
        )
    except Exception as _inject_err:
        print(f"⚠️  Failed to inject kandc_io into trace: {_inject_err}")

    print(f"💾 [capture_model] {model_name}.{method_name} #{current} → {trace_name}.json")
    return result


def _install_fallback_method_wrappers(
    model_instance: Any,
    model_name: str,
    record_shapes: bool,
    profile_memory: bool,
    fallback_methods: Optional[List[str]] = None,
) -> None:
    """Wrap a set of common non-forward methods if they exist on the instance.

    Avoid wrapping __call__ for torch.nn.Module, since it typically dispatches to forward().
    """
    methods = tuple(fallback_methods) if fallback_methods else DEFAULT_FALLBACK_METHODS

    # Avoid wrapping __call__ for torch.nn.Module to prevent double-profiling
    try:
        import torch

        is_torch_module = isinstance(model_instance, torch.nn.Module)
    except Exception:
        is_torch_module = False

    for method_name in methods:
        if method_name == "__call__" and is_torch_module:
            continue
        _wrap_method_on_instance(
            model_instance,
            method_name,
            model_name,
            record_shapes,
            profile_memory,
        )


def _wrap_method_on_instance(
    model_instance: Any,
    method_name: str,
    model_name: str,
    record_shapes: bool,
    profile_memory: bool,
) -> None:
    original = getattr(model_instance, method_name, None)
    if not callable(original):
        return

    # Prevent double-wrapping
    flag_name = f"_kandc_wrapped_{method_name}"
    if getattr(model_instance, flag_name, False):
        return

    def wrapper(*args: Any, **kwargs: Any):
        # Re-entrancy guard: if profiling already active (e.g., method calls into another wrapped method)
        if getattr(model_instance, "_kandc_profiling_active", False):
            return original(*args, **kwargs)
        try:
            model_instance._kandc_profiling_active = True
            if method_name == "forward":
                return _execute_model_forward(
                    model_instance,
                    original,
                    model_name,
                    record_shapes,
                    profile_memory,
                    True,
                    *args,
                    **kwargs,
                )
            return _execute_model_method(
                model_instance,
                original,
                model_name,
                method_name,
                record_shapes,
                profile_memory,
                True,
                *args,
                **kwargs,
            )
        finally:
            model_instance._kandc_profiling_active = False

    setattr(model_instance, method_name, wrapper)
    setattr(model_instance, flag_name, True)


def _summarize_value(val, max_elems: int = 16):
    """Summarize a Python value/tensor into a small JSON-serializable structure."""
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    # Torch tensor summary
    if torch is not None and isinstance(val, torch.Tensor):  # type: ignore[attr-defined]
        try:
            with torch.no_grad():
                cpu = val.detach().to("cpu")
                flat = cpu.reshape(-1)
                numel = int(flat.numel())
                sample = flat[:max_elems].tolist()
                stats = None
                try:
                    stats = {
                        "min": float(flat.min().item()),
                        "max": float(flat.max().item()),
                        "mean": float(flat.float().mean().item()),
                    }
                except Exception:
                    stats = None
                return {
                    "type": "tensor",
                    "dtype": str(cpu.dtype).replace("torch.", ""),
                    "shape": list(cpu.shape),
                    "strides": list(cpu.stride()),
                    "numel": numel,
                    "sample": sample,
                    "stats": stats,
                }
        except Exception:
            return {"type": "tensor", "repr": repr(val)}

    # Simple scalars
    if isinstance(val, (int, float, str, bool)):
        return {"type": "scalar", "value": val}

    # Sequences
    if isinstance(val, (list, tuple)):
        return {"type": "list", "items": [_summarize_value(v, max_elems) for v in val]}

    # Dicts
    if isinstance(val, dict):
        return {
            "type": "dict",
            "items": {str(k): _summarize_value(v, max_elems) for k, v in val.items()},
        }

    # Fallback
    return {"type": "unknown", "repr": repr(val)}


def _summarize_inputs_outputs(original_forward: Callable, args: tuple, kwargs: dict):
    """Summarize positional/keyword inputs to model.forward with parameter names when available."""
    import inspect

    sig = None
    try:
        sig = inspect.signature(original_forward)
    except Exception:
        pass

    # Map positional args to names (skip 'self')
    param_names = []
    if sig:
        for i, (name, _p) in enumerate(sig.parameters.items()):
            if i == 0 and name == "self":
                continue
            param_names.append(name)

    inputs = []
    for idx, v in enumerate(args):
        name = param_names[idx] if idx < len(param_names) else f"arg{idx}"
        inputs.append({"name": name, "value": _summarize_value(v)})

    for k, v in kwargs.items():
        inputs.append({"name": str(k), "value": _summarize_value(v)})

    return inputs


def _inject_kandc_io_into_trace(trace_path: str, kandc_io: dict) -> None:
    """Attach kandc_io to the model forward event in a chrome trace file.

    If a forward event is not found, append a small synthetic kandc_io event so
    the frontend can still discover and render it.
    """
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both array format and object with traceEvents
        events = data if isinstance(data, list) else data.get("traceEvents", [])
        if not isinstance(events, list):
            events = []

        # Find model forward event (python_function with 'forward' but not _execute_model_forward)
        candidates = [
            e
            for e in events
            if isinstance(e, dict)
            and e.get("ph") == "X"
            and e.get("cat") == "python_function"
            and "forward" in str(e.get("name", "")).lower()
            and "_execute_model_forward" not in str(e.get("name", ""))
        ]
        target = None
        if candidates:
            target = max(candidates, key=lambda e: (e.get("dur") or 0))

        if target is not None:
            target.setdefault("args", {})
            target["args"]["kandc_io"] = kandc_io
        else:
            ts0 = 0
            try:
                if events:
                    ts0 = int(events[0].get("ts", 0))
            except Exception:
                ts0 = 0
            io_event = {
                "ph": "X",
                "cat": "kandc_io",
                "name": "kandc.model_forward_io",
                "ts": ts0,
                "dur": 0,
                "args": {"kandc_io": kandc_io},
            }
            events.append(io_event)
            if not isinstance(data, list):
                data["traceEvents"] = events
            else:
                data = events

        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"⚠️  Failed to inject kandc_io into trace {trace_path}: {e}")


def _analyze_model_trace(trace_file: str, model_name: str) -> Optional[Dict]:
    """
    Analyze a model trace file to extract layer-level information.

    Args:
        trace_file: Path to the Chrome trace JSON file
        model_name: Name of the model for reference

    Returns:
        Dictionary with layer analysis or None if analysis fails
    """
    try:
        with open(trace_file, "r") as f:
            trace_data = json.load(f)

        # Extract events from trace
        events = trace_data.get("traceEvents", [])

        # Group events by layer/function
        layer_stats = {}

        for event in events:
            if event.get("ph") == "X" and event.get("cat") == "cpu_op":
                # This is a CPU operation event
                name = event.get("name", "")
                dur = event.get("dur", 0)  # Duration in microseconds
                args = event.get("args", {})

                # Extract shape information
                input_dims = args.get("Input Dims", [])
                shapes = []
                if input_dims:
                    for dims in input_dims:
                        if dims and len(dims) > 0:
                            shapes.append(tuple(dims))

                # Try to identify the layer from stack info
                stack = args.get("Stack", [])
                layer_name = _extract_layer_name(name, stack, model_name)

                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {
                        "total_time_us": 0,
                        "call_count": 0,
                        "shapes": set(),
                        "operations": set(),
                    }

                layer_stats[layer_name]["total_time_us"] += dur
                layer_stats[layer_name]["call_count"] += 1
                layer_stats[layer_name]["operations"].add(name)

                for shape in shapes:
                    layer_stats[layer_name]["shapes"].add(shape)

        return {"model_name": model_name, "trace_file": trace_file, "layer_stats": layer_stats}

    except Exception as e:
        print(f"⚠️  Failed to analyze trace {trace_file}: {e}")
        return None


def _extract_layer_name(op_name: str, stack: List, model_name: str) -> str:
    """
    Extract layer name from operation name and stack information.

    Args:
        op_name: Name of the PyTorch operation
        stack: Stack trace information
        model_name: Name of the model

    Returns:
        Extracted layer name
    """
    # Try to find module information in the stack
    for frame in stack:
        if isinstance(frame, dict):
            # Look for module-related information
            if "module" in str(frame).lower() or "forward" in str(frame).lower():
                # Extract module name from frame
                frame_str = str(frame)
                if "forward" in frame_str:
                    # Try to extract the module name before 'forward'
                    parts = frame_str.split("forward")
                    if len(parts) > 1:
                        module_part = parts[0].strip()
                        if "." in module_part:
                            return module_part.split(".")[-1].strip()

    # Fallback: try to extract from operation name
    if "::" in op_name:
        # PyTorch operations like "aten::conv2d"
        op_type = op_name.split("::")[-1]
        return f"Unknown_{op_type}"

    return f"Unknown_{op_name}"


def _print_layer_summary(layer_analysis: Dict):
    """
    Print a formatted summary of layer analysis.

    Args:
        layer_analysis: Layer analysis dictionary
    """
    layer_stats = layer_analysis["layer_stats"]

    if not layer_stats:
        print("No layer information found in trace")
        return

    # Sort layers by total time
    sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1]["total_time_us"], reverse=True)

    print(
        f"{'Layer':<25} {'Calls':<8} {'Total Time (ms)':<15} {'Avg Time (ms)':<15} {'Shapes':<20}"
    )
    print("─" * 85)

    for layer_name, stats in sorted_layers:
        total_time_ms = stats["total_time_us"] / 1000
        avg_time_ms = total_time_ms / stats["call_count"] if stats["call_count"] > 0 else 0

        # Format shapes for display
        shapes_str = ""
        if stats["shapes"]:
            shape_list = list(stats["shapes"])
            if len(shape_list) <= 2:
                shapes_str = ", ".join(str(s) for s in shape_list)
            else:
                shapes_str = f"{len(shape_list)} unique shapes"

        print(
            f"{layer_name:<25} {stats['call_count']:<8} "
            f"{total_time_ms:<15.2f} {avg_time_ms:<15.2f} {shapes_str:<20}"
        )

    # Print total model time
    total_model_time = sum(stats["total_time_us"] for stats in layer_stats.values()) / 1000
    print("─" * 85)
    print(f"{'TOTAL':<25} {'':<8} {total_model_time:<15.2f} {'':<15} {'':<20}")


def parse_model_trace(trace_file: str, model_name: str = "Unknown") -> Optional[Dict]:
    """
    Parse a model trace file and return detailed analysis.

    This is a public function that can be used to analyze trace files
    after they've been generated.

    Args:
        trace_file: Path to the Chrome trace JSON file
        model_name: Name of the model for reference

    Returns:
        Dictionary with detailed layer analysis or None if analysis fails
    """
    return _analyze_model_trace(trace_file, model_name)
