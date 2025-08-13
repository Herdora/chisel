import os
import json
import time
import threading
from pathlib import Path
from typing import Callable, Optional, Any

from .constants import (
    KANDC_BACKEND_RUN_ENV_KEY,
    KANDC_BACKEND_APP_NAME_ENV_KEY,
    KANDC_JOB_ID_ENV_KEY,
    KANDC_TRACE_BASE_DIR_ENV_KEY,
    TRACE_DIR,
)

_counter_lock = threading.Lock()
_fn_counters: dict[str, int] = {}


def _job_trace_dir() -> Optional[Path]:
    if os.getenv(KANDC_BACKEND_RUN_ENV_KEY) != "1":
        return None
    app = os.getenv(KANDC_BACKEND_APP_NAME_ENV_KEY)
    job = os.getenv(KANDC_JOB_ID_ENV_KEY)
    base = os.getenv(KANDC_TRACE_BASE_DIR_ENV_KEY) or "/volume"
    if not app or not job:
        return None
    p = Path(base) / app / job / TRACE_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def timed(name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that measures execution time and appends a JSON line per call.

    Writes to timings.jsonl inside the job's traces directory when running on backend.
    No-ops locally.
    """

    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        label = name or fn.__name__

        def wrapper(*args, **kwargs):
            td = _job_trace_dir()
            if td is None:
                return fn(*args, **kwargs)

            t0 = time.perf_counter_ns()
            status = "ok"
            err_msg = None
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # pragma: no cover - never block
                status = "error"
                err_msg = repr(e)
                raise
            finally:
                t1 = time.perf_counter_ns()
                with _counter_lock:
                    idx = _fn_counters.get(label, 0) + 1
                    _fn_counters[label] = idx

                rec = {
                    "name": label,
                    "call_index": idx,
                    "started_ns": t0,
                    "duration_us": (t1 - t0) // 1000,
                    "status": status,
                }
                if err_msg:
                    rec["error"] = err_msg

                out = td / "timings.jsonl"
                try:
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                except Exception:
                    pass

        return wrapper

    return _decorate


def timed_call(name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Convenience wrapper to time a single function call without decorating.
    # Example with multiple positional and keyword arguments:
    # def my_func(a, b, c=3, d=4):
    #     return a + b + c + d
    #
    # result = timed_call("my_func_timing", my_func, 1, 2, d=10)

    Example:
        result = timed_call("preprocess_batch", preprocess, batch)
    """
    return timed(name)(fn)(*args, **kwargs)
