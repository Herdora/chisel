import os
import json
import time
import threading
from pathlib import Path
from typing import Callable, Optional, Any

from ..constants import (
    KANDC_BACKEND_RUN_ENV_KEY,
    KANDC_BACKEND_APP_NAME_ENV_KEY,
    KANDC_JOB_ID_ENV_KEY,
    KANDC_TRACE_BASE_DIR_ENV_KEY,
    KANDC_DISABLED_ENV_KEY,
    ARTIFACTS_DIR,
)

_counter_lock = threading.Lock()
_fn_counters: dict[str, int] = {}


def _job_artifacts_dir() -> Optional[Path]:
    if os.getenv(KANDC_DISABLED_ENV_KEY):
        return None
    app = os.getenv(KANDC_BACKEND_APP_NAME_ENV_KEY)
    job = os.getenv(KANDC_JOB_ID_ENV_KEY)
    base = os.getenv(KANDC_TRACE_BASE_DIR_ENV_KEY) or "/volume"
    if not app or not job:
        return None
    p = Path(base) / app / job / ARTIFACTS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def timed(name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that measures execution time and appends a JSON line per call.

    Writes to timings.jsonl inside the job's traces directory when running on backend.
    No-ops locally.
    """
    if os.environ.get(KANDC_DISABLED_ENV_KEY):
        return lambda fn: fn

    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        label = name or fn.__name__

        def wrapper(*args, **kwargs):
            td = _job_artifacts_dir()
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

                    # Persist one JSON file per call under artifacts and upload it
                    try:
                        # Write per-call file to artifacts dir
                        call_file = td / f"timing_{label}_{idx}.json"
                        try:
                            with open(call_file, "w", encoding="utf-8") as tf:
                                json.dump(rec, tf)
                        except Exception:
                            pass

                        from ..core.run import _current_run

                        if (
                            _current_run
                            and hasattr(_current_run, "_api_client")
                            and hasattr(_current_run, "_run_data")
                        ):
                            if _current_run._api_client and _current_run._run_data:
                                artifact_data = {
                                    "name": f"timing_{label}_{idx}.json",
                                    "artifact_type": "timing",
                                    "file_size": len(json.dumps(rec)),
                                    "metadata": rec,
                                }

                                _current_run._api_client.create_artifact(
                                    _current_run._run_data["id"], artifact_data, str(call_file)
                                )
                    except Exception:
                        pass
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
