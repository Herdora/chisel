"""
PyTorch Profiler annotators for GPU/CPU performance monitoring.

This module provides profiler wrappers and decorators that integrate with PyTorch's
built-in profiler to capture detailed performance metrics including GPU memory usage,
CUDA kernel execution, CPU operations, and more.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, List
from functools import wraps
from contextlib import contextmanager

from ..constants import (
    KANDC_BACKEND_RUN_ENV_KEY,
    KANDC_BACKEND_APP_NAME_ENV_KEY,
    KANDC_JOB_ID_ENV_KEY,
    KANDC_TRACE_BASE_DIR_ENV_KEY,
    KANDC_PROFILER_DISABLED_ENV_KEY,
    ARTIFACTS_DIR,
)

# Try to import PyTorch profiler
try:
    import torch
    import torch.profiler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ProfilerWrapper:
    """
    A wrapper that uses PyTorch profiler to capture detailed GPU/CPU performance metrics.
    
    This wrapper integrates with PyTorch's built-in profiler to capture:
    - GPU memory usage and allocation patterns
    - CUDA kernel execution times
    - CPU operation timings
    - Memory transfers between CPU/GPU
    - Detailed call stacks and operation traces
    """
    
    def __init__(self, obj: Any, name: Optional[str] = None, 
                 activities: Optional[List[str]] = None,
                 record_shapes: bool = True,
                 profile_memory: bool = True,
                 with_stack: bool = True):
        """
        Initialize the PyTorch profiler wrapper.
        
        Args:
            obj: The object to wrap and profile
            name: Optional name for the wrapped object in logs
            activities: List of activities to profile ['cpu', 'cuda']. Defaults to both.
            record_shapes: Whether to record tensor shapes
            profile_memory: Whether to profile memory usage
            with_stack: Whether to record call stacks
        """
        self._obj = obj
        self._name = name or obj.__class__.__name__
        self._profiling_enabled = not os.environ.get(KANDC_PROFILER_DISABLED_ENV_KEY) and TORCH_AVAILABLE
        
        # PyTorch profiler configuration
        if activities is None:
            activities = ['cpu']
            if TORCH_AVAILABLE and torch.cuda.is_available():
                activities.append('cuda')
        
        self._activities = []
        if TORCH_AVAILABLE:
            for activity in activities:
                if activity.lower() == 'cpu':
                    self._activities.append(torch.profiler.ProfilerActivity.CPU)
                elif activity.lower() == 'cuda' and torch.cuda.is_available():
                    self._activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        self._record_shapes = record_shapes
        self._profile_memory = profile_memory
        self._with_stack = with_stack
        self._profiler = None
        self._trace_data = []
        
    def __getattr__(self, name: str) -> Any:
        """
        Intercept attribute access and wrap callable methods with PyTorch profiling.
        
        Args:
            name: The attribute name being accessed
            
        Returns:
            The attribute value, wrapped with profiling if it's a callable method
        """
        attr = getattr(self._obj, name)
        
        # Only wrap callable attributes that are methods (not private)
        if (callable(attr) and 
            not name.startswith('_') and 
            hasattr(attr, '__self__') and 
            self._profiling_enabled):
            
            return self._wrap_method_with_profiler(name, attr)
        
        return attr
    
    def _wrap_method_with_profiler(self, method_name: str, method: Callable) -> Callable:
        """
        Wrap a method with PyTorch profiler integration.
        
        Args:
            method_name: Name of the method being wrapped
            method: The original method to wrap
            
        Returns:
            A wrapped version of the method with PyTorch profiling
        """
        @wraps(method)
        def profiled_method(*args, **kwargs):
            if not self._profiling_enabled:
                return method(*args, **kwargs)
            
            # Create a profiler for this method call
            with self._create_profiler(method_name):
                try:
                    result = method(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"⚠️  {self._name}.{method_name} ERROR: {e}")
                    raise
        
        return profiled_method
    
    @contextmanager
    def _create_profiler(self, method_name: str):
        """
        Create a PyTorch profiler context for a method call.
        
        Args:
            method_name: Name of the method being profiled
        """
        if not TORCH_AVAILABLE:
            print(f"⚠️  PyTorch not available, skipping profiling for {method_name}")
            yield
            return
        
        # Create profiler with configuration
        profiler = torch.profiler.profile(
            activities=self._activities,
            record_shapes=self._record_shapes,
            profile_memory=self._profile_memory,
            with_stack=self._with_stack,
            on_trace_ready=lambda prof: self._save_trace(prof, method_name)
        )
        
        try:
            profiler.start()
            yield
        finally:
            profiler.stop()
    
    def _save_trace(self, prof, method_name: str):
        """
        Save the profiler trace data as artifacts following the trace.py pattern.
        
        Args:
            prof: PyTorch profiler object
            method_name: Name of the method that was profiled
        """
        try:
            from pathlib import Path
            import time
            import uuid
            
            # Follow the same pattern as trace.py for directory structure
            base_path_env = os.getenv(KANDC_TRACE_BASE_DIR_ENV_KEY)
            base_path = Path(base_path_env) if base_path_env else Path("/volume")
            app = os.getenv(KANDC_BACKEND_APP_NAME_ENV_KEY)
            job = os.getenv(KANDC_JOB_ID_ENV_KEY)
            
            if not (app and job):
                # Missing run context → skip saving
                return
            
            job_artifacts_dir = base_path / app / job / ARTIFACTS_DIR
            job_artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique trace name using timestamp + method name (similar to trace.py)
            timestamp = int(time.time() * 1000000)  # microseconds
            unique_id = str(uuid.uuid4())[:8]
            
            trace_name = f"{self._name}_{method_name}_{timestamp}_{unique_id}"
            trace_file = job_artifacts_dir / f"{trace_name}.json"
            
            # Export Chrome trace (same as trace.py)
            prof.export_chrome_trace(str(trace_file))
            
            # Upload as artifact to backend (same pattern as trace.py)
            self._upload_trace_artifact(trace_file, trace_name, method_name, timestamp, unique_id)
            
        except Exception as e:
            print(f"⚠️  Failed to save profiler trace: {e}")
    
    def _upload_trace_artifact(self, trace_file: Path, trace_name: str, method_name: str, timestamp: int, unique_id: str):
        """
        Upload trace file as artifact to backend (following trace.py pattern).
        
        Args:
            trace_file: Path to the trace file
            trace_name: Name of the trace
            method_name: Name of the method that was profiled
            timestamp: Timestamp for the trace
            unique_id: Unique identifier for the trace
        """
        try:
            from ..core.run import _current_run
            
            if (
                _current_run
                and hasattr(_current_run, "_api_client")
                and hasattr(_current_run, "_run_data")
            ):
                if _current_run._api_client and _current_run._run_data:
                    # Use same metadata structure as trace.py
                    artifact_data = {
                        "name": f"{trace_name}.json",
                        "artifact_type": "trace",  # Same as trace.py
                        "file_size": trace_file.stat().st_size,
                        "metadata": {
                            "model_name": self._name,
                            "method_name": method_name,
                            "timestamp": timestamp,
                            "unique_id": unique_id,
                            "record_shapes": self._record_shapes,
                            "profile_memory": self._profile_memory,
                            "trace_format": "chrome_trace",
                        },
                    }
                    
                    _current_run._api_client.create_artifact(
                        _current_run._run_data["id"], artifact_data, str(trace_file)
                    )
                    
                    print(f"🔥 PyTorch trace saved: {trace_name}.json")
                    
        except Exception as e:
            print(f"⚠️  Failed to upload trace artifact: {e}")
    

    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get profiling statistics summary.
        
        Returns:
            Dictionary containing profiling information
        """
        return {
            "object_name": self._name,
            "profiling_enabled": self._profiling_enabled,
            "torch_available": TORCH_AVAILABLE,
            "activities": [str(act) for act in self._activities] if self._activities else [],
            "config": {
                "record_shapes": self._record_shapes,
                "profile_memory": self._profile_memory,
                "with_stack": self._with_stack
            }
        }
    
    def print_summary(self):
        """Print a summary of profiler configuration."""
        if not self._profiling_enabled:
            if not TORCH_AVAILABLE:
                print(f"🔥 PyTorch profiling disabled for {self._name} (PyTorch not available)")
            else:
                print(f"🔥 PyTorch profiling disabled for {self._name} (KANDC_PROFILER_DISABLED set)")
            return
            
        print(f"🔥 PyTorch Profiler: {self._name} - {len(self._activities)} activities, memory: {self._profile_memory}, shapes: {self._record_shapes}")



class ProfilerDecorator:
    """
    A decorator that automatically wraps classes or functions with PyTorch profiling.
    
    This decorator can be applied to classes or functions to automatically
    add PyTorch profiler capabilities. For classes, it wraps instances with ProfilerWrapper.
    For functions, it adds PyTorch profiling directly.
    """
    
    def __init__(self, name: Optional[str] = None, 
                 activities: Optional[List[str]] = None,
                 record_shapes: bool = True,
                 profile_memory: bool = True,
                 with_stack: bool = True):
        """
        Initialize the PyTorch profiler decorator.
        
        Args:
            name: Optional name for the profiled object
            activities: List of activities to profile ['cpu', 'cuda']. Defaults to both.
            record_shapes: Whether to record tensor shapes
            profile_memory: Whether to profile memory usage
            with_stack: Whether to record call stacks
        """
        self._name = name
        self._activities = activities
        self._record_shapes = record_shapes
        self._profile_memory = profile_memory
        self._with_stack = with_stack
    
    def __call__(self, target: Union[type, Callable]) -> Union[type, Callable]:
        """
        Apply profiling to the target.
        
        Args:
            target: Either a class or function to profile
            
        Returns:
            Profiled version of the target
        """
        if isinstance(target, type):
            # It's a class - wrap the constructor
            return self._wrap_class(target)
        else:
            # It's a function - wrap it directly
            return self._wrap_function(target)
    
    def _wrap_class(self, cls: type) -> type:
        """
        Wrap a class to profile all its instances with PyTorch profiler.
        
        Args:
            cls: The class to wrap
            
        Returns:
            A modified class that wraps instances with ProfilerWrapper
        """
        original_init = cls.__init__
        decorator_name = self._name  # Store the decorator's name
        decorator_config = {
            'activities': self._activities,
            'record_shapes': self._record_shapes,
            'profile_memory': self._profile_memory,
            'with_stack': self._with_stack
        }
        
        @wraps(original_init)
        def profiled_init(self, *args, **kwargs):
            # Call original constructor
            original_init(self, *args, **kwargs)
            
            # Wrap the instance with ProfilerWrapper
            wrapped = ProfilerWrapper(
                self, 
                decorator_name or cls.__name__,
                **decorator_config
            )
            
            # Replace the instance's methods with profiled versions
            for attr_name in dir(self):
                if not attr_name.startswith('_'):
                    attr = getattr(self, attr_name)
                    if callable(attr) and hasattr(attr, '__self__'):
                        # This is a method - replace it with the profiled version
                        setattr(self, attr_name, getattr(wrapped, attr_name))
        
        cls.__init__ = profiled_init
        return cls
    
    def _wrap_function(self, func: Callable) -> Callable:
        """
        Wrap a function with PyTorch profiling.
        
        Args:
            func: The function to wrap
            
        Returns:
            A wrapped function with PyTorch profiling
        """
        if os.environ.get(KANDC_PROFILER_DISABLED_ENV_KEY) or not TORCH_AVAILABLE:
            return func
        
        @wraps(func)
        def profiled_func(*args, **kwargs):
            func_name = self._name or func.__name__
            
            # Create a temporary wrapper to use the profiler infrastructure
            temp_wrapper = ProfilerWrapper(
                type('TempObject', (), {func.__name__: lambda self: func(*args, **kwargs)})(),
                func_name,
                activities=self._activities,
                record_shapes=self._record_shapes,
                profile_memory=self._profile_memory,
                with_stack=self._with_stack
            )
            
            # Use the profiler context
            with temp_wrapper._create_profiler(func_name):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"⚠️  {func_name} ERROR: {e}")
                    raise
        
        return profiled_func


# Convenience functions
def profile(obj: Any, name: Optional[str] = None, 
           activities: Optional[List[str]] = None,
           record_shapes: bool = True,
           profile_memory: bool = True,
           with_stack: bool = True) -> ProfilerWrapper:
    """
    Convenience function to wrap an object with PyTorch profiling.
    
    Args:
        obj: The object to profile
        name: Optional name for the object
        activities: List of activities to profile ['cpu', 'cuda']. Defaults to both.
        record_shapes: Whether to record tensor shapes
        profile_memory: Whether to profile memory usage
        with_stack: Whether to record call stacks
        
    Returns:
        A ProfilerWrapper around the object
    """
    return ProfilerWrapper(obj, name, activities, record_shapes, profile_memory, with_stack)


def profiler(name: Optional[str] = None,
            activities: Optional[List[str]] = None,
            record_shapes: bool = True,
            profile_memory: bool = True,
            with_stack: bool = True) -> ProfilerDecorator:
    """
    Convenience function to create a PyTorch profiler decorator.
    
    Args:
        name: Optional name for the profiled object
        activities: List of activities to profile ['cpu', 'cuda']. Defaults to both.
        record_shapes: Whether to record tensor shapes
        profile_memory: Whether to profile memory usage
        with_stack: Whether to record call stacks
        
    Returns:
        A ProfilerDecorator instance
    """
    return ProfilerDecorator(name, activities, record_shapes, profile_memory, with_stack)
