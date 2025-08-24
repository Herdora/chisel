"""
Keys & Caches (kandc) - Experiment tracking and profiling library.

This package provides tools for tracking machine learning experiments,
capturing performance metrics, and profiling model execution.
"""

__version__ = "0.0.29"

# Import from new modular structure

# Core functionality
from .core import (
    init,
    finish,
    log,
    get_current_run,
    is_initialized,
)

# Annotators (decorators and wrapping functions)
from .annotators import (
    timed,
    timed_call,
    capture_trace,
    capture_model_instance,
    capture_model_class,
    parse_model_trace,
    ProfilerWrapper,
    ProfilerDecorator,
    profile,
    profiler,
)

# API functionality (if needed directly)
from .api import (
    APIClient,
    APIError,
    AuthenticationError,
    get_api_key,
    ensure_authenticated,
)

__all__ = [
    # Main API
    "init",
    "finish",
    "log",
    "get_current_run",
    "is_initialized",
    # Capture decorators
    "capture_trace",
    "capture_model_instance",
    "capture_model_class",
    "parse_model_trace",
    # Timing decorators
    "timed",
    "timed_call",
    # Profiler decorators and wrappers
    "ProfilerWrapper",
    "ProfilerDecorator",
    "profile",
    "profiler",
    "__version__",
    # API exports (for advanced usage)
    "APIClient",
    "APIError",
    "AuthenticationError",
    "get_api_key",
    "ensure_authenticated",
]
