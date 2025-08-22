"""
Annotators for capturing metrics and traces.

This module provides decorators and wrapping functions for collecting metrics,
timing information, and model traces.
"""

from .timing import timed, timed_call
from .trace import (
    capture_trace,
    capture_model_instance,
    capture_model_class,
    parse_model_trace,
)
from .profiler import (
    ProfilerWrapper,
    ProfilerDecorator,
    profile,
    profiler,
)

__all__ = [
    # Timing decorators
    "timed",
    "timed_call",
    # Trace capture decorators
    "capture_trace",
    "capture_model_instance",
    "capture_model_class",
    "parse_model_trace",
    # Profiler decorators and wrappers
    "ProfilerWrapper",
    "ProfilerDecorator",
    "profile",
    "profiler",
]
