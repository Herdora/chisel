"""
Core functionality for kandc.

This module provides the main run management and code snapshot functionality.
"""

from .run import init, finish, get_current_run, is_initialized, log, Run
from .snapshot import capture_project_source_code, create_snapshot_archive

__all__ = [
    # Run management
    "init",
    "finish",
    "log",
    "get_current_run",
    "is_initialized",
    "Run",
    # Code snapshot
    "capture_project_source_code",
    "create_snapshot_archive",
]
