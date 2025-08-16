"""
Command-line interface for kandc.

This module provides CLI commands for authentication, sweep operations,
and other command-line utilities.
"""

from .main import main
from .sweep import main as sweep_main

__all__ = [
    "main",
    "sweep_main",
]
