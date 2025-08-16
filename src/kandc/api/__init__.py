"""
API communication layer for Keys & Caches.

This module handles all communication with the backend API.
"""

from .auth import AuthManager, get_api_key, ensure_authenticated
from .client import (
    APIClient,
    APIError,
    AuthenticationError,
    # Type exports
    ProjectResponse,
    RunResponse,
    MetricResponse,
    ArtifactResponse,
    AuthInitResponse,
    AuthStatusResponse,
    CodeSnapshotResponse,
)

__all__ = [
    # Auth
    "AuthManager",
    "get_api_key",
    "ensure_authenticated",
    # Client
    "APIClient",
    "APIError",
    "AuthenticationError",
    # Types
    "ProjectResponse",
    "RunResponse",
    "MetricResponse",
    "ArtifactResponse",
    "AuthInitResponse",
    "AuthStatusResponse",
    "CodeSnapshotResponse",
]
