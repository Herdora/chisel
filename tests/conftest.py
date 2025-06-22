"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import os


@pytest.fixture
def mock_droplet_info():
    """Mock droplet info for testing."""
    return {
        "id": 12345,
        "name": "chisel-dev",
        "ip": "192.168.1.100",
        "status": "active",
        "region": {"slug": "atl1"},
        "size": {"slug": "gpu-mi300x1-192gb"},
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def mock_config():
    """Mock config for testing."""
    with patch("chisel.config.Config") as mock_config_class:
        mock_config_instance = Mock()
        mock_config_instance.token = "test-token-123"
        mock_config_instance.config_file = "/tmp/test-config.toml"
        mock_config_class.return_value = mock_config_instance
        yield mock_config_instance


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_ssh_client():
    """Mock paramiko SSH client."""
    with patch("paramiko.SSHClient") as mock_ssh:
        mock_client = Mock()
        mock_ssh.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_state():
    """Mock state manager."""
    with patch("chisel.state.State") as mock_state_class:
        mock_state_instance = Mock()
        mock_state_class.return_value = mock_state_instance
        yield mock_state_instance