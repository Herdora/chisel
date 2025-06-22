"""Smoke tests for basic CLI functionality."""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch

from chisel.main import app


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


def test_cli_help(runner):
    """Test that CLI help works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "chisel" in result.stdout
    assert "configure" in result.stdout
    assert "up" in result.stdout
    assert "down" in result.stdout
    assert "sync" in result.stdout
    assert "run" in result.stdout
    assert "pull" in result.stdout
    assert "profile" in result.stdout


def test_individual_command_help(runner):
    """Test that individual command help works."""
    commands = ["configure", "up", "down", "list", "sync", "run", "pull", "profile", "version"]
    
    for command in commands:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, f"Help failed for command: {command}"
        assert command in result.stdout.lower()


def test_version_command(runner):
    """Test version command works."""
    with patch("chisel.__version__", "0.1.0"):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout


def test_pull_command_structure(runner):
    """Test pull command has correct structure."""
    result = runner.invoke(app, ["pull", "--help"])
    assert result.exit_code == 0
    assert "remote_path" in result.stdout
    assert "--local" in result.stdout


def test_interrupt_handler_creation():
    """Test that InterruptHandler can be created."""
    from chisel.ssh_manager import InterruptHandler
    
    handler = InterruptHandler()
    assert handler is not None
    assert handler.interrupted is False


def test_ssh_manager_creation():
    """Test that SSHManager can be created."""
    from chisel.ssh_manager import SSHManager
    
    # This will create real state directory but that's ok for smoke test
    ssh_manager = SSHManager()
    assert ssh_manager is not None
    assert ssh_manager.state is not None