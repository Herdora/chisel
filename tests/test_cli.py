"""Tests for CLI commands."""

import pytest
from unittest.mock import Mock, patch
from typer.testing import CliRunner

from chisel.main import app


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


def test_version_command(runner):
    """Test version command."""
    with patch("chisel.__version__", "0.1.0"):
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout


def test_configure_command_with_token(runner, mock_config):
    """Test configure command with token argument."""
    with patch("chisel.do_client.DOClient") as mock_do_client:
        mock_client = Mock()
        mock_client.validate_token.return_value = (True, {"account": {"email": "test@example.com"}})
        mock_client.get_balance.return_value = {"balance": {"account_balance": "10.00"}}
        mock_do_client.return_value = mock_client
        
        result = runner.invoke(app, ["configure", "--token", "test-token"])
        
        assert result.exit_code == 0
        assert "validated successfully" in result.stdout


def test_configure_command_invalid_token(runner, mock_config):
    """Test configure command with invalid token."""
    with patch("chisel.do_client.DOClient") as mock_do_client:
        mock_client = Mock()
        mock_client.validate_token.return_value = (False, None)
        mock_do_client.return_value = mock_client
        
        result = runner.invoke(app, ["configure", "--token", "invalid-token"])
        
        assert result.exit_code == 1
        assert "Invalid API token" in result.stdout


def test_up_command_no_config(runner):
    """Test up command without configuration."""
    with patch("chisel.config.Config") as mock_config_class:
        mock_config = Mock()
        mock_config.token = None
        mock_config_class.return_value = mock_config
        
        result = runner.invoke(app, ["up"])
        
        assert result.exit_code == 1
        assert "No API token configured" in result.stdout


def test_sync_command(runner):
    """Test sync command."""
    with patch("chisel.ssh_manager.SSHManager") as mock_ssh_manager:
        mock_manager = Mock()
        mock_manager.sync.return_value = True
        mock_ssh_manager.return_value = mock_manager
        
        result = runner.invoke(app, ["sync", "test.txt"])
        
        assert result.exit_code == 0
        mock_manager.sync.assert_called_once_with("test.txt", None)


def test_sync_command_with_destination(runner):
    """Test sync command with destination."""
    with patch("chisel.ssh_manager.SSHManager") as mock_ssh_manager:
        mock_manager = Mock()
        mock_manager.sync.return_value = True
        mock_ssh_manager.return_value = mock_manager
        
        result = runner.invoke(app, ["sync", "test.txt", "--dest", "/tmp/"])
        
        assert result.exit_code == 0
        mock_manager.sync.assert_called_once_with("test.txt", "/tmp/")


def test_run_command(runner):
    """Test run command."""
    with patch("chisel.ssh_manager.SSHManager") as mock_ssh_manager:
        mock_manager = Mock()
        mock_manager.run.return_value = 0
        mock_ssh_manager.return_value = mock_manager
        
        result = runner.invoke(app, ["run", "echo hello"])
        
        assert result.exit_code == 0
        mock_manager.run.assert_called_once_with("echo hello")


def test_pull_command(runner):
    """Test pull command."""
    with patch("chisel.ssh_manager.SSHManager") as mock_ssh_manager:
        mock_manager = Mock()
        mock_manager.pull.return_value = True
        mock_ssh_manager.return_value = mock_manager
        
        result = runner.invoke(app, ["pull", "/remote/file.txt"])
        
        assert result.exit_code == 0
        mock_manager.pull.assert_called_once_with("/remote/file.txt", None)


def test_pull_command_with_local_path(runner):
    """Test pull command with local path."""
    with patch("chisel.ssh_manager.SSHManager") as mock_ssh_manager:
        mock_manager = Mock()
        mock_manager.pull.return_value = True
        mock_ssh_manager.return_value = mock_manager
        
        result = runner.invoke(app, ["pull", "/remote/file.txt", "--local", "./local_file.txt"])
        
        assert result.exit_code == 0
        mock_manager.pull.assert_called_once_with("/remote/file.txt", "./local_file.txt")


def test_profile_command(runner):
    """Test profile command."""
    with patch("chisel.ssh_manager.SSHManager") as mock_ssh_manager:
        mock_manager = Mock()
        mock_manager.profile.return_value = "/path/to/results"
        mock_ssh_manager.return_value = mock_manager
        
        result = runner.invoke(app, ["profile", "my-binary"])
        
        assert result.exit_code == 0
        mock_manager.profile.assert_called_once()


def test_list_command_no_config(runner):
    """Test list command without configuration."""
    with patch("chisel.config.Config") as mock_config_class:
        mock_config = Mock()
        mock_config.token = None
        mock_config_class.return_value = mock_config
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 1
        assert "No API token configured" in result.stdout