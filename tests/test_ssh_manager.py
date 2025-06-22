"""Tests for SSH manager."""

import pytest
from unittest.mock import Mock, patch, call
from pathlib import Path
import subprocess

from chisel.ssh_manager import SSHManager, InterruptHandler


def test_ssh_manager_init(mock_state):
    """Test SSH manager initialization."""
    ssh_manager = SSHManager()
    assert ssh_manager.state is not None


def test_get_droplet_info_no_droplet(mock_state):
    """Test getting droplet info when no droplet exists."""
    mock_state.get_droplet_info.return_value = None
    
    ssh_manager = SSHManager()
    result = ssh_manager.get_droplet_info()
    
    assert result is None
    mock_state.get_droplet_info.assert_called_once()


def test_get_droplet_info_with_droplet(mock_state, mock_droplet_info):
    """Test getting droplet info when droplet exists."""
    mock_state.get_droplet_info.return_value = mock_droplet_info
    
    ssh_manager = SSHManager()
    result = ssh_manager.get_droplet_info()
    
    assert result == mock_droplet_info
    mock_state.get_droplet_info.assert_called_once()


def test_sync_no_droplet(mock_state):
    """Test sync when no droplet is available."""
    mock_state.get_droplet_info.return_value = None
    
    ssh_manager = SSHManager()
    result = ssh_manager.sync("test.txt")
    
    assert result is False


def test_sync_nonexistent_source(mock_state, mock_droplet_info):
    """Test sync with nonexistent source file."""
    mock_state.get_droplet_info.return_value = mock_droplet_info
    
    with patch("pathlib.Path.exists", return_value=False):
        ssh_manager = SSHManager()
        result = ssh_manager.sync("nonexistent.txt")
        
        assert result is False


@patch("subprocess.run")
def test_sync_success(mock_run, mock_state, mock_droplet_info, temp_dir):
    """Test successful file sync."""
    mock_state.get_droplet_info.return_value = mock_droplet_info
    mock_run.return_value = Mock(returncode=0)
    
    # Create a test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    
    with patch("pathlib.Path.resolve", return_value=test_file):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=False):
                ssh_manager = SSHManager()
                result = ssh_manager.sync(str(test_file))
                
                assert result is True
                mock_run.assert_called_once()
                
                # Check rsync command
                args = mock_run.call_args[0][0]
                assert "rsync" == args[0]
                assert "-avz" in args
                assert "--progress" in args


def test_pull_no_droplet(mock_state):
    """Test pull when no droplet is available."""
    mock_state.get_droplet_info.return_value = None
    
    ssh_manager = SSHManager()
    result = ssh_manager.pull("/remote/path")
    
    assert result is False


@patch("subprocess.run")
def test_pull_success(mock_run, mock_ssh_client, mock_state, mock_droplet_info):
    """Test successful file pull."""
    mock_state.get_droplet_info.return_value = mock_droplet_info
    mock_run.return_value = Mock(returncode=0)
    
    # Mock SSH commands for checking remote file
    mock_stdout_exists = Mock()
    mock_stdout_exists.read.return_value = b"exists"
    mock_stdout_type = Mock()
    mock_stdout_type.read.return_value = b"file"
    
    mock_ssh_client.exec_command.side_effect = [
        (None, mock_stdout_exists, None),  # Check if exists
        (None, mock_stdout_type, None),    # Check if file or dir
    ]
    
    with patch("pathlib.Path.parent") as mock_parent:
        mock_parent.mkdir = Mock()
        
        ssh_manager = SSHManager()
        result = ssh_manager.pull("/remote/file.txt", "./local_file.txt")
        
        assert result is True
        mock_run.assert_called_once()
        
        # Check scp command
        args = mock_run.call_args[0][0]
        assert "scp" == args[0]
        assert f"root@{mock_droplet_info['ip']}:/remote/file.txt" in args


def test_interrupt_handler():
    """Test interrupt handler functionality."""
    handler = InterruptHandler()
    
    # Test initial state
    assert handler.interrupted is False
    
    # Test signal handling
    handler._signal_handler(2, None)  # SIGINT
    assert handler.interrupted is True
    
    # Test check_interrupted raises exception
    with pytest.raises(KeyboardInterrupt):
        handler.check_interrupted()


def test_interrupt_handler_context_manager():
    """Test interrupt handler as context manager."""
    with patch("signal.signal") as mock_signal:
        old_handler = Mock()
        mock_signal.return_value = old_handler
        
        with InterruptHandler() as handler:
            # Should set signal handler
            mock_signal.assert_called()
            assert not handler.interrupted
        
        # Should restore old handler
        restore_call = mock_signal.call_args_list[-1]
        assert restore_call[0][1] == old_handler