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


def test_up_command_success(runner, mock_config):
    """Test successful up command."""
    with patch("chisel.do_client.DOClient") as mock_do_client, \
         patch("chisel.droplet.DropletManager") as mock_droplet_manager:
        
        mock_client = Mock()
        mock_do_client.return_value = mock_client
        
        mock_manager = Mock()
        mock_manager.up.return_value = {
            "id": 123,
            "name": "chisel-dev",
            "ip": "1.2.3.4",
            "region": {"slug": "sfo3"},
            "size": {"slug": "gpu-mi300x-1x"}
        }
        mock_droplet_manager.return_value = mock_manager
        
        result = runner.invoke(app, ["up"])
        
        assert result.exit_code == 0
        assert "Droplet is ready!" in result.stdout
        assert "1.2.3.4" in result.stdout


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


def test_list_command_with_droplets(runner, mock_config, mock_droplet_info):
    """Test list command with existing droplets."""
    with patch("chisel.do_client.DOClient") as mock_do_client, \
         patch("chisel.droplet.DropletManager") as mock_droplet_manager:
        
        mock_client = Mock()
        mock_do_client.return_value = mock_client
        
        mock_manager = Mock()
        mock_manager.list_droplets.return_value = [mock_droplet_info]
        mock_manager.state = Mock()
        mock_manager.state.get_droplet_info.return_value = {
            "droplet_id": mock_droplet_info["id"],
            "name": mock_droplet_info["name"],
            "ip": "1.2.3.4"
        }
        mock_droplet_manager.return_value = mock_manager
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        assert "chisel-dev" in result.stdout
        assert "1.2.3.4" in result.stdout
        assert "Active droplet from state" in result.stdout


def test_down_command_no_config(runner):
    """Test down command without configuration."""
    with patch("chisel.config.Config") as mock_config_class:
        mock_config = Mock()
        mock_config.token = None
        mock_config_class.return_value = mock_config
        
        result = runner.invoke(app, ["down"])
        
        assert result.exit_code == 1
        assert "No API token configured" in result.stdout


def test_down_command_with_droplet(runner, mock_config):
    """Test down command with existing droplet."""
    with patch("chisel.do_client.DOClient") as mock_do_client, \
         patch("chisel.droplet.DropletManager") as mock_droplet_manager:
        
        mock_client = Mock()
        mock_do_client.return_value = mock_client
        
        mock_manager = Mock()
        mock_manager.down.return_value = True
        mock_droplet_manager.return_value = mock_manager
        
        # Simulate user confirming destruction
        result = runner.invoke(app, ["down"], input="y\n")
        
        assert result.exit_code == 0
        mock_manager.down.assert_called_once()


def test_down_command_cancelled(runner, mock_config):
    """Test down command when user cancels."""
    with patch("chisel.do_client.DOClient") as mock_do_client, \
         patch("chisel.droplet.DropletManager") as mock_droplet_manager:
        
        mock_client = Mock()
        mock_do_client.return_value = mock_client
        
        mock_manager = Mock()
        mock_droplet_manager.return_value = mock_manager
        
        # Simulate user cancelling destruction
        result = runner.invoke(app, ["down"], input="n\n")
        
        assert result.exit_code == 0
        assert "Cancelled" in result.stdout
        mock_manager.down.assert_not_called()


def test_sweep_command_no_config(runner):
    """Test sweep command without configuration."""
    with patch("chisel.config.Config") as mock_config_class:
        mock_config = Mock()
        mock_config.token = None
        mock_config_class.return_value = mock_config
        
        result = runner.invoke(app, ["sweep"])
        
        assert result.exit_code == 1
        assert "No API token configured" in result.stdout


def test_sweep_command_no_droplets(runner, mock_config):
    """Test sweep command with no droplets."""
    with patch("chisel.do_client.DOClient") as mock_do_client, \
         patch("chisel.droplet.DropletManager") as mock_droplet_manager:
        
        mock_client = Mock()
        mock_do_client.return_value = mock_client
        
        mock_manager = Mock()
        mock_manager.list_droplets.return_value = []
        mock_droplet_manager.return_value = mock_manager
        
        result = runner.invoke(app, ["sweep"])
        
        assert result.exit_code == 0
        assert "No chisel droplets found" in result.stdout


def test_sweep_command_with_old_droplets(runner, mock_config, mock_droplet_info):
    """Test sweep command with old droplets."""
    from datetime import datetime, timezone, timedelta
    
    # Create droplet data with old creation time
    old_time = datetime.now(timezone.utc) - timedelta(hours=8)
    old_droplet = dict(mock_droplet_info)
    old_droplet["created_at"] = old_time.isoformat()
    
    with patch("chisel.do_client.DOClient") as mock_do_client, \
         patch("chisel.droplet.DropletManager") as mock_droplet_manager:
        
        mock_client = Mock()
        mock_do_client.return_value = mock_client
        
        mock_manager = Mock()
        mock_manager.list_droplets.return_value = [old_droplet]
        mock_manager.destroy_droplet.return_value = None
        mock_manager.state = Mock()
        mock_manager.state.get_droplet_info.return_value = None
        mock_droplet_manager.return_value = mock_manager
        
        # Run with auto-confirm
        result = runner.invoke(app, ["sweep", "--yes"])
        
        assert result.exit_code == 0
        assert "Successfully destroyed" in result.stdout
        mock_manager.destroy_droplet.assert_called_once_with(old_droplet["id"])


def test_ssh_setup_command_no_config(runner):
    """Test ssh-setup command without configuration."""
    with patch("chisel.config.Config") as mock_config_class:
        mock_config = Mock()
        mock_config.token = None
        mock_config_class.return_value = mock_config
        
        result = runner.invoke(app, ["ssh-setup"])
        
        assert result.exit_code == 1
        assert "No API token configured" in result.stdout


def test_ssh_setup_command_with_existing_key(runner, mock_config, tmp_path):
    """Test ssh-setup command with existing SSH key."""
    # Create a fake SSH key
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    pub_key_path = ssh_dir / "id_ed25519.pub"
    pub_key_path.write_text("ssh-ed25519 AAAAC3... test@example.com")
    
    with patch("os.path.expanduser", return_value=str(tmp_path)), \
         patch("chisel.do_client.DOClient") as mock_do_client:
        
        mock_client = Mock()
        mock_client.client = Mock()
        mock_client.client.ssh_keys = Mock()
        mock_client.client.ssh_keys.create.return_value = {"ssh_key": {"id": 123}}
        mock_do_client.return_value = mock_client
        
        result = runner.invoke(app, ["ssh-setup"])
        
        assert result.exit_code == 0
        assert "Your SSH public key" in result.stdout
        assert "Successfully added SSH key" in result.stdout