"""Tests for Droplet Manager."""

import pytest
from unittest.mock import Mock, patch, call
from datetime import datetime, timezone

from chisel.droplet import DropletManager


@pytest.fixture
def mock_do_client():
    """Create a mock DO client."""
    return Mock()


@pytest.fixture
def mock_state():
    """Create a mock state manager."""
    with patch("chisel.droplet.State") as mock_state_class:
        mock_state = Mock()
        mock_state_class.return_value = mock_state
        yield mock_state


@pytest.fixture
def droplet_manager(mock_do_client, mock_state):
    """Create a DropletManager instance with mocks."""
    return DropletManager(mock_do_client)


def test_droplet_manager_init(mock_do_client, mock_state):
    """Test DropletManager initialization."""
    manager = DropletManager(mock_do_client)
    
    assert manager.client == mock_do_client
    assert manager.state == mock_state
    assert manager.droplet_name == "chisel-dev"
    assert manager.droplet_size == "gpu-mi300x1-192gb"


def test_get_ssh_keys(droplet_manager, mock_do_client):
    """Test getting SSH keys."""
    mock_keys = [
        {"id": 1, "fingerprint": "aa:bb:cc"},
        {"id": 2, "fingerprint": "dd:ee:ff"}
    ]
    mock_do_client.client.ssh_keys.list.return_value = {"ssh_keys": mock_keys}
    
    with patch("requests.get") as mock_requests:
        mock_requests.return_value.status_code = 404  # Not on DO droplet
        
        key_ids = droplet_manager.get_ssh_keys()
        
        assert key_ids == [1, 2]
        mock_do_client.client.ssh_keys.list.assert_called_once()


def test_get_ssh_keys_empty(droplet_manager, mock_do_client):
    """Test getting SSH keys when none exist."""
    mock_do_client.client.ssh_keys.list.return_value = {"ssh_keys": []}
    
    with patch("requests.get") as mock_requests:
        mock_requests.return_value.status_code = 404  # Not on DO droplet
        
        key_ids = droplet_manager.get_ssh_keys()
        
        assert key_ids == []


def test_find_existing_droplet_found(droplet_manager):
    """Test finding an existing droplet."""
    mock_droplet = {"id": 123, "name": "chisel-dev", "status": "active"}
    
    with patch.object(droplet_manager, 'list_droplets') as mock_list:
        mock_list.return_value = [mock_droplet]
        
        droplet = droplet_manager.find_existing_droplet()
        
        assert droplet == mock_droplet


def test_find_existing_droplet_not_found(droplet_manager):
    """Test finding existing droplet when none exists."""
    with patch.object(droplet_manager, 'list_droplets') as mock_list:
        mock_list.return_value = []
        
        droplet = droplet_manager.find_existing_droplet()
        
        assert droplet is None


def test_find_existing_droplet_multiple(droplet_manager):
    """Test finding existing droplet when multiple exist."""
    mock_droplets = [
        {"id": 1, "name": "chisel-dev", "status": "active"},
        {"id": 2, "name": "chisel-dev", "status": "active"}
    ]
    
    with patch.object(droplet_manager, 'list_droplets') as mock_list:
        mock_list.return_value = mock_droplets
        
        droplet = droplet_manager.find_existing_droplet()
        
        # Should return the first one
        assert droplet == mock_droplets[0]


def test_create_droplet_success(droplet_manager, mock_do_client):
    """Test successful droplet creation."""
    mock_keys = [{"id": 123, "fingerprint": "aa:bb:cc"}]
    mock_do_client.client.ssh_keys.list.return_value = {"ssh_keys": mock_keys}
    
    mock_create_response = {
        "droplet": {
            "id": 456,
            "name": "chisel-dev",
            "status": "new"
        }
    }
    mock_do_client.client.droplets.create.return_value = mock_create_response
    
    with patch("requests.get") as mock_requests:
        mock_requests.return_value.status_code = 404  # Not on DO droplet
        
        droplet = droplet_manager.create_droplet()
        
        assert droplet == mock_create_response["droplet"]
        mock_do_client.client.droplets.create.assert_called_once()


def test_list_droplets(droplet_manager, mock_do_client):
    """Test listing droplets."""
    mock_droplets = [
        {"id": 1, "name": "chisel-dev"},
        {"id": 2, "name": "chisel-test"}
    ]
    
    # Mock the client response structure
    mock_response = {"droplets": mock_droplets}
    mock_do_client.client.droplets.list.return_value = mock_response
    
    droplets = droplet_manager.list_droplets()
    
    assert droplets == mock_droplets


def test_wait_for_droplet_success(droplet_manager, mock_do_client):
    """Test waiting for droplet to become active."""
    # Simulate droplet becoming active after a few checks
    active_droplet = {
        "id": 123,
        "status": "active",
        "networks": {"v4": [{"type": "public", "ip_address": "1.2.3.4"}]}
    }
    
    mock_responses = [
        {"droplet": {"status": "new"}},
        {"droplet": {"status": "new"}},
        {"droplet": active_droplet}
    ]
    mock_do_client.client.droplets.get.side_effect = mock_responses
    
    with patch("time.sleep"):  # Mock sleep to speed up test
        result = droplet_manager.wait_for_droplet(123)
        
        assert result == active_droplet
        assert mock_do_client.client.droplets.get.call_count == 3


def test_wait_for_droplet_timeout(droplet_manager, mock_do_client):
    """Test timeout when waiting for droplet."""
    # Droplet never becomes active
    mock_do_client.client.droplets.get.return_value = {"droplet": {"status": "new"}}
    
    with patch("time.sleep"):  # Mock sleep to speed up test
        with pytest.raises(Exception, match="Droplet failed to become active"):
            droplet_manager.wait_for_droplet(123, timeout=2)


def test_wait_for_ssh_success(droplet_manager):
    """Test waiting for SSH to become available."""
    mock_socket = Mock()
    mock_socket.connect.side_effect = [
        ConnectionRefusedError,  # First try fails
        None,  # Second try succeeds
    ]
    
    with patch("socket.socket", return_value=mock_socket), \
         patch("time.sleep"):
        
        result = droplet_manager.wait_for_ssh("1.2.3.4")
        
        assert result is True
        assert mock_socket.connect.call_count == 2


def test_wait_for_ssh_timeout(droplet_manager):
    """Test SSH timeout."""
    mock_socket = Mock()
    mock_socket.connect.side_effect = ConnectionRefusedError
    
    with patch("socket.socket", return_value=mock_socket), \
         patch("time.sleep"):
        
        result = droplet_manager.wait_for_ssh("1.2.3.4", timeout=2)
        
        assert result is False


def test_up_with_existing_droplet(droplet_manager, mock_state):
    """Test up command with existing droplet."""
    existing_droplet = {
        "id": 123,
        "name": "chisel-dev",
        "status": "active",
        "networks": {"v4": [{"type": "public", "ip_address": "1.2.3.4"}]}
    }
    
    with patch.object(droplet_manager, 'find_existing_droplet') as mock_find:
        mock_find.return_value = existing_droplet
        
        result = droplet_manager.up()
        
        assert result == existing_droplet
        # Should save to state
        mock_state.save.assert_called_once()


def test_up_create_new_droplet(droplet_manager, mock_state):
    """Test up command creating new droplet."""
    # No existing droplet
    created_droplet = {
        "id": 456,
        "name": "chisel-dev",
        "status": "new"
    }
    
    active_droplet = {
        "id": 456,
        "name": "chisel-dev",
        "status": "active",
        "networks": {"v4": [{"type": "public", "ip_address": "5.6.7.8"}]}
    }
    
    with patch.object(droplet_manager, 'find_existing_droplet') as mock_find, \
         patch.object(droplet_manager, 'create_droplet') as mock_create, \
         patch.object(droplet_manager, 'wait_for_droplet') as mock_wait, \
         patch.object(droplet_manager, 'wait_for_ssh') as mock_ssh:
        
        mock_find.return_value = None
        mock_create.return_value = created_droplet
        mock_wait.return_value = active_droplet
        mock_ssh.return_value = True
        
        result = droplet_manager.up()
        
        assert result == active_droplet
        mock_state.save.assert_called_once()


def test_down_with_state_droplet(droplet_manager, mock_do_client, mock_state):
    """Test down command with droplet in state."""
    state_info = {
        "droplet_id": 123,
        "name": "chisel-dev",
        "ip": "1.2.3.4"
    }
    mock_state.get_droplet_info.return_value = state_info
    
    # Droplet exists
    mock_do_client.client.droplets.get.return_value = {
        "droplet": {"id": 123, "name": "chisel-dev"}
    }
    
    with patch("chisel.droplet.console"):
        result = droplet_manager.down()
        
        assert result is True
        mock_do_client.client.droplets.destroy.assert_called_once_with(123)
        mock_state.clear.assert_called_once()


def test_down_no_droplet(droplet_manager, mock_state):
    """Test down command with no droplet."""
    mock_state.get_droplet_info.return_value = None
    
    with patch("chisel.droplet.console"):
        result = droplet_manager.down()
        
        assert result is True


def test_down_droplet_already_destroyed(droplet_manager, mock_do_client, mock_state):
    """Test down command when droplet already destroyed."""
    state_info = {
        "droplet_id": 123,
        "name": "chisel-dev",
        "ip": "1.2.3.4"
    }
    mock_state.get_droplet_info.return_value = state_info
    
    # Droplet doesn't exist (404)
    mock_do_client.client.droplets.get.side_effect = Exception("404")
    
    with patch("chisel.droplet.console"):
        result = droplet_manager.down()
        
        assert result is True
        mock_state.clear.assert_called_once()
        mock_do_client.client.droplets.destroy.assert_not_called()


def test_destroy_droplet_success(droplet_manager, mock_do_client):
    """Test successful droplet destruction."""
    droplet_manager.destroy_droplet(123)
    
    mock_do_client.client.droplets.destroy.assert_called_once_with(123)