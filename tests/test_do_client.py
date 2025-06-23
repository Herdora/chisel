"""Tests for DigitalOcean client."""

import pytest
from unittest.mock import Mock, patch

from chisel.do_client import DOClient


@pytest.fixture
def mock_pydo_client():
    """Mock pydo client."""
    with patch("chisel.do_client.pydo.Client") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        yield mock_client


def test_do_client_init(mock_pydo_client):
    """Test DOClient initialization."""
    token = "test-token"
    client = DOClient(token)
    
    assert client.token == token
    assert client.client == mock_pydo_client


def test_validate_token_success(mock_pydo_client):
    """Test successful token validation."""
    mock_pydo_client.account.get.return_value = {
        "account": {
            "email": "test@example.com",
            "status": "active",
            "droplet_limit": 25
        }
    }
    
    client = DOClient("test-token")
    valid, account_info = client.validate_token()
    
    assert valid is True
    assert account_info["account"]["email"] == "test@example.com"
    mock_pydo_client.account.get.assert_called_once()


def test_validate_token_failure(mock_pydo_client):
    """Test failed token validation."""
    mock_pydo_client.account.get.side_effect = Exception("Invalid token")
    
    client = DOClient("invalid-token")
    valid, account_info = client.validate_token()
    
    assert valid is False
    assert account_info is None


def test_get_balance_success(mock_pydo_client):
    """Test successful balance retrieval."""
    mock_pydo_client.balance.get.return_value = {
        "balance": {
            "account_balance": "100.00",
            "month_to_date_usage": "25.50"
        }
    }
    
    client = DOClient("test-token")
    balance = client.get_balance()
    
    assert balance["balance"]["account_balance"] == "100.00"
    mock_pydo_client.balance.get.assert_called_once()


def test_get_balance_failure(mock_pydo_client):
    """Test failed balance retrieval."""
    mock_pydo_client.balance.get.side_effect = Exception("API error")
    
    client = DOClient("test-token")
    balance = client.get_balance()
    
    assert balance is None


def test_list_droplets_success(mock_pydo_client):
    """Test successful droplet listing."""
    mock_droplets = {
        "droplets": [
            {"id": 1, "name": "droplet1"},
            {"id": 2, "name": "droplet2"}
        ],
        "links": {},
        "meta": {"total": 2}
    }
    mock_pydo_client.droplets.list.return_value = mock_droplets
    
    client = DOClient("test-token")
    droplets = client.list_droplets()
    
    assert len(droplets) == 2
    assert droplets[0]["name"] == "droplet1"
    mock_pydo_client.droplets.list.assert_called_once()


def test_list_droplets_with_tag(mock_pydo_client):
    """Test droplet listing with tag filter."""
    mock_droplets = {
        "droplets": [{"id": 1, "name": "chisel-dev"}],
        "links": {},
        "meta": {"total": 1}
    }
    mock_pydo_client.droplets.list.return_value = mock_droplets
    
    client = DOClient("test-token")
    droplets = client.list_droplets(tag_name="chisel")
    
    assert len(droplets) == 1
    mock_pydo_client.droplets.list.assert_called_once_with(tag_name="chisel")


def test_list_droplets_pagination(mock_pydo_client):
    """Test droplet listing with pagination."""
    # First page
    page1 = {
        "droplets": [{"id": 1, "name": "droplet1"}],
        "links": {"pages": {"next": "http://api.do.com/droplets?page=2"}},
        "meta": {"total": 2}
    }
    # Second page
    page2 = {
        "droplets": [{"id": 2, "name": "droplet2"}],
        "links": {},
        "meta": {"total": 2}
    }
    
    mock_pydo_client.droplets.list.side_effect = [page1, page2]
    
    client = DOClient("test-token")
    droplets = client.list_droplets()
    
    assert len(droplets) == 2
    assert droplets[0]["name"] == "droplet1"
    assert droplets[1]["name"] == "droplet2"
    assert mock_pydo_client.droplets.list.call_count == 2


def test_create_droplet_success(mock_pydo_client):
    """Test successful droplet creation."""
    create_request = {
        "name": "test-droplet",
        "region": "nyc3",
        "size": "s-1vcpu-1gb",
        "image": "ubuntu-20-04-x64"
    }
    
    mock_response = {
        "droplet": {
            "id": 12345,
            "name": "test-droplet",
            "status": "new"
        }
    }
    mock_pydo_client.droplets.create.return_value = mock_response
    
    client = DOClient("test-token")
    result = client.create_droplet(create_request)
    
    assert result == mock_response
    mock_pydo_client.droplets.create.assert_called_once_with(body=create_request)


def test_get_droplet_success(mock_pydo_client):
    """Test successful droplet retrieval."""
    mock_droplet = {
        "droplet": {
            "id": 12345,
            "name": "test-droplet",
            "status": "active",
            "networks": {
                "v4": [{"type": "public", "ip_address": "1.2.3.4"}]
            }
        }
    }
    mock_pydo_client.droplets.get.return_value = mock_droplet
    
    client = DOClient("test-token")
    droplet = client.get_droplet(12345)
    
    assert droplet["droplet"]["id"] == 12345
    mock_pydo_client.droplets.get.assert_called_once_with(12345)


def test_destroy_droplet_success(mock_pydo_client):
    """Test successful droplet destruction."""
    mock_pydo_client.droplets.destroy.return_value = None
    
    client = DOClient("test-token")
    client.destroy_droplet(12345)
    
    mock_pydo_client.droplets.destroy.assert_called_once_with(12345)


def test_list_ssh_keys_success(mock_pydo_client):
    """Test successful SSH key listing."""
    mock_keys = {
        "ssh_keys": [
            {"id": 1, "name": "key1", "fingerprint": "aa:bb:cc"},
            {"id": 2, "name": "key2", "fingerprint": "dd:ee:ff"}
        ]
    }
    mock_pydo_client.ssh_keys.list.return_value = mock_keys
    
    client = DOClient("test-token")
    keys = client.list_ssh_keys()
    
    assert len(keys) == 2
    assert keys[0]["name"] == "key1"
    mock_pydo_client.ssh_keys.list.assert_called_once()


def test_wait_for_droplet_active(mock_pydo_client):
    """Test waiting for droplet to become active."""
    # Simulate droplet transitioning from new to active
    responses = [
        {"droplet": {"status": "new"}},
        {"droplet": {"status": "new"}},
        {"droplet": {"status": "active", "networks": {"v4": [{"type": "public", "ip_address": "1.2.3.4"}]}}}
    ]
    mock_pydo_client.droplets.get.side_effect = responses
    
    client = DOClient("test-token")
    
    with patch("time.sleep"):  # Mock sleep to speed up test
        active_droplet = client.wait_for_droplet_active(12345, timeout=30)
    
    assert active_droplet["status"] == "active"
    assert mock_pydo_client.droplets.get.call_count == 3


def test_wait_for_droplet_timeout(mock_pydo_client):
    """Test timeout when waiting for droplet."""
    # Droplet never becomes active
    mock_pydo_client.droplets.get.return_value = {"droplet": {"status": "new"}}
    
    client = DOClient("test-token")
    
    with patch("time.sleep"):  # Mock sleep to speed up test
        active_droplet = client.wait_for_droplet_active(12345, timeout=2)
    
    assert active_droplet is None