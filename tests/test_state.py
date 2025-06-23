"""Tests for state management."""

import json
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime, timezone

from chisel.state import State


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create a temporary state directory."""
    return tmp_path


@pytest.fixture
def state_with_file(temp_state_dir):
    """Create a State instance with a temporary file."""
    with patch("chisel.state.Path.home", return_value=temp_state_dir):
        return State()


def test_state_init_creates_directory(temp_state_dir):
    """Test State initialization creates state directory if it doesn't exist."""
    with patch("chisel.state.Path.home", return_value=temp_state_dir):
        state = State()
        assert state.state_dir.exists()
        assert (state.state_dir / "state.json").parent.exists()


def test_state_load_existing_file(temp_state_dir):
    """Test State loads existing state file."""
    # Create existing state file
    cache_dir = temp_state_dir / ".cache" / "chisel"
    cache_dir.mkdir(parents=True)
    state_file = cache_dir / "state.json"
    existing_data = {"droplet_id": 123, "name": "test", "ip": "1.2.3.4"}
    state_file.write_text(json.dumps(existing_data))
    
    with patch("chisel.state.Path.home", return_value=temp_state_dir):
        state = State()
        loaded_data = state.load()
        assert loaded_data == existing_data


def test_state_load_handles_invalid_json(temp_state_dir):
    """Test State load handles invalid JSON gracefully."""
    # Create state file with invalid JSON
    cache_dir = temp_state_dir / ".cache" / "chisel"
    cache_dir.mkdir(parents=True)
    state_file = cache_dir / "state.json"
    state_file.write_text("invalid json")
    
    with patch("chisel.state.Path.home", return_value=temp_state_dir):
        state = State()
        assert state.load() == {}


def test_save_droplet_info(state_with_file):
    """Test saving droplet information."""
    created_at = datetime.now(timezone.utc).isoformat()
    
    state_with_file.save(
        droplet_id=12345,
        ip="192.168.1.1",
        name="chisel-dev",
        created_at=created_at
    )
    
    # Check file was written
    saved_data = json.loads(state_with_file.state_file.read_text())
    assert saved_data["droplet_id"] == 12345
    assert saved_data["name"] == "chisel-dev"
    assert saved_data["ip"] == "192.168.1.1"
    assert saved_data["created_at"] == created_at
    assert "saved_at" in saved_data


def test_save_without_created_at(state_with_file):
    """Test saving droplet information without created_at timestamp."""
    state_with_file.save(
        droplet_id=12345,
        ip="192.168.1.1",
        name="chisel-dev"
    )
    
    # Check file was written
    saved_data = json.loads(state_with_file.state_file.read_text())
    assert saved_data["droplet_id"] == 12345
    assert saved_data["name"] == "chisel-dev"
    assert saved_data["ip"] == "192.168.1.1"
    assert "created_at" in saved_data  # Should auto-generate
    assert "saved_at" in saved_data


def test_get_droplet_info(state_with_file):
    """Test getting droplet information."""
    # Save some data first
    state_with_file.save(
        droplet_id=123,
        ip="1.2.3.4",
        name="test-droplet"
    )
    
    # Get the info
    info = state_with_file.get_droplet_info()
    
    assert info["droplet_id"] == 123
    assert info["name"] == "test-droplet"
    assert info["ip"] == "1.2.3.4"


def test_get_droplet_info_empty(state_with_file):
    """Test getting droplet information when empty."""
    info = state_with_file.get_droplet_info()
    assert info is None


def test_get_droplet_info_partial(state_with_file):
    """Test getting droplet information with partial data."""
    # Save only partial data
    state_data = {"droplet_id": 123}  # Missing ip and name
    state_with_file.state_file.write_text(json.dumps(state_data))
    
    info = state_with_file.get_droplet_info()
    assert info is None  # Should return None if not all required fields present


def test_clear(state_with_file):
    """Test clearing state."""
    # Add some data
    state_with_file.save(
        droplet_id=123,
        ip="1.2.3.4",
        name="test"
    )
    
    # Verify file exists
    assert state_with_file.state_file.exists()
    
    # Clear it
    state_with_file.clear()
    
    # Check file is deleted
    assert not state_with_file.state_file.exists()


def test_concurrent_access(state_with_file):
    """Test that state handles concurrent access gracefully."""
    # Save initial data
    state_with_file.save(
        droplet_id=123,
        ip="1.2.3.4",
        name="test"
    )
    
    # Simulate another process modifying the file
    other_data = {
        "droplet_id": 456, 
        "name": "other", 
        "ip": "5.6.7.8",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    state_with_file.state_file.write_text(json.dumps(other_data))
    
    # Load fresh data
    loaded = state_with_file.load()
    assert loaded["droplet_id"] == 456
    
    # Get droplet info should show new data
    info = state_with_file.get_droplet_info()
    assert info["droplet_id"] == 456


def test_get_droplet_uptime_hours(state_with_file):
    """Test getting droplet uptime in hours."""
    # Save with specific created_at time
    created_at = datetime.now(timezone.utc)
    created_at_str = created_at.isoformat()
    
    state_with_file.save(
        droplet_id=123,
        ip="1.2.3.4",
        name="test",
        created_at=created_at_str
    )
    
    # Mock current time to be 2.5 hours later
    future_time = created_at.timestamp() + (2.5 * 3600)
    with patch("chisel.state.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime.fromtimestamp(future_time, tz=timezone.utc)
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        uptime = state_with_file.get_droplet_uptime_hours()
        assert abs(uptime - 2.5) < 0.01  # Allow small floating point difference


def test_get_droplet_uptime_hours_no_state(state_with_file):
    """Test getting droplet uptime when no state exists."""
    assert state_with_file.get_droplet_uptime_hours() == 0.0


def test_get_droplet_uptime_hours_no_created_at(state_with_file):
    """Test getting droplet uptime when created_at is missing."""
    # Save state without created_at
    state_data = {
        "droplet_id": 123,
        "ip": "1.2.3.4",
        "name": "test"
    }
    state_with_file.state_file.write_text(json.dumps(state_data))
    
    assert state_with_file.get_droplet_uptime_hours() == 0.0


def test_get_estimated_cost(state_with_file):
    """Test getting estimated cost."""
    # Save with specific created_at time
    created_at = datetime.now(timezone.utc)
    created_at_str = created_at.isoformat()
    
    state_with_file.save(
        droplet_id=123,
        ip="1.2.3.4",
        name="test",
        created_at=created_at_str
    )
    
    # Mock current time to be 10 hours later
    future_time = created_at.timestamp() + (10 * 3600)
    with patch("chisel.state.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime.fromtimestamp(future_time, tz=timezone.utc)
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        # Default rate is $1.99/hour
        cost = state_with_file.get_estimated_cost()
        assert abs(cost - 19.90) < 0.01
        
        # Test with custom rate
        cost = state_with_file.get_estimated_cost(hourly_rate=2.50)
        assert abs(cost - 25.00) < 0.01


def test_should_warn_cost(state_with_file):
    """Test cost warning logic."""
    # Save with specific created_at time
    created_at = datetime.now(timezone.utc)
    created_at_str = created_at.isoformat()
    
    state_with_file.save(
        droplet_id=123,
        ip="1.2.3.4",
        name="test",
        created_at=created_at_str
    )
    
    # Test with 8 hours uptime (no warning)
    future_time = created_at.timestamp() + (8 * 3600)
    with patch("chisel.state.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime.fromtimestamp(future_time, tz=timezone.utc)
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        should_warn, uptime, cost = state_with_file.should_warn_cost()
        assert should_warn is False
        assert abs(uptime - 8.0) < 0.01
        assert abs(cost - 15.92) < 0.01
    
    # Test with 15 hours uptime (warning)
    future_time = created_at.timestamp() + (15 * 3600)
    with patch("chisel.state.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime.fromtimestamp(future_time, tz=timezone.utc)
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        should_warn, uptime, cost = state_with_file.should_warn_cost()
        assert should_warn is True
        assert abs(uptime - 15.0) < 0.01
        assert abs(cost - 29.85) < 0.01
    
    # Test with custom warning hours
    should_warn, uptime, cost = state_with_file.should_warn_cost(warning_hours=20.0)
    assert should_warn is False  # 15 hours < 20 hours