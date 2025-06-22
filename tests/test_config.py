"""Tests for config module."""

import pytest
from unittest.mock import patch, mock_open
from pathlib import Path
import os

from chisel.config import Config


def test_config_init():
    """Test config initialization."""
    with patch("pathlib.Path.exists", return_value=False):
        config = Config()
        assert config.config_file is not None
        assert "chisel" in str(config.config_file)


def test_config_token_from_env():
    """Test getting token from environment variable."""
    with patch.dict(os.environ, {"CHISEL_DO_TOKEN": "env-token-123"}):
        with patch("pathlib.Path.exists", return_value=False):
            config = Config()
            assert config.token == "env-token-123"


def test_config_token_from_file():
    """Test getting token from config file."""
    mock_config_content = 'token = "file-token-123"\n'
    
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_config_content)):
            with patch("toml.load", return_value={"token": "file-token-123"}):
                config = Config()
                assert config.token == "file-token-123"


def test_config_set_token():
    """Test setting token saves to file."""
    with patch("pathlib.Path.exists", return_value=False):
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("toml.dump") as mock_dump:
                    config = Config()
                    config.token = "new-token-456"
                    
                    # Verify file operations
                    mock_mkdir.assert_called_once()
                    mock_file.assert_called()
                    mock_dump.assert_called_once()


def test_config_env_overrides_file():
    """Test that environment variable overrides config file."""
    mock_config_content = 'token = "file-token-123"\n'
    
    with patch.dict(os.environ, {"CHISEL_DO_TOKEN": "env-token-456"}):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=mock_config_content)):
                with patch("toml.load", return_value={"token": "file-token-123"}):
                    config = Config()
                    assert config.token == "env-token-456"