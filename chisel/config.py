"""Configuration management for Chisel."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import toml
from rich.console import Console

console = Console()

class Config:
    """Manage Chisel configuration."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".config" / "chisel"
        self.config_file = self.config_dir / "config.toml"
        self._config: Dict[str, Any] = {}
        self._load()
    
    def _load(self) -> None:
        """Load configuration from file if it exists."""
        if self.config_file.exists():
            try:
                self._config = toml.load(self.config_file)
            except Exception as e:
                console.print(f"[red]Error loading config: {e}[/red]")
                self._config = {}
    
    def _save(self) -> None:
        """Save configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            toml.dump(self._config, f)
    
    @property
    def token(self) -> Optional[str]:
        """Get DigitalOcean API token."""
        # Check environment variable first
        env_token = os.getenv("CHISEL_DO_TOKEN")
        if env_token:
            return env_token
        
        # Fall back to config file
        return self._config.get("digitalocean", {}).get("token")
    
    @token.setter
    def token(self, value: str) -> None:
        """Set DigitalOcean API token."""
        if "digitalocean" not in self._config:
            self._config["digitalocean"] = {}
        self._config["digitalocean"]["token"] = value
        self._save()
    
    def clear(self) -> None:
        """Clear all configuration."""
        self._config = {}
        if self.config_file.exists():
            self.config_file.unlink()