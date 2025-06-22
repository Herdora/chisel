"""State management for chisel droplets."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class State:
    def __init__(self):
        self.state_dir = Path.home() / ".cache" / "chisel"
        self.state_file = self.state_dir / "state.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Any]:
        """Load state from disk."""
        if not self.state_file.exists():
            return {}

        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def save(self, droplet_id: int, ip: str, name: str, created_at: Optional[str] = None) -> None:
        """Save droplet state."""
        # If no creation time provided, use current time (for backwards compatibility)
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()
        
        state = {
            "droplet_id": droplet_id, 
            "ip": ip, 
            "name": name,
            "created_at": created_at,
            "saved_at": datetime.now(timezone.utc).isoformat()
        }

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def clear(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()

    def get_droplet_info(self) -> Optional[Dict[str, Any]]:
        state = self.load()
        if state and all(k in state for k in ["droplet_id", "ip", "name"]):
            return state
        return None
    
    def get_droplet_uptime_hours(self) -> float:
        """Get droplet uptime in hours since creation."""
        state = self.load()
        if not state or "created_at" not in state:
            return 0.0
        
        try:
            created_at = datetime.fromisoformat(state["created_at"].replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            uptime_seconds = (now - created_at).total_seconds()
            return uptime_seconds / 3600  # Convert to hours
        except (ValueError, TypeError):
            return 0.0
    
    def get_estimated_cost(self, hourly_rate: float = 1.99) -> float:
        """Get estimated cost based on uptime. Default rate is for gpu-mi300x1-192gb."""
        uptime_hours = self.get_droplet_uptime_hours()
        return uptime_hours * hourly_rate
    
    def should_warn_cost(self, warning_hours: float = 12.0, hourly_rate: float = 1.99) -> tuple[bool, float, float]:
        """
        Check if cost warning should be shown. Only warns for droplets running >12 hours.
        
        Returns:
            (should_warn, uptime_hours, estimated_cost)
        """
        uptime_hours = self.get_droplet_uptime_hours()
        estimated_cost = self.get_estimated_cost(hourly_rate)
        should_warn = uptime_hours >= warning_hours
        
        return should_warn, uptime_hours, estimated_cost
