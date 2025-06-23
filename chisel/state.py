"""State management for chisel droplets."""

import json
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
            return {"droplets": {}}

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                # Handle legacy single-droplet format
                if "droplet_id" in state and "droplets" not in state:
                    # This is old format, clear it and start fresh
                    return {"droplets": {}}
                # Ensure droplets key exists
                if "droplets" not in state:
                    state["droplets"] = {}
                return state
        except (json.JSONDecodeError, IOError):
            return {"droplets": {}}

    def save_droplet(
        self, gpu_type: str, droplet_id: int, ip: str, name: str, created_at: Optional[str] = None
    ) -> None:
        """Save droplet state for specific GPU type."""
        # If no creation time provided, use current time
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()

        state = self.load()
        state["droplets"][gpu_type] = {
            "droplet_id": droplet_id,
            "ip": ip,
            "name": name,
            "created_at": created_at,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "gpu_type": gpu_type,
        }

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def get_droplet_info(self, gpu_type: str) -> Optional[Dict[str, Any]]:
        """Get droplet info for specific GPU type."""
        state = self.load()
        droplet_info = state.get("droplets", {}).get(gpu_type)
        if droplet_info and all(k in droplet_info for k in ["droplet_id", "ip", "name"]):
            return droplet_info
        return None

    def get_all_droplets(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked droplets."""
        state = self.load()
        return state.get("droplets", {})

    def clear_droplet(self, gpu_type: str) -> None:
        """Clear state for specific GPU type."""
        state = self.load()
        if gpu_type in state.get("droplets", {}):
            del state["droplets"][gpu_type]
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

    def clear_all(self) -> None:
        """Clear all droplet state."""
        if self.state_file.exists():
            self.state_file.unlink()

    def get_droplet_uptime_hours(self, gpu_type: str) -> float:
        """Get droplet uptime in hours since creation for specific GPU type."""
        droplet_info = self.get_droplet_info(gpu_type)
        if not droplet_info or "created_at" not in droplet_info:
            return 0.0

        try:
            created_at = datetime.fromisoformat(
                droplet_info["created_at"].replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)
            uptime_seconds = (now - created_at).total_seconds()
            return uptime_seconds / 3600  # Convert to hours
        except (ValueError, TypeError):
            return 0.0

    def get_estimated_cost(self, gpu_type: str, hourly_rate: float = 1.99) -> float:
        """Get estimated cost based on uptime for specific GPU type."""
        uptime_hours = self.get_droplet_uptime_hours(gpu_type)
        return uptime_hours * hourly_rate

    def should_warn_cost(
        self, gpu_type: str, warning_hours: float = 12.0, hourly_rate: float = 1.99
    ) -> tuple[bool, float, float]:
        """
        Check if cost warning should be shown for specific GPU type.

        Returns:
            (should_warn, uptime_hours, estimated_cost)
        """
        uptime_hours = self.get_droplet_uptime_hours(gpu_type)
        estimated_cost = self.get_estimated_cost(gpu_type, hourly_rate)
        should_warn = uptime_hours >= warning_hours

        return should_warn, uptime_hours, estimated_cost
