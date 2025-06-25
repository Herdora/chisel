"""State management for the profiling system."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


class ProfilingState:
    """Manages state for the simplified profiling system."""

    def __init__(self):
        self.state_dir = Path.home() / ".chisel"
        self.state_file = self.state_dir / "profile_state.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, any]:
        """Load state from disk."""
        if not self.state_file.exists():
            return {"droplets": {}}

        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"droplets": {}}

    def save(self, state: Dict[str, any]):
        """Save state to disk."""
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def get_droplet(self, gpu_type: str) -> Optional[Dict[str, any]]:
        """Get droplet info if it exists and hasn't timed out."""
        state = self.load()

        if gpu_type not in state["droplets"]:
            return None

        droplet_info = state["droplets"][gpu_type]

        # Check if droplet has timed out (15 minutes)
        if self.should_cleanup(gpu_type):
            # Don't return timed out droplets
            return None

        return droplet_info

    def save_droplet(self, gpu_type: str, droplet_info: Dict[str, any]):
        """Save droplet information."""
        state = self.load()

        # Add timestamp
        droplet_info["last_activity"] = datetime.now(timezone.utc).isoformat()

        state["droplets"][gpu_type] = droplet_info
        self.save(state)

    def update_activity(self, gpu_type: str):
        """Update the last activity timestamp for a droplet."""
        state = self.load()

        if gpu_type in state["droplets"]:
            state["droplets"][gpu_type]["last_activity"] = datetime.now(
                timezone.utc
            ).isoformat()
            self.save(state)

    def should_cleanup(self, gpu_type: str, timeout_minutes: int = 15) -> bool:
        """Check if a droplet should be cleaned up due to inactivity."""
        state = self.load()

        if gpu_type not in state["droplets"]:
            return False

        droplet_info = state["droplets"][gpu_type]

        if "last_activity" not in droplet_info:
            return True

        try:
            last_activity = datetime.fromisoformat(droplet_info["last_activity"])
            now = datetime.now(timezone.utc)
            inactive_minutes = (now - last_activity).total_seconds() / 60

            return inactive_minutes > timeout_minutes
        except (ValueError, TypeError):
            return True

    def remove_droplet(self, gpu_type: str):
        """Remove a droplet from state."""
        state = self.load()

        if gpu_type in state["droplets"]:
            del state["droplets"][gpu_type]
            self.save(state)

    def get_all_droplets(self) -> Dict[str, Dict[str, any]]:
        """Get all tracked droplets."""
        state = self.load()
        return state.get("droplets", {})
