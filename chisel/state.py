"""State management for chisel droplets."""

import json
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

    def save(self, droplet_id: int, ip: str, name: str) -> None:
        """Save droplet state."""
        state = {"droplet_id": droplet_id, "ip": ip, "name": name}

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
