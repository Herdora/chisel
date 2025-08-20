"""
Run management for experiment tracking.
"""

import os
import queue
import threading
import atexit
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from ..api.client import APIClient, APIError, AuthenticationError
from ..api.auth import AuthManager
from ..constants import (
    KANDC_BACKEND_APP_NAME_ENV_KEY,
    KANDC_JOB_ID_ENV_KEY,
    KANDC_BACKEND_RUN_ENV_KEY,
    KANDC_TRACE_BASE_DIR_ENV_KEY,
)


@dataclass
class SystemInfo:
    """System information for a run."""

    os: str
    os_version: str
    python_version: str
    hostname: str
    cpu_count: int


@dataclass
class RunConfig:
    """Configuration for a run."""

    project: str
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    mode: str = "online"  # online, offline, disabled
    dir: Optional[Path] = None
    job_id: Optional[str] = None

    def flatten_config(self) -> Dict[str, Any]:
        """Flatten nested config dictionary."""

        def _flatten(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten(v, new_key).items())
                else:
                    items.append((new_key, v))

            return dict(items)

        return _flatten(self.config)


class MetricsQueue:
    """Thread-safe queue for metrics logging."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._metrics_cache: List[Dict[str, Any]] = []

    def start(self):
        """Start the background thread for processing metrics."""
        self._thread = threading.Thread(target=self._process_metrics, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background thread and flush remaining metrics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def log(self, metrics: Dict[str, Any], _: Optional[Any] = None):
        """Add metrics to the queue."""
        # Filter out non-numeric values and keep track of what was filtered
        numeric_metrics = {}
        filtered_metrics = {}

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                numeric_metrics[key] = value
            else:
                filtered_metrics[key] = value

        if filtered_metrics:
            print(f"⚠️  Warning: Filtered out non-numeric metrics: {filtered_metrics}")

        if numeric_metrics:
            self._queue.put({"metrics": numeric_metrics})

    def _process_metrics(self):
        """Background thread that processes metrics."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
                self._metrics_cache.append(item)
                # In a real implementation, this would sync to a server
            except queue.Empty:
                continue

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all cached metrics."""
        return self._metrics_cache.copy()


class Run:
    """Represents a Keys & Caches run session."""

    def __init__(self, config: RunConfig, system_info: SystemInfo, api_client: APIClient = None):
        self.config = config
        self.system_info = system_info
        self._api_client = api_client
        self._metrics_queue = MetricsQueue()
        self._summaries: Dict[str, Any] = {}
        self._artifacts: List[str] = []
        self._run_dir: Optional[Path] = None
        self._finished = False
        self._project_data: Optional[Dict[str, Any]] = None
        self._run_data: Optional[Dict[str, Any]] = None
        self._code_snapshot_uploaded = False

        # Set up the run
        self._setup_environment()
        self._setup_directories()
        self._save_metadata()

        # Create project and run on backend
        if self._api_client and self.config.mode != "disabled":
            self._create_backend_run()

        # Start background services
        if self.config.mode != "disabled":
            self._metrics_queue.start()

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def _setup_environment(self):
        """Set up environment variables for the run."""
        os.environ[KANDC_BACKEND_APP_NAME_ENV_KEY] = self.config.project
        os.environ[KANDC_JOB_ID_ENV_KEY] = self.config.job_id
        os.environ[KANDC_BACKEND_RUN_ENV_KEY] = "1"

        # Always set the trace base directory to avoid defaulting to /volume
        # Include the 'kandc' subdirectory in the base path so traces go to the right place
        base_dir = self.config.dir or Path.cwd()
        os.environ[KANDC_TRACE_BASE_DIR_ENV_KEY] = str(base_dir / "kandc")

    def _setup_directories(self):
        """Create run directories."""
        # Set up run directory
        base_dir = self.config.dir or Path.cwd()
        self._run_dir = base_dir / "kandc" / "simple-demo" / f"{self.config.name}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self._run_dir / "artifacts").mkdir(exist_ok=True)
        (self._run_dir / "logs").mkdir(exist_ok=True)
        (self._run_dir / "traces").mkdir(exist_ok=True)

    def _save_metadata(self):
        """Save run metadata to disk."""
        metadata = {
            "project": self.config.project,
            "name": self.config.name,
            "config": self.config.config,
            "tags": self.config.tags,
            "notes": self.config.notes,
            "mode": self.config.mode,
            "system_info": {
                "os": self.system_info.os,
                "os_version": self.system_info.os_version,
                "python_version": self.system_info.python_version,
                "hostname": self.system_info.hostname,
                "cpu_count": self.system_info.cpu_count,
            },
        }

        if self._run_dir:
            with open(self._run_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

    def _create_backend_run(self):
        """Create project and run on backend."""
        try:
            # Get or create project
            self._project_data = self._api_client.get_or_create_project(self.config.project)

            # Create run
            run_data = {
                "name": self.config.name,
                "config": self.config.config,
                "tags": self.config.tags,
                "notes": self.config.notes,
                "mode": self.config.mode,
            }
            self._run_data = self._api_client.create_run(self.config.project, run_data)

            # Save config.json with run ID
            if self._run_dir:
                config_data = {
                    "run_id": self._run_data["id"],
                    "project_id": self._project_data["id"],
                    **run_data,
                }
                with open(self._run_dir / "config.json", "w") as f:
                    json.dump(config_data, f, indent=2)

        except (APIError, AuthenticationError) as e:
            print(f"⚠️  Warning: Could not create backend run: {e}")
            if self.config.mode == "online":
                print("⚠️  Backend run not created; operating offline")
                self.config.mode = "offline"

    def _cleanup(self):
        """Clean up resources."""
        if not self._finished:
            self.finish()

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics for the run."""
        if self.config.mode != "disabled":
            self._metrics_queue.log(metrics, step)

    def log_artifact(self, name: str, data: Any, artifact_type: str = "file"):
        """Log an artifact for the run."""
        if not self._run_dir:
            return

        # Save artifact locally
        artifact_dir = self._run_dir / "artifacts"
        artifact_path = artifact_dir / name

        # Convert data to JSON if needed
        if isinstance(data, (dict, list)):
            artifact_path = artifact_path.with_suffix(".json")
            with open(artifact_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(artifact_path, "w") as f:
                f.write(str(data))

        self._artifacts.append(str(artifact_path))

        # Upload to backend if available
        if self._api_client and self._run_data and self.config.mode != "disabled":
            try:
                with open(artifact_path, "rb") as f:
                    self._api_client.upload_artifact(
                        self._run_data["id"], name, f.read(), artifact_type=artifact_type
                    )
            except (APIError, AuthenticationError) as e:
                print(f"⚠️  Warning: Could not upload artifact: {e}")

    def open_dashboard(self):
        """Open the run dashboard in a browser."""
        if self._api_client and self._run_data:
            try:
                url = self._api_client.get_run_url(self._run_data["id"])
                print(f"🌐 View results: {url}")
                webbrowser.open(url)
            except Exception:
                pass

    def _sync_metrics_to_backend(self):
        """Sync metrics to backend."""
        if not (self._api_client and self._run_data):
            return

        try:
            metrics = self._metrics_queue.get_metrics()
            if metrics:
                for metric_data in metrics:
                    # Fix: Use "metrics" not "data" to match the queue structure
                    metrics_dict = metric_data.get("metrics", {})
                    step = metric_data.get("step")

                    # Only send if we have actual metrics data
                    if metrics_dict:
                        self._api_client.log_metrics(self._run_data["id"], metrics_dict, step)
        except (APIError, AuthenticationError) as e:
            print(f"⚠️  Warning: Could not sync metrics: metrics: {metrics}, err: {e}")

    def upload_code_snapshot(self, archive_bytes: bytes) -> bool:
        """Upload code snapshot to backend."""
        if not self._api_client:
            return False

        try:
            # Use the run ID from backend
            run_id = self._run_data.get("id") if self._run_data else None
            if not run_id:
                return False

            response = self._api_client.upload_code_snapshot(run_id, archive_bytes)
            self._code_snapshot_uploaded = response.get("success", False)
            return self._code_snapshot_uploaded

        except Exception as e:
            print(f"Failed to upload code snapshot: {e}")
            return False

    def finish(self):
        """Finish the run and clean up."""
        if self._finished:
            return

        # Stop metrics queue
        if self.config.mode != "disabled":
            self._metrics_queue.stop()

            # Sync final metrics
            self._sync_metrics_to_backend()

        # Mark run as finished on backend
        if self._api_client and self._run_data:
            try:
                self._api_client.finish_run(self._run_data["id"])
            except (APIError, AuthenticationError) as e:
                print(f"⚠️  Warning: Could not finish run: {e}")

        self._finished = True
