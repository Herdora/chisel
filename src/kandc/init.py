"""
Keys & Caches initialization module.

This module provides the main entry point for initializing Keys & Caches tracking,
"""

import os
import sys
import json
import uuid
import time
import platform
import subprocess
import threading
import queue
import atexit
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
import warnings
import webbrowser
# import fnmatch  # No longer needed for source code capture

from .constants import (
    KANDC_BACKEND_APP_NAME_ENV_KEY,
    KANDC_BACKEND_RUN_ENV_KEY,
    KANDC_JOB_ID_ENV_KEY,
    KANDC_TRACE_BASE_DIR_ENV_KEY,
)
from .auth import get_auth_manager, AuthenticationError
from .api_client import APIClient, APIError


@dataclass
class SystemInfo:
    """System information for reproducibility."""

    os: str = field(default_factory=lambda: platform.system())
    os_version: str = field(default_factory=lambda: platform.version())
    python_version: str = field(default_factory=lambda: platform.python_version())
    python_executable: str = field(default_factory=lambda: sys.executable)
    hostname: str = field(default_factory=lambda: platform.node())
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 0)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False

    def __post_init__(self):
        """Collect git information if available."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                self.git_commit = result.stdout.strip()
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                self.git_branch = result.stdout.strip()
            result = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                self.git_dirty = bool(result.stdout.strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            pass


@dataclass
class RunConfig:
    """Configuration for a Keys & Caches run."""

    project: str
    name: str
    job_id: str
    config: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    dir: Optional[Path] = None
    mode: str = "online"  # online, offline, disabled
    start_time: datetime = field(default_factory=datetime.now)

    def flatten_config(self) -> Dict[str, Any]:
        """Flatten nested config dictionary for easier access."""

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

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Add metrics to the queue."""
        self._queue.put({"metrics": metrics, "step": step, "timestamp": time.time()})

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
        base_dir = self.config.dir or Path.cwd()
        self._run_dir = base_dir / "kandc" / self.config.project / self.config.job_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self._run_dir / "traces").mkdir(exist_ok=True)
        (self._run_dir / "logs").mkdir(exist_ok=True)
        (self._run_dir / "artifacts").mkdir(exist_ok=True)

    def _save_metadata(self):
        """Save run metadata to disk."""
        metadata = {
            "project": self.config.project,
            "name": self.config.name,
            "job_id": self.config.job_id,
            "config": self.config.config,
            "flattened_config": self.config.flatten_config(),
            "tags": self.config.tags,
            "notes": self.config.notes,
            "mode": self.config.mode,
            "start_time": self.config.start_time.isoformat(),
            "system_info": asdict(self.system_info),
        }

        metadata_path = self._run_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Also save config separately for easier access
        config_path = self._run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.config, f, indent=2)

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log metrics or other data."""
        if self.config.mode == "disabled":
            return

        # Update summaries with latest values
        self._summaries.update(data)

        # Add to metrics queue
        self._metrics_queue.log(data, step)

        # Sync to backend if available
        if self._api_client and self._run_data:
            try:
                self._api_client.log_metrics(self._run_data["id"], data, step)
            except (APIError, AuthenticationError):
                # Don't spam warnings, just continue
                pass

    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None):
        """Log an artifact file."""
        path = Path(path)
        if not path.exists():
            warnings.warn(f"Artifact path does not exist: {path}")
            return

        artifact_name = name or path.name
        self._artifacts.append(artifact_name)

        # Copy to artifacts directory
        if self._run_dir:
            artifact_dest = self._run_dir / "artifacts" / artifact_name
            if path.is_file():
                import shutil

                shutil.copy2(path, artifact_dest)

    def _create_backend_run(self):
        """Create project and run on the backend."""
        try:
            # Get or create project
            self._project_data = self._api_client.get_or_create_project(self.config.project)

            # Create run - pass project name, not ID
            run_data = {
                "name": self.config.name,
                "config": self.config.config,
                "tags": self.config.tags,
                "notes": self.config.notes,
            }
            self._run_data = self._api_client.create_run(self.config.project, run_data)

        except (APIError, AuthenticationError) as e:
            print(f"⚠️  Backend connection failed, continuing offline")

    def get_dashboard_url(self) -> Optional[str]:
        """Get the dashboard URL for this run."""
        if self._api_client and self._run_data:
            return self._api_client.get_dashboard_url(
                project_id=self._project_data["id"], run_id=self._run_data["id"]
            )
        return None

    def open_dashboard(self):
        """Open the dashboard for this run in browser."""
        url = self.get_dashboard_url()
        if url:
            try:
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

    def finish(self):
        """Finish the run and clean up."""
        if self._finished:
            return

        self._finished = True

        # Stop metrics queue and get final metrics
        self._metrics_queue.stop()

        # Sync final metrics to backend
        self._sync_metrics_to_backend()

        # Mark run as finished on backend
        if self._api_client and self._run_data:
            try:
                self._api_client.finish_run(self._run_data["id"])
            except (APIError, AuthenticationError):
                pass

        # Save summaries
        if self._run_dir:
            summaries_path = self._run_dir / "summaries.json"
            with open(summaries_path, "w") as f:
                json.dump(self._summaries, f, indent=2)

            # Save metrics history
            metrics_history = self._metrics_queue.get_metrics()
            if metrics_history:
                metrics_path = self._run_dir / "metrics.jsonl"
                with open(metrics_path, "w") as f:
                    for metric in metrics_history:
                        f.write(json.dumps(metric) + "\n")

        # Update metadata with finish time
        if self._run_dir:
            metadata_path = self._run_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                metadata["finish_time"] = datetime.now().isoformat()
                metadata["duration_seconds"] = (
                    datetime.now() - self.config.start_time
                ).total_seconds()
                metadata["status"] = "finished"

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

    def _cleanup(self):
        """Cleanup function called on exit."""
        if not self._finished:
            self.finish()

    @property
    def dir(self) -> Optional[Path]:
        """Get the run directory."""
        return self._run_dir

    @property
    def id(self) -> str:
        """Get the run ID."""
        return self.config.job_id

    @property
    def name(self) -> str:
        """Get the run name."""
        return self.config.name

    @property
    def project(self) -> str:
        """Get the project name."""
        return self.config.project


# Global run instance
_current_run: Optional[Run] = None
_browser_opened_this_session = False  # Track if browser was already opened


def _load_settings() -> Dict[str, Any]:
    """Load settings from config file and environment."""
    settings = {}

    # Check for config file
    config_paths = [
        Path.home() / ".config" / "kandc" / "settings.json",
        Path.home() / ".kandc" / "settings.json",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    settings.update(json.load(f))
                break
            except (json.JSONDecodeError, IOError):
                pass

    # Override with environment variables
    env_mapping = {
        "KANDC_PROJECT": "project",
        "KANDC_MODE": "mode",
        "KANDC_DIR": "dir",
        "KANDC_TAGS": "tags",
        "KANDC_NOTES": "notes",
    }

    for env_key, setting_key in env_mapping.items():
        if env_value := os.environ.get(env_key):
            if setting_key == "tags":
                settings[setting_key] = env_value.split(",")
            else:
                settings[setting_key] = env_value

    return settings


# def capture_project_source_code(
#     project_root: Optional[Union[str, Path]] = None,
#     exclude_patterns: Optional[List[str]] = None,
#     max_file_size: int = 1024 * 1024,  # 1MB default max file size
# ) -> List[Dict[str, Any]]:
# """
# Capture all Python source code in the project directory.
#
# Args:
#     project_root: Root directory to scan (defaults to current working directory)
#     exclude_patterns: List of patterns to exclude (supports wildcards)
#     max_file_size: Maximum file size to capture in bytes
#
# Returns:
#     List of dictionaries containing source file information
# """
# if project_root is None:
#     project_root = os.getcwd()
#
# project_root = Path(project_root).resolve()
#
# # Default exclude patterns
# default_exclude = [
#     ".git",
#     "__pycache__",
#     ".venv",
#     "venv",
#     ".env",
#     "env",
#     "node_modules",
#     "dist",
#     "build",
#     ".pytest_cache",
#     ".ruff_cache",
#     "*.pyc",
#     "*.pyo",
#     "*.pyd",
#     ".DS_Store",
#     "*.egg-info",
#     ".mypy_cache",
#     ".tox",
#     "htmlcov",
#     ".coverage",
# ]
#
# exclude_patterns = exclude_patterns or []
# exclude_patterns.extend(default_exclude)
#
# source_files = []
#
# def should_exclude(path: Path) -> bool:
#     """Check if a path should be excluded."""
#     path_str = str(path)
#     for pattern in exclude_patterns:
#         if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(path_str, pattern):
#                 return True
#         return False
#
# # Walk through the project directory
# for root, dirs, files in os.walk(project_root):
#     root_path = Path(root)
#
#     # Filter out excluded directories
#     dirs[:] = [d for d in dirs if not should_exclude(root_path / d)]
#
#     for file in files:
#         # Only capture Python files
#         if not file.endswith(".py"):
#             continue
#
#         file_path = root_path / file
#
#         # Skip excluded files
#         if should_exclude(file_path):
#             continue
#
#         # Skip files that are too large
#         try:
#             file_size = file_path.stat().st_size
#             file_size > max_file_size:
#                 continue
#
#             # Read file content
#             with open(file_path, "r", encoding="utf-8") as f:
#                 content = f.read()
#
#             # Calculate relative path
#             try:
#                 relative_path = file_path.relative_to(project_root)
#             except ValueError:
#                 relative_path = file_path
#
#             source_files.append(
#                 {
#                     "file_path": str(file_path),
#                     "file_name": file,
#                     "file_size": file_size,
#                     "last_modified": file_path.stat().st_mtime,
#                     "language": "python",
#                 }
#             )
#
#         except Exception:
#             # Skip files that can't be read
#             continue
#
# return source_files


def init(
    project: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    dir: Optional[Union[str, Path]] = None,
    mode: Optional[str] = None,
    reinit: bool = False,
    open_browser: bool = True,
    # capture_source: bool = True,  # Source code capture feature removed
    # source_exclude_patterns: Optional[List[str]] = None,  # Source code capture feature removed
    **kwargs,
) -> Run:
    """
    Initialize Keys & Caches tracking.

    Args:
        project: Project name for grouping runs (default: from settings or "default-project")
        name: Human-readable name for this run (default: timestamp-based)
        config: Configuration dictionary to track hyperparameters
        tags: List of tags for categorizing the run
        notes: Notes or description for the run
        dir: Directory for storing run data (default: current directory)
        mode: Run mode - "online", "offline", or "disabled"
        reinit: Force reinitialization even if a run is already active
        open_browser: Whether to automatically open dashboard in browser (default: True)
        **kwargs: Additional configuration passed to config dict

    Returns:
        Run: The initialized run object

    Example:
        import kandc

        # Basic initialization
        run = kandc.init()

        # With project and config
        run = kandc.init(
            project="my-ml-project",
            name="experiment-1",
            config={
                "learning_rate": 0.001,
                "batch_size": 32,
                "model": "resnet18",
            },
            tags=["baseline", "resnet"],
            notes="Testing new learning rate schedule"
        )

        # Log metrics
        run.log({"loss": 0.5, "accuracy": 0.92})

        # Finish the run
        kandc.finish()
    """
    global _current_run

    # Check if we already have an active run
    if _current_run is not None and not reinit:
        warnings.warn(
            "Keys & Caches run already initialized. Use reinit=True to force reinitialization."
        )
        return _current_run

    # Load settings from config file and environment
    settings = _load_settings()

    # Merge settings with explicit arguments (explicit args take precedence)
    project = project or settings.get("project", "default-project")
    name = name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tags = tags or settings.get("tags", [])
    notes = notes or settings.get("notes")
    dir = Path(dir) if dir else (Path(settings.get("dir", ".")) if "dir" in settings else None)
    mode = mode or settings.get("mode", "online")

    # Merge kwargs into config
    if config is None:
        config = {}
    config.update(kwargs)

    # Generate unique job ID
    job_id = f"{name}-{str(uuid.uuid4())[:8]}"

    # Authenticate with backend if in online mode
    api_client = None
    if mode == "online":
        try:
            auth_manager = get_auth_manager()
            api_client = auth_manager.ensure_authenticated()
        except (AuthenticationError, Exception):
            mode = "offline"

    # Collect system information
    system_info = SystemInfo()

    # Create run configuration
    run_config = RunConfig(
        project=project,
        name=name,
        job_id=job_id,
        config=config,
        tags=tags,
        notes=notes,
        dir=dir,
        mode=mode,
    )

    # Create new run
    _current_run = Run(run_config, system_info, api_client)

    # Source code capture disabled
    # if capture_source and mode != "disabled":
    #     # Source code capture feature removed
    #     pass

    # Print minimal initialization message
    print(f"🚀 kandc initialized: {_current_run.name}")

    # Open dashboard if connected to backend
    if mode == "online" and api_client and _current_run._run_data:
        dashboard_url = _current_run.get_dashboard_url()
        if dashboard_url:
            print(f"🌐 Dashboard: {dashboard_url}")

            # Only auto-open dashboard if requested and not already opened
            global _browser_opened_this_session
            if open_browser and not _browser_opened_this_session:
                _current_run.open_dashboard()
                _browser_opened_this_session = True

    return _current_run


def finish():
    """
    Finish the current Keys & Caches run.

    This performs the following cleanup:
    - Flushes any pending logs
    - Saves final summaries
    - Uploads artifacts
    - Marks the run as complete
    - Cleans up resources
    """
    global _current_run, _browser_opened_this_session

    if _current_run is None:
        warnings.warn("No active Keys & Caches run to finish.")
        return

    # Finish the run
    _current_run.finish()

    # Show dashboard URL if available
    if _current_run.config.mode == "online" and _current_run._run_data:
        if dashboard_url := _current_run.get_dashboard_url():
            print(f"🌐 View results: {dashboard_url}")

    # Clear environment variables
    for key in [
        KANDC_BACKEND_RUN_ENV_KEY,
        KANDC_BACKEND_APP_NAME_ENV_KEY,
        KANDC_JOB_ID_ENV_KEY,
        KANDC_TRACE_BASE_DIR_ENV_KEY,
    ]:
        os.environ.pop(key, None)

    # Clear config environment variables
    for key in list(os.environ.keys()):
        if key.startswith("KANDC_CONFIG_"):
            os.environ.pop(key)

    print(f"✅ Run completed: {_current_run.name}")

    # Clear the global run and reset browser flag for next session
    _current_run = None
    _browser_opened_this_session = False


def log(data: Dict[str, Any], step: Optional[int] = None):
    """
    Log metrics to the current run.

    Args:
        data: Dictionary of metric names to values
        step: Optional step counter for the metrics

    Example:
        kandc.log({"loss": 0.5, "accuracy": 0.92})
        kandc.log({"loss": 0.4, "accuracy": 0.94}, step=100)
    """
    if _current_run is None:
        warnings.warn("No active Keys & Caches run. Call kandc.init() first.")
        return

    _current_run.log(data, step)


def get_current_run() -> Optional[Run]:
    """Get the current active run, if any."""
    return _current_run


def is_initialized() -> bool:
    """Check if Keys & Caches is initialized."""
    return _current_run is not None or os.environ.get(KANDC_BACKEND_RUN_ENV_KEY) == "1"
