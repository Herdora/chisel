"""
Cached files API client for Keys & Caches.
Handles communication with the backend for large file caching.
"""

import os
import hashlib
import tempfile
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import requests

from .constants import KANDC_BACKEND_URL, KANDC_BACKEND_URL_ENV_KEY

# Rich imports for progress tracking
try:
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeRemainingColumn,
        FileSizeColumn,
        TransferSpeedColumn,
    )
    from rich.console import Console

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Minimum file size for caching (1GB) - DISABLED
MIN_CACHE_FILE_SIZE = 1 * 1024 * 1024 * 1024

# Maximum file size for upload (5GB)
MAX_UPLOAD_FILE_SIZE = 5 * 1024 * 1024 * 1024


class ProgressTrackingFile:
    """File-like object that tracks read progress during upload."""

    def __init__(self, file_path, progress, task):
        self.file_path = file_path
        self.progress = progress
        self.task = task
        self.file_obj = None
        self.uploaded = 0
        self.total_size = file_path.stat().st_size

    def open(self):
        """Open the file for reading."""
        self.file_obj = open(self.file_path, "rb")
        return self

    def read(self, size=-1):
        """Read data and track progress."""
        if not self.file_obj:
            return b""

        chunk = self.file_obj.read(size)
        if chunk:
            self.uploaded += len(chunk)
            self.progress.update(self.task, completed=self.uploaded)
        return chunk

    def close(self):
        """Close the file."""
        if self.file_obj:
            self.file_obj.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def name(self):
        return self.file_path.name


def upload_with_progress(url, headers, file_path, progress, task):
    """Upload file with REAL progress tracking - NO memory bloat."""

    # Use progress tracking file object
    with ProgressTrackingFile(file_path, progress, task) as tracker:
        files = {"file": (file_path.name, tracker, "application/octet-stream")}
        response = requests.post(url, headers=headers, files=files)

    return response


class CachedFilesClient:
    """Client for interacting with the cached files API."""

    def __init__(self, api_key: str):
        """Initialize the cached files client."""
        self.api_key = api_key
        self.backend_url = os.environ.get(KANDC_BACKEND_URL_ENV_KEY) or KANDC_BACKEND_URL
        self.base_url = f"{self.backend_url}/api/v1/cached-files"

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def check_file_hash(self, file_hash: str) -> Dict[str, Any]:
        """
        Check if a file with the given hash exists in the cache.

        Returns:
            dict: Response with 'exists' boolean and optional 'file_info'
        """
        try:
            response = requests.post(
                f"{self.base_url}/check-hash",
                headers=self._get_headers(),
                data={"file_hash": file_hash},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error checking file hash: {e}")
            return {"exists": False}

    def upload_cached_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Upload a file to the cache with progress tracking.

        Returns:
            dict: Response with upload result or None if failed
        """
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        try:
            if RICH_AVAILABLE and file_size > 100 * 1024 * 1024:  # Show progress for files >100MB
                console = Console()

                with Progress(
                    TextColumn("[bold blue]Caching", justify="right"),
                    BarColumn(bar_width=40),
                    "[progress.percentage]{task.percentage:>3.1f}%",
                    "•",
                    FileSizeColumn(),
                    "•",
                    TransferSpeedColumn(),
                    "•",
                    TimeRemainingColumn(),
                    console=console,
                    transient=False,  # Keep progress visible
                ) as progress:
                    task = progress.add_task(f"[cyan]{file_path.name}[/cyan]", total=file_size)

                    # Use our custom upload function with realistic progress
                    response = upload_with_progress(
                        f"{self.base_url}/upload", self._get_headers(), file_path, progress, task
                    )

                console.print(
                    f"✅ [green]Successfully cached {file_path.name} ({file_size_mb:.1f}MB)[/green]"
                )
            else:
                # Fallback for smaller files or when Rich is not available
                print(f"📤 Caching {file_path.name} ({file_size_mb:.1f}MB)...")
                with open(file_path, "rb") as f:
                    files = {"file": (file_path.name, f, "application/octet-stream")}
                    response = requests.post(
                        f"{self.base_url}/upload", headers=self._get_headers(), files=files
                    )
                print(f"✅ Successfully cached {file_path.name}")

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            if RICH_AVAILABLE:
                console = Console()
                console.print(f"❌ [red]Error caching file: {e}[/red]")
                # Print more detailed error info
                if hasattr(e, "response") and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        console.print(f"🔍 [yellow]Server response: {error_detail}[/yellow]")
                    except Exception:
                        console.print(f"🔍 [yellow]Server response: {e.response.text}[/yellow]")
            else:
                print(f"❌ Error caching file: {e}")
                # Print more detailed error info
                if hasattr(e, "response") and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        print(f"🔍 Server response: {error_detail}")
                    except Exception:
                        print(f"🔍 Server response: {e.response.text}")
            return None

    def increment_file_reference(self, file_id: str) -> bool:
        """
        Increment the reference count for a cached file.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/{file_id}/increment-reference", headers=self._get_headers()
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"Error incrementing file reference: {e}")
            return False

    def decrement_file_reference(self, file_id: str) -> bool:
        """
        Decrement the reference count for a cached file.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/{file_id}/decrement-reference", headers=self._get_headers()
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"Error decrementing file reference: {e}")
            return False


def scan_directory_for_large_files(
    directory: Path, min_size: int = MIN_CACHE_FILE_SIZE
) -> List[Path]:
    """
    Scan a directory for all files (caching check disabled).

    NOTE: Caching check is disabled - this function now returns all files
    regardless of size to allow uploading folders of any size.

    Args:
        directory: Directory to scan
        min_size: Minimum file size in bytes (ignored, kept for compatibility)

    Returns:
        List of all file paths in the directory
    """
    all_files = []
    print(f"Scanning directory: {directory}")
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            try:
                # Caching check disabled - include all files
                all_files.append(file_path)
            except (OSError, IOError):
                print(f"Error accessing file: {file_path}")
                # Skip files we can't access
                continue

    return all_files


def create_cached_file_placeholder(
    original_path: Path, cached_file_info: Dict[str, Any], relative_to: Path
) -> str:
    """
    Create a placeholder file content for a cached file.

    Args:
        original_path: Original file path
        cached_file_info: Information about the cached file from the API
        relative_to: Path to make the original_path relative to

    Returns:
        JSON string content for the placeholder
    """
    try:
        relative_path = original_path.relative_to(relative_to)
    except ValueError:
        relative_path = original_path

    placeholder_data = {
        "type": "keysandcaches_cached_file_link",
        "version": "1.0",
        "original_path": str(relative_path),
        "cached_file": {
            "id": cached_file_info["file_info"]["id"],
            "hash": cached_file_info["file_info"]["file_hash"],
            "filename": cached_file_info["file_info"]["original_filename"],
            "size": cached_file_info["file_info"]["file_size"],
        },
        "metadata": {
            "created_by": "kandc",
            "cache_date": cached_file_info["file_info"]["upload_date"],
        },
    }

    return json.dumps(placeholder_data, indent=2)


def is_cached_file_placeholder(file_path: Path) -> bool:
    """
    Check if a file is a cached file placeholder.

    Args:
        file_path: Path to the file to check

    Returns:
        True if it's a placeholder, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            data = json.loads(content)
            return data.get("type") == "keysandcaches_cached_file_link"
    except (IOError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def parse_cached_file_placeholder(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse a cached file placeholder.

    Args:
        file_path: Path to the placeholder file

    Returns:
        Placeholder data if valid, None otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            data = json.loads(content)
            if data.get("type") == "keysandcaches_cached_file_link":
                return data
            return None
    except (IOError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def process_directory_for_cached_files(
    directory: Path, api_key: str, temp_dir: Optional[Path] = None
) -> Tuple[Path, List[Dict[str, Any]]]:
    """
    Process a directory - CACHING DISABLED.

    NOTE: Caching is disabled. This function now just returns the original
    directory without any caching processing to allow uploading folders of any size.

    Args:
        directory: Directory to process
        api_key: API key for authentication (ignored)
        temp_dir: Temporary directory (ignored)

    Returns:
        Tuple of (original_directory_path, empty_list)
    """
    print(f"Processing directory: {directory} (caching disabled)")

    # Return the original directory without any processing
    # No caching, no file replacement, no placeholders
    return directory, []
