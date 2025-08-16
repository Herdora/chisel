"""
API client for communicating with Keys & Caches backend.
"""

import requests
from typing import Dict, Any, Optional
import webbrowser
import time
from urllib.parse import urljoin

from .constants import KANDC_BACKEND_URL, KANDC_FRONTEND_URL


class APIError(Exception):
    """Exception raised when API calls fail."""

    pass


class AuthenticationError(APIError):
    """Exception raised when authentication fails."""

    pass


class APIClient:
    """Client for communicating with Keys & Caches backend."""

    def __init__(self, base_url: str = KANDC_BACKEND_URL, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update(
                {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            )

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the API."""
        url = urljoin(f"{self.base_url}/", endpoint.lstrip("/"))

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {e}")
            elif response.status_code == 403:
                raise AuthenticationError(f"Access forbidden: {e}")
            else:
                raise APIError(f"HTTP {response.status_code}: {e}")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {e}")

    def authenticate_with_browser(self) -> str:
        """
        Authenticate user via browser and return API key.

        Returns:
            API key string

        Raises:
            AuthenticationError: If authentication fails
        """
        # Create auth session on backend and get session ID
        try:
            auth_response = self._request("GET", "/api/v1/auth/init")
            session_id = auth_response.get("session_id")
            auth_url = auth_response.get("auth_url")

            if not session_id or not auth_url:
                raise AuthenticationError("Invalid response from auth init endpoint")

            print("✅ Created authentication session")
        except APIError as e:
            raise AuthenticationError(f"Failed to create auth session: {e}")

        # Open browser to auth URL
        print("🌐 Opening browser for authentication...")
        print(f"   URL: {auth_url}")

        try:
            webbrowser.open(auth_url)
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"   Please manually open: {auth_url}")

        # Poll for authentication completion
        print("⏳ Waiting for authentication...")
        max_attempts = 60  # 5 minutes with 5-second intervals

        for attempt in range(max_attempts):
            remaining = max_attempts - attempt
            print(f"   ⏰ {remaining} attempts remaining ({remaining * 5}s)...")

            try:
                # Check auth status
                auth_status = self._request("GET", f"/api/v1/auth/check/{session_id}")

                if auth_status.get("authenticated"):
                    api_key = auth_status.get("api_key")
                    if api_key:
                        print("🎉 Authentication successful!")
                        print(f"   Email: {auth_status.get('email', 'Unknown')}")
                        return api_key
                    else:
                        raise AuthenticationError(
                            "Authentication succeeded but no API key returned"
                        )

                # Wait before next check
                time.sleep(5)

            except APIError:
                # Continue polling on API errors
                time.sleep(5)
                continue

        raise AuthenticationError("Authentication timed out. Please try again.")

    def create_project(self, name: str, description: str = None) -> Dict[str, Any]:
        """Create a new project."""
        data = {"name": name, "description": description or f"Project {name}"}
        return self._request("POST", "/api/v1/projects", json=data)

    def get_or_create_project(self, name: str) -> Dict[str, Any]:
        """Get existing project or create new one."""
        try:
            # Try to get existing project
            projects = self._request("GET", "/api/v1/projects")
            for project in projects:
                if project["name"] == name:
                    return project
        except APIError:
            pass

        # Create new project if not found
        return self.create_project(name)

    def create_run(self, project_name: str, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new run within a project."""
        data = {
            "project_name": project_name,  # Backend expects project_name, not project_id
            "name": run_data.get("name", "unnamed-run"),
            "config": run_data.get("config", {}),
            "tags": run_data.get("tags", []),
            "notes": run_data.get("notes"),
            "mode": "online",  # Set mode to online
        }
        return self._request("POST", "/api/v1/runs", json=data)

    def update_run(self, run_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update run data."""
        return self._request("PUT", f"/api/v1/runs/{run_id}", json=data)

    def log_metrics(
        self, run_id: str, metrics: Dict[str, Any], step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Log metrics for a run."""
        # Backend expects metrics as query params, not nested in JSON
        params = {}
        if step is not None:
            params["step"] = step
        return self._request("POST", f"/api/v1/runs/{run_id}/metrics", json=metrics, params=params)

    def finish_run(self, run_id: str) -> Dict[str, Any]:
        """Mark run as finished."""
        return self.update_run(run_id, {"status": "completed"})

    def create_artifact(self, run_id: str, artifact_data: dict, file_path: str) -> dict:
        """Upload an artifact file to the backend.

        Note: Currently just registers the artifact metadata.
        File upload to storage would be implemented separately.
        """
        try:
            # Read file to get actual size
            import os

            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

            # Prepare the artifact data for the backend
            artifact_metadata = {
                "name": artifact_data.get("name", "unnamed"),
                "artifact_type": artifact_data.get("artifact_type", "file"),
                "file_size": file_size,
                "storage_path": file_path,  # Required field - using local path for now
                "original_path": file_path,  # Store original local path
                "metadata": artifact_data.get("metadata", {}),
            }

            # Send artifact metadata to backend
            return self._request("POST", f"/api/v1/runs/{run_id}/artifacts", json=artifact_metadata)

        except Exception as e:
            print(f"⚠️  Failed to register artifact {artifact_data.get('name', 'unknown')}: {e}")
            return {}

    # Source code artifact method removed
    # def create_source_code_artifact(...):
    #     # Method removed
    #     pass

    def get_dashboard_url(self, project_id: str = None, run_id: str = None) -> str:
        """Get dashboard URL for project or run."""
        if run_id:
            return f"{KANDC_FRONTEND_URL}/runs/{run_id}"
        elif project_id:
            return f"{KANDC_FRONTEND_URL}/projects/{project_id}"
        else:
            return f"{KANDC_FRONTEND_URL}/dashboard"
