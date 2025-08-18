"""
API client for communicating with Keys & Caches backend.
"""

import os
import time
import webbrowser
from typing import Dict, Any, Optional, List, Union, TypedDict, Literal, cast
import json
from urllib.parse import urljoin

import requests
from requests import Session, Response

from ..constants import KANDC_BACKEND_URL, KANDC_FRONTEND_URL


class APIError(Exception):
    """Exception raised when API calls fail."""

    pass


class AuthenticationError(APIError):
    """Exception raised when authentication fails."""

    pass


# Type definitions for API responses
class ProjectResponse(TypedDict, total=False):
    """Type definition for project response."""

    id: str
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str
    user_id: str
    tags: List[str]
    metadata: Dict[str, Any]


class RunResponse(TypedDict, total=False):
    """Type definition for run response."""

    id: str
    project_id: str
    project_name: str
    name: str
    status: Literal["running", "completed", "failed", "crashed"]
    config: Dict[str, Any]
    tags: List[str]
    notes: Optional[str]
    mode: Literal["online", "offline"]
    start_time: str
    end_time: Optional[str]
    duration_seconds: Optional[float]
    created_at: str
    updated_at: str
    user_id: str
    metrics_count: int
    artifacts_count: int


class MetricResponse(TypedDict, total=False):
    """Type definition for metric response."""

    id: str
    run_id: str
    metric_name: str
    value: Union[float, int]
    step: Optional[int]
    timestamp: str
    created_at: str


class ArtifactResponse(TypedDict, total=False):
    """Type definition for artifact response."""

    id: str
    run_id: str
    name: str
    artifact_type: Literal[
        "file",
        "model",
        "dataset",
        "image",
        "other",
        "trace",
        "timing",
        "profile",
        "logs",
        "code_snapshot",
        "source_code",
    ]
    file_size: int
    storage_path: str
    original_path: Optional[str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class AuthInitResponse(TypedDict):
    """Type definition for auth init response."""

    session_id: str
    auth_url: str


class AuthStatusResponse(TypedDict, total=False):
    """Type definition for auth status response."""

    authenticated: bool
    api_key: Optional[str]
    email: Optional[str]
    user_id: Optional[str]


class CodeSnapshotResponse(TypedDict):
    """Type definition for code snapshot upload response."""

    success: bool
    message: str
    artifact_id: Optional[str]
    s3_path: Optional[str]


class APIClient:
    """Client for communicating with Keys & Caches backend."""

    def __init__(self, base_url: str = KANDC_BACKEND_URL, api_key: Optional[str] = None) -> None:
        """
        Initialize API client.

        Args:
            base_url: Base URL for the API server
            api_key: Optional API key for authentication
        """
        self.base_url: str = base_url.rstrip("/")
        self.api_key: Optional[str] = api_key
        self.session: Session = requests.Session()

        if self.api_key:
            self.session.headers.update(
                {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            )

    def _request(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments to pass to requests

        Returns:
            JSON response from the API

        Raises:
            AuthenticationError: If authentication fails
            APIError: If the request fails
        """
        url: str = urljoin(f"{self.base_url}/", endpoint.lstrip("/"))

        response: Optional[Response] = None
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response is not None:
                error_detail = ""
                try:
                    error_json = response.json()
                    if isinstance(error_json, dict):
                        error_detail = error_json.get("detail", error_json.get("message", ""))
                except:
                    error_detail = response.text

                if response.status_code == 401:
                    raise AuthenticationError(
                        f"Authentication failed (401): {error_detail or 'Invalid or expired credentials'}"
                    )
                elif response.status_code == 403:
                    raise AuthenticationError(
                        f"Access forbidden (403): {error_detail or 'Insufficient permissions'}"
                    )
                else:
                    raise APIError(
                        f"HTTP {response.status_code}: {error_detail or str(e)}\n"
                        f"URL: {url}\n"
                        f"Method: {method}"
                    )
            else:
                raise APIError(f"Request failed: {e}")
        except requests.exceptions.ConnectionError as e:
            raise APIError(
                f"Connection failed: Could not connect to {self.base_url}\n"
                f"Please check:\n"
                f"1. Your internet connection\n"
                f"2. The backend server is running\n"
                f"3. The KANDC_BACKEND_URL environment variable is correct\n"
                f"Error: {str(e)}"
            )
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed:\nURL: {url}\nMethod: {method}\nError: {str(e)}")

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
            auth_response = cast(AuthInitResponse, self._request("GET", "/api/v1/auth/init"))
            session_id: Optional[str] = auth_response.get("session_id")
            auth_url: Optional[str] = auth_response.get("auth_url")

            if not session_id or not auth_url:
                raise AuthenticationError("Invalid response from auth init endpoint")

            print("✅ Created authentication session")
        except APIError as e:
            raise AuthenticationError(f"Failed to create auth session: {e}")

        # Open browser to auth URL
        print("🌐 Opening browser for authentication...")
        print(f"   URL: {auth_url}")

        # Check if we're in development mode
        from ..constants import DEV_MODE

        if DEV_MODE:
            print("   🔧 Running in development mode")
        else:
            print("   🌍 Running in production mode")

        try:
            webbrowser.open(auth_url)
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"   Please manually open: {auth_url}")

        # Poll for authentication completion
        print("⏳ Waiting for authentication...")
        max_attempts: int = 60  # 5 minutes with 5-second intervals

        for attempt in range(max_attempts):
            remaining: int = max_attempts - attempt
            # Only show progress every 10 attempts to reduce noise
            if attempt % 10 == 0:
                print(f"   ⏰ {remaining} attempts remaining ({remaining * 5}s)...")

            try:
                # Check auth status
                auth_status = cast(
                    AuthStatusResponse, self._request("GET", f"/api/v1/auth/check/{session_id}")
                )

                if auth_status.get("authenticated"):
                    api_key: Optional[str] = auth_status.get("api_key")
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

    def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProjectResponse:
        """
        Create a new project.

        Args:
            name: Project name
            description: Optional project description
            tags: Optional list of tags
            metadata: Optional metadata dictionary

        Returns:
            Project response dictionary
        """
        data: Dict[str, Any] = {"name": name, "description": description or f"Project {name}"}
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata

        return cast(ProjectResponse, self._request("POST", "/api/v1/projects", json=data))

    def get_or_create_project(self, name: str) -> ProjectResponse:
        """
        Get existing project or create new one.

        Args:
            name: Project name

        Returns:
            Project response dictionary
        """
        try:
            # Try to get existing project
            projects = cast(List[ProjectResponse], self._request("GET", "/api/v1/projects"))
            for project in projects:
                if project["name"] == name:
                    return project
        except APIError:
            pass

        # Create new project if not found
        return self.create_project(name)

    def create_run(self, project_name: str, run_data: Dict[str, Any]) -> RunResponse:
        """
        Create new run within a project.

        Args:
            project_name: Name of the project
            run_data: Dictionary containing run configuration

        Returns:
            Run response dictionary
        """
        data: Dict[str, Any] = {
            "project_name": project_name,  # Backend expects project_name, not project_id
            "name": run_data.get("name", "unnamed-run"),
            "config": run_data.get("config", {}),
            "tags": run_data.get("tags", []),
            "notes": run_data.get("notes"),
            "mode": run_data.get("mode", "online"),  # Default to online mode
        }
        return cast(RunResponse, self._request("POST", "/api/v1/runs", json=data))

    def update_run(self, run_id: str, data: Dict[str, Any]) -> RunResponse:
        """
        Update run data.

        Args:
            run_id: ID of the run to update
            data: Dictionary containing fields to update

        Returns:
            Updated run response dictionary
        """
        return cast(RunResponse, self._request("PUT", f"/api/v1/runs/{run_id}", json=data))

    def log_metrics(
        self, run_id: str, metrics: Dict[str, Union[float, int]], step: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Log metrics for a run.

        Args:
            run_id: ID of the run
            metrics: Dictionary of metric names to values
            step: Optional step number

        Returns:
            Response dictionary
        """
        params: Dict[str, int] = {}
        if step is not None:
            params["step"] = step
        result = self._request(
            "POST", f"/api/v1/runs/{run_id}/metrics", json=metrics, params=params
        )
        return cast(Dict[str, Any], result)

    def finish_run(self, run_id: str) -> RunResponse:
        """
        Mark run as finished.

        Args:
            run_id: ID of the run to finish

        Returns:
            Updated run response dictionary
        """
        return self.update_run(run_id, {"status": "completed"})

    def create_artifact(
        self, run_id: str, artifact_data: Dict[str, Any], file_path: str
    ) -> ArtifactResponse:
        """
        Upload an artifact file (multipart) so the backend stores it in S3.

        Args:
            run_id: ID of the run
            artifact_data: Dict with optional keys: name, artifact_type, metadata
            file_path: Local path to the artifact file to upload

        Returns:
            Artifact response dictionary
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Artifact file not found: {file_path}")

            import mimetypes

            filename = os.path.basename(file_path)
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"

            # Build multipart form
            files: Dict[str, tuple] = {
                "file": (filename, open(file_path, "rb"), mime),
            }

            data: Dict[str, Any] = {
                "name": artifact_data.get("name", filename),
                "artifact_type": artifact_data.get("artifact_type", "file"),
                "metadata": json.dumps(artifact_data.get("metadata", {})),
            }

            # Temporarily remove Content-Type header for multipart upload
            original_headers: Dict[str, Any] = dict(self.session.headers)
            if "Content-Type" in self.session.headers:
                del self.session.headers["Content-Type"]

            try:
                resp: Response = self.session.post(
                    f"{self.base_url}/api/v1/runs/{run_id}/artifacts", files=files, data=data
                )
            finally:
                # close file handle(s)
                try:
                    file_obj = files["file"][1]
                    if hasattr(file_obj, "close"):
                        file_obj.close()
                except Exception:
                    pass
                # restore headers
                self.session.headers.clear()
                self.session.headers.update(original_headers)

            if not resp.ok:
                raise APIError(f"Upload failed: {resp.status_code} {resp.text}")

            return cast(ArtifactResponse, resp.json())

        except Exception as e:
            print(f"⚠️  Failed to upload artifact {artifact_data.get('name', 'unknown')}: {e}")
            return cast(ArtifactResponse, {})

    def upload_code_snapshot(self, run_id: str, archive_bytes: bytes) -> CodeSnapshotResponse:
        """
        Upload a code snapshot archive.

        Args:
            run_id: ID of the run
            archive_bytes: Compressed archive bytes

        Returns:
            Code snapshot response dictionary
        """
        files: Dict[str, tuple] = {"file": ("snapshot.tar.gz", archive_bytes, "application/gzip")}

        # Temporarily remove Content-Type header for multipart upload
        original_headers: Dict[str, Any] = dict(self.session.headers)
        if "Content-Type" in self.session.headers:
            del self.session.headers["Content-Type"]

        try:
            response: Response = self.session.post(
                f"{self.base_url}/api/v1/code-snapshots/upload/{run_id}", files=files
            )
            response.raise_for_status()
            return response.json()
        finally:
            # Restore original headers
            self.session.headers.update(original_headers)

    def get_dashboard_url(
        self, project_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> str:
        """
        Get dashboard URL for project or run.

        Args:
            project_id: Optional project ID
            run_id: Optional run ID

        Returns:
            Dashboard URL string
        """
        if run_id:
            return f"{KANDC_FRONTEND_URL}/runs/{run_id}"
        elif project_id:
            return f"{KANDC_FRONTEND_URL}/projects/{project_id}"
        else:
            return f"{KANDC_FRONTEND_URL}/dashboard"

    def list_projects(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[ProjectResponse]:
        """
        List all projects for the authenticated user.

        Args:
            limit: Optional limit on number of results
            offset: Optional offset for pagination

        Returns:
            List of project response dictionaries
        """
        params: Dict[str, int] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        return cast(List[ProjectResponse], self._request("GET", "/api/v1/projects", params=params))

    def get_project(self, project_id: str) -> ProjectResponse:
        """
        Get a specific project by ID.

        Args:
            project_id: ID of the project

        Returns:
            Project response dictionary
        """
        return cast(ProjectResponse, self._request("GET", f"/api/v1/projects/{project_id}"))

    def list_runs(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[RunResponse]:
        """
        List runs, optionally filtered by project or status.

        Args:
            project_id: Optional project ID to filter by
            status: Optional status to filter by
            limit: Optional limit on number of results
            offset: Optional offset for pagination

        Returns:
            List of run response dictionaries
        """
        params: Dict[str, Union[str, int]] = {}
        if project_id:
            params["project_id"] = project_id
        if status:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        return cast(List[RunResponse], self._request("GET", "/api/v1/runs", params=params))

    def get_run(self, run_id: str) -> RunResponse:
        """
        Get a specific run by ID.

        Args:
            run_id: ID of the run

        Returns:
            Run response dictionary
        """
        return cast(RunResponse, self._request("GET", f"/api/v1/runs/{run_id}"))

    def get_metrics(
        self,
        run_id: str,
        metric_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[MetricResponse]:
        """
        Get metrics for a run.

        Args:
            run_id: ID of the run
            metric_name: Optional metric name to filter by
            limit: Optional limit on number of results
            offset: Optional offset for pagination

        Returns:
            List of metric response dictionaries
        """
        params: Dict[str, Union[str, int]] = {}
        if metric_name:
            params["metric_name"] = metric_name
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        return cast(
            List[MetricResponse],
            self._request("GET", f"/api/v1/runs/{run_id}/metrics", params=params),
        )

    def list_artifacts(
        self,
        run_id: str,
        artifact_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ArtifactResponse]:
        """
        List artifacts for a run.

        Args:
            run_id: ID of the run
            artifact_type: Optional artifact type to filter by
            limit: Optional limit on number of results
            offset: Optional offset for pagination

        Returns:
            List of artifact response dictionaries
        """
        params: Dict[str, Union[str, int]] = {}
        if artifact_type:
            params["artifact_type"] = artifact_type
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        return cast(
            List[ArtifactResponse],
            self._request("GET", f"/api/v1/runs/{run_id}/artifacts", params=params),
        )

    def delete_project(self, project_id: str) -> Dict[str, Any]:
        """
        Delete a project.

        Args:
            project_id: ID of the project to delete

        Returns:
            Response dictionary
        """
        result = self._request("DELETE", f"/api/v1/projects/{project_id}")
        return cast(Dict[str, Any], result)

    def delete_run(self, run_id: str) -> Dict[str, Any]:
        """
        Delete a run.

        Args:
            run_id: ID of the run to delete

        Returns:
            Response dictionary
        """
        result = self._request("DELETE", f"/api/v1/runs/{run_id}")
        return cast(Dict[str, Any], result)
