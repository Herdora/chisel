import os
from pathlib import Path
from typing import Optional
import json as _json
import io as _io
import requests

from .constants import (
    KANDC_BACKEND_URL,
    KANDC_BACKEND_URL_ENV_KEY,
    KANDC_JOB_ID_ENV_KEY,
)
from .auth import _auth_service


class ArtifactDir:
    """Unified interface to persist run-time artifacts.

    - In local capture, files are written under ~/.kandc/runs/<app>/<job>/assets/.
    - In cloud runs, files are uploaded to backend via REST.
    """

    def __init__(self, base_subdir: str = "assets") -> None:
        self.base_subdir = base_subdir.strip("/") or "assets"
        self.job_id = os.environ.get(KANDC_JOB_ID_ENV_KEY)
        self.backend_url = os.environ.get(KANDC_BACKEND_URL_ENV_KEY) or KANDC_BACKEND_URL
        self.api_key: Optional[str] = None

        # Local capture detection: job id present and KANDC_TRACE_BASE_DIR points to local runs
        self.trace_base = os.environ.get("KANDC_TRACE_BASE_DIR")
        self.app_name = os.environ.get("KANDC_BACKEND_APP_NAME") or Path.cwd().name

        self.is_local = bool(self.trace_base and self.job_id)
        if self.is_local:
            self.local_root = (
                Path(self.trace_base).expanduser().resolve()
                / "runs"
                / self.app_name
                / self.job_id
                / self.base_subdir
            )
            self.local_root.mkdir(parents=True, exist_ok=True)
        else:
            self.local_root = None

    def _ensure_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        self.api_key = _auth_service.authenticate(self.backend_url)
        return self.api_key

    def _upload(self, rel_path: str, data: bytes, content_type: Optional[str] = None) -> None:
        api_key = self._ensure_api_key()
        url = f"{self.backend_url.rstrip('/')}/api/v1/jobs/{self.job_id}/assets/upload"
        headers = {"Authorization": f"Bearer {api_key}"}
        files = {
            "file": (
                Path(rel_path).name,
                _io.BytesIO(data),
                content_type or "application/octet-stream",
            )
        }
        resp = requests.post(
            url, headers=headers, files=files, params={"path": rel_path}, timeout=60
        )
        resp.raise_for_status()

    def write_bytes(self, path: str, content: bytes, content_type: Optional[str] = None) -> Path:
        rel = path.lstrip("./")
        if self.is_local:
            assert self.local_root is not None
            out = (self.local_root / rel).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                f.write(content)
            return out
        else:
            if not self.job_id:
                raise RuntimeError("KANDC_JOB_ID not set; cannot upload assets")
            self._upload(rel, content, content_type)
            return Path(rel)

    def write_text(self, path: str, text: str) -> Path:
        return self.write_bytes(path, text.encode("utf-8"), content_type="text/plain")

    def save_json(self, path: str, obj) -> Path:
        return self.write_text(path, _json.dumps(obj, ensure_ascii=False, indent=2))
