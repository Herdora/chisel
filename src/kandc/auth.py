from pathlib import Path
from typing import Optional, Dict, Any
import os
import requests
import json
import webbrowser
import time
from .constants import KANDC_BACKEND_URL


class AuthService:
    def __init__(self):
        self.config_dir = Path.home() / ".kandc"
        self.config_path = self.config_dir / "credentials.json"
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        # Check for new directory structure first
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}

        # Check for legacy single-file format and migrate
        if self.config_dir.exists() and self.config_dir.is_file():
            try:
                with open(self.config_dir, "r") as f:
                    config = json.load(f)
                print(f"🔄 Migrating config from legacy format {self.config_dir}")

                # Save in new format (this will handle the directory creation)
                temp_config = self.config
                self.config = config
                self._save_config()

                print(f"✅ Migration complete! Config now stored in {self.config_path}")

                return config
            except (json.JSONDecodeError, IOError):
                print(f"⚠️ Failed to migrate legacy config from {self.config_dir}")
                return {}

        return {}

    def _save_config(self) -> None:
        # Handle potential conflicts with existing .kandc file
        if self.config_dir.exists() and self.config_dir.is_file():
            # Backup existing .kandc file and convert to directory structure
            backup_path = self.config_dir.with_suffix(".kandc.backup")
            print(f"⚠️ Found existing .kandc file, backing up to {backup_path}")
            self.config_dir.rename(backup_path)

        # Create .kandc directory
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Ensure .kandc directory has proper permissions
        os.chmod(self.config_dir, 0o700)

        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        # Set restrictive permissions on credentials file
        os.chmod(self.config_path, 0o600)

    def get_api_key(self) -> Optional[str]:
        return self.config.get("api_key")

    def set_api_key(self, api_key: str, email: Optional[str] = None) -> None:
        self.config["api_key"] = api_key
        if email:
            self.config["email"] = email
        self._save_config()

    def get_email(self) -> Optional[str]:
        return self.config.get("email")

    def clear(self) -> None:
        if self.config_path.exists():
            self.config_path.unlink()
            self.config = {}
            print(f"🗑️  Cleared credentials from {self.config_path}")

        # Also handle legacy single-file format
        if self.config_dir.exists() and self.config_dir.is_file():
            self.config_dir.unlink()
            print(f"🗑️  Removed legacy config file {self.config_dir}")

    def is_authenticated(self) -> bool:
        return bool(self.get_api_key())

    def validate_api_key(self, api_key: str, backend_url: str) -> tuple[bool, Optional[str]]:
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(
                f"{backend_url}/api/v1/auth/validate", headers=headers, timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                return True, data.get("email")
            else:
                return False, None
        except requests.exceptions.RequestException:
            return True, None

    def authenticate(self, backend_url: str) -> Optional[str]:
        api_key = self.get_api_key()
        if api_key:
            is_valid, email = self.validate_api_key(api_key, backend_url)

            if is_valid:
                if email:
                    print(f"🔑 Authenticated as: {email}")
                return api_key
            else:
                api_key = None

        if not api_key:
            print("🔑 Starting authentication...")
            api_key = self._perform_browser_auth(backend_url)

        return api_key

    def _perform_browser_auth(self, backend_url: str) -> Optional[str]:
        try:
            response = requests.get(f"{backend_url}/api/v1/auth/init")
            if response.status_code == 200:
                auth_data = response.json()
                session_id = auth_data["session_id"]
                auth_url = auth_data["auth_url"]

                print("🌐 Opening browser for authentication...")
                print(f"📱 If browser doesn't open, visit: {auth_url}")
                webbrowser.open(auth_url)

                print("⏳ Waiting for authentication...")
                for i in range(60):
                    time.sleep(1)
                    check_response = requests.get(f"{backend_url}/api/v1/auth/check/{session_id}")
                    if check_response.status_code == 200:
                        check_data = check_response.json()
                        if check_data.get("authenticated"):
                            api_key = check_data.get("api_key")
                            email = check_data.get("email", "unknown")
                            print(f"✅ Authentication successful! Welcome {email}")

                            self.set_api_key(api_key, email)
                            return api_key

                    if i % 10 == 0 and i > 0:
                        print(f"⏳ Still waiting... ({60 - i}s remaining)")

                print("❌ Authentication timed out. Please try again.")
                return None
            else:
                print(f"❌ Failed to initiate authentication: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to backend: {e}")
            return None


_auth_service = AuthService()


def get_api_key() -> Optional[str]:
    return _auth_service.get_api_key()


def set_api_key(api_key: str, email: Optional[str] = None) -> None:
    _auth_service.set_api_key(api_key, email)


def get_email() -> Optional[str]:
    return _auth_service.get_email()


def clear_credentials() -> None:
    _auth_service.clear()


def is_authenticated() -> bool:
    return _auth_service.is_authenticated()


def authenticate(backend_url: str = None) -> Optional[str]:
    backend_url = backend_url or KANDC_BACKEND_URL
    return _auth_service.authenticate(backend_url)
