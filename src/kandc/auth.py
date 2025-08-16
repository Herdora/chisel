"""
Authentication management for Keys & Caches.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import webbrowser

from .api_client import APIClient, AuthenticationError


class AuthManager:
    """Manages authentication credentials and settings."""

    def __init__(self):
        self.config_dir = Path.home() / ".kandc"
        self.settings_file = self.config_dir / "settings.json"
        self.config_dir.mkdir(exist_ok=True)

    def load_settings(self) -> Dict[str, Any]:
        """Load settings from config file."""
        if not self.settings_file.exists():
            return {}

        try:
            with open(self.settings_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def save_settings(self, settings: Dict[str, Any]):
        """Save settings to config file."""
        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=2)
        except IOError as e:
            print(f"⚠️  Warning: Could not save settings: {e}")

    def get_api_key(self) -> Optional[str]:
        """Get stored API key."""
        settings = self.load_settings()
        return settings.get("api_key")

    def set_api_key(self, api_key: str, email: str = None):
        """Store API key and email."""
        settings = self.load_settings()
        settings["api_key"] = api_key
        if email:
            settings["email"] = email
        self.save_settings(settings)

    def clear_credentials(self):
        """Clear stored credentials."""
        settings = self.load_settings()
        settings.pop("api_key", None)
        settings.pop("email", None)
        self.save_settings(settings)

    def verify_api_key(self, api_key: str) -> bool:
        """Verify if API key is valid by testing with backend."""
        try:
            client = APIClient(api_key=api_key)
            # Test API key by trying to access a simple endpoint
            client._request("GET", "/")
            return True
        except AuthenticationError:
            return False
        except Exception:
            # Network errors or other issues - assume key might be valid
            return True

    def ensure_authenticated(self) -> APIClient:
        """
        Ensure user is authenticated and return API client.

        Returns:
            APIClient: Authenticated API client

        Raises:
            AuthenticationError: If authentication fails
        """
        # Try existing API key
        api_key = self.get_api_key()

        if api_key and self.verify_api_key(api_key):
            print(f"✅ Using existing authentication")
            settings = self.load_settings()
            if email := settings.get("email"):
                print(f"   Logged in as: {email}")
            return APIClient(api_key=api_key)

        # Need to authenticate
        print("🔐 Authentication required")

        if api_key:
            print("   Existing credentials are invalid")
            self.clear_credentials()

        # Authenticate via browser
        client = APIClient()  # No API key yet
        try:
            new_api_key = client.authenticate_with_browser()

            # Get user info to store email
            authenticated_client = APIClient(api_key=new_api_key)

            # Store credentials
            self.set_api_key(new_api_key)
            print(f"💾 Credentials saved to {self.settings_file}")

            return authenticated_client

        except AuthenticationError as e:
            raise AuthenticationError(f"Authentication failed: {e}")

    def get_dashboard_url(self, project_id: str = None, run_id: str = None) -> str:
        """Get dashboard URL."""
        api_key = self.get_api_key()
        if not api_key:
            raise AuthenticationError("Not authenticated")

        client = APIClient(api_key=api_key)
        return client.get_dashboard_url(project_id, run_id)

    def open_dashboard(self, project_id: str = None, run_id: str = None):
        """Open dashboard in browser."""
        try:
            url = self.get_dashboard_url(project_id, run_id)
            print(f"🌐 Opening dashboard: {url}")
            webbrowser.open(url)
        except Exception as e:
            print(f"⚠️  Could not open dashboard: {e}")


# Global auth manager instance
_auth_manager = AuthManager()


def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance."""
    return _auth_manager
